#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# prepare_worldsense_clue_opsd_npu.py —— 一次性数据准备：SFT 数据 → CLUE-OPSD 数据
# =============================================================================
# 背景（SFT 与 CLUE-OPSD 的数据差别）：
#   - SFT 行：学生看到“全片视频 + 问题”，监督目标是 assistant 的标准答案；
#   - CLUE-OPSD 行：学生仍看“全片”，但**教师**改看“标注的证据时间段”（clue_intervals），
#     并删除 assistant 标答（学生自己生成 completion，再和教师分布对比学习）。
#
# 本脚本做的事（每行）：
#   1) 从 annotation（人工/模型标注文件）里按 case_id 找到该题的证据区间；
#   2) 用 dynamic_clue_budget_for 计算“所有证据段共享一份”的帧数/分辨率预算；
#   3) 把总帧数按时长比例分配到各个证据区间（allocate_frames）；
#   4) 生成教师侧视频描述 teacher_videos（每段一个结构化的“虚拟切片”描述，
#      不实际剪视频文件，训练时由读取器按 video_start/video_end 抽取）；
#   5) 生成学生 prompt 与教师 prompt（可选：教师 prompt 里带标准答案，
#      对应 --answer-conditioned 的“答案特权教师”）；
#   6) 输出新的 JSONL（默认 1453 行）。
#
# 用法（见文档 §5）：
#   PYTHONPATH=training_code/src python training_code/scripts/prepare_worldsense_clue_opsd_npu.py \
#     --sft <已适配本机路径的 sft jsonl> \
#     --annotation data/annotation/merged.evidence.jsonl \
#     --output training_runs/.../data/clue_opsd.jsonl
# =============================================================================

# from __future__ import annotations：让类型注解（list[...] 等）在旧 Python 也能用。
from __future__ import annotations

import argparse   # 解析命令行参数
import json       # 读写 JSONL
import re         # 正则表达式（提取 <answer>X</answer>、统计选项行）
from pathlib import Path   # 面向对象的路径操作

# 动态预算函数：计算“证据段共享预算”的帧数、分辨率、token 占用等。
from omni_opsd.data.dynamic_budget import dynamic_clue_budget_for


def allocate_frames(spans: list[list[float]], total: int) -> list[int]:
    """把总帧数按时长比例分配到多个证据区间，且每段都是偶数帧。

    为什么要偶数：Qwen-Omni 会把相邻 2 帧合并成 1 个时间组（temporal_patch_size=2），
    每段帧数必须是偶数，合并才成立。
    分配策略：
      - 先给每段保底 2 帧（result 初始化为 1，最后乘 2）；
      - 剩余帧数按“各段应得帧数(raw) − 已分配帧数”的差值，每次补给差值最大的段，
        这样分配结果接近“按时长比例”，又保证每段都是偶数。
    """
    pairs = total // 2                       # 总“帧对”数（每 2 帧一组）
    durations = [end - start for start, end in spans]   # 每段时长
    # raw[i]：第 i 段按比例应得的帧对数（小数）。
    raw = [pairs * duration / sum(durations) for duration in durations]
    result = [1] * len(spans)                # 每段先分 1 个帧对（=2 帧）
    for _ in range(pairs - len(spans)):      # 还剩多少个帧对没分
        # 找出“应得 − 已得”差距最大的段，给它加 1 个帧对。
        index = max(range(len(spans)), key=lambda i: raw[i] - result[i])
        result[index] += 1
    return [2 * value for value in result]   # 帧对数 ×2 = 帧数（保证偶数）


def main() -> None:
    # ---- 命令行参数 ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)          # 输入 SFT JSONL
    parser.add_argument("--annotation", type=Path, required=True)   # 证据区间标注 JSONL
    parser.add_argument("--output", type=Path, required=True)       # 输出 CLUE JSONL
    # 可选开关：加上它 → 教师 prompt 里带标准答案（答案特权教师），并给题目换成
    # “只回一个字母”的直答 prompt。
    prompt_mode = parser.add_mutually_exclusive_group()
    prompt_mode.add_argument("--answer-conditioned", action="store_true",
                             help="legacy answer-conditioned direct-letter prompts")
    prompt_mode.add_argument("--thinking-answer-conditioned", action="store_true",
                             help="reasoning prompts, full gold option in teacher, and SFT reasoning target")
    args = parser.parse_args()

    # ---- 读取标注：question_id → 该题的标注行（只保留 status=submitted 的） ----
    # 这里用到了海象运算符 :=（Python 3.8+）：
    #   先把 json.loads(line) 赋值给 row，再用 row.get(...) 做条件判断。
    annotations = {
        row["question_id"]: row
        for line in args.annotation.open(encoding="utf-8")
        if (row := json.loads(line)).get("status") == "submitted"
    }
    # 输出目录不存在就创建（parents=True 连父目录一起建）。
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0   # 统计写出了多少行
    # 同时打开输入（读）和输出（写）。
    with args.sft.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            sft = json.loads(line)                 # 一条 SFT 数据
            case_id = sft["case_id"]               # 题号（与标注的 question_id 对应）
            full = sft["videos"][0]                # 学生看的全片视频描述
            annotation = annotations[case_id]      # 该题的证据区间标注
            # 视频总时长 = 结束 - 开始。
            duration = float(full["video_end"]) - float(full["video_start"])
            # 把标注区间裁剪到 [0, duration]，防止越界。
            spans = [[max(0.0, float(a)), min(duration, float(b))]
                     for a, b in annotation["clue_intervals"]]
            if not spans or any(b <= a for a, b in spans):
                raise ValueError(f"bad clue intervals: {case_id}: {spans}")

            # ---- 计算教师侧“共享预算” -------------------------------------
            # 多个证据段被当作一个整体：共享视觉 token 预算、音频预算按证据总时长算。
            # 输入需要 duration 和源分辨率（从学生动态预算里取 source_resolution）。
            budget = dynamic_clue_budget_for({
                "duration": duration,
                "metadata": {"resolution": sft["dynamic_student_budget"]["source_resolution"]},
            }, spans)
            # 把总帧数按时长比例分给各段（每段偶数帧）。
            frame_caps = allocate_frames(spans, int(budget["nframes"]))

            original_prompt = sft["messages"][0]["content"]   # 原始学生 prompt（含 <video> 与问题）
            assistant_target = None
            target_truncated = False
            if args.answer_conditioned or args.thinking_answer_conditioned:
                # ---- 答案特权模式：从 assistant 回复里提取标准答案字母 ----
                # 正则匹配 <answer>D</answer> 这类格式。
                match = re.search(r"<answer>\s*([A-D])\s*</answer>", sft["messages"][-1]["content"])
                if not match:
                    raise ValueError(f"missing gold answer: {case_id}")
                gold = match.group(1)      # 例如 "D"
                # 原 prompt 里有一句“Briefly analyze ...”的分析指令。
                marker = "\nBriefly analyze the video and audio evidence"
                if marker not in original_prompt:
                    raise ValueError(f"unexpected SFT prompt: {case_id}")
                qa_prompt = original_prompt.split(marker, 1)[0]   # 只保留问题和选项部分
                # 数一数有几个选项行（A. / B. / C. / D. 开头）。
                option_count = len(re.findall(r"^[A-D]\. ", qa_prompt, flags=re.MULTILINE))
                if option_count not in (3, 4) or gold not in "ABCD"[:option_count]:
                    raise ValueError(f"invalid options or gold answer: {case_id}")
                teacher_qa = "\n".join(["<video>"] * len(spans)) + "\n" + qa_prompt.split("\n", 1)[1]
                if args.thinking_answer_conditioned:
                    # 学生保持原 SFT 的“分析 + <answer>字母</answer>”指令；
                    # 仅教师得到正确字母和对应的完整选项文字。
                    option_match = re.search(rf"^{gold}\.\s*(.+)$", qa_prompt, flags=re.MULTILINE)
                    if not option_match:
                        raise ValueError(f"missing gold option text: {case_id}")
                    student_prompt = original_prompt
                    teacher_prompt = (teacher_qa + f"\nVerified correct answer: {gold}. "
                                      + option_match.group(1).strip() + original_prompt[len(qa_prompt):])
                    # 原 SFT 数据有少量超过 120 词的分析。离线分支的标准回答
                    # 必须服从同一个 prompt，保留最多 120 词的分析和原答案标签。
                    analysis = sft["messages"][-1]["content"].split("<answer>", 1)[0].strip()
                    words = analysis.split()
                    if len(words) > 120:
                        candidate = " ".join(words[:120])
                        sentence_end = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
                        analysis = candidate[:sentence_end + 1] if sentence_end > len(candidate) * 0.6 else candidate.rstrip(",;:") + "."
                        target_truncated = True
                    assistant_target = analysis + f"\n<answer>{gold}</answer>"
                else:
                    instruction = "Reply with exactly one uppercase option letter. No explanation or tags."
                    student_prompt = qa_prompt + "\n" + instruction
                    teacher_prompt = teacher_qa + f"\nVerified correct option: {gold}.\n" + instruction
            else:
                # ---- 非答案特权模式（普通 CLUE-OPSD）----------------------
                gold = None
                student_prompt = original_prompt
                # 教师 prompt：证据段占位符 + 原问题（不含答案）。
                teacher_prompt = ("\n".join(["<video>"] * len(spans))
                                  + "\n" + original_prompt.split("\n", 1)[1])
            # ---- 构造教师侧视频描述（每个证据段一个“结构化切片”）----------
            # 注意：这里不真正剪出视频文件，只写清楚“从哪个时间点到哪个时间点、
            # 抽多少帧、缩放到多大”。训练时由 qwen_omni_utils 按这些字段解码。
            teacher_videos = [{
                "video": full["video"],             # 同一个源视频文件
                "video_start": start,               # 该证据段开始秒数
                "video_end": end,                   # 该证据段结束秒数
                "nframes": cap,                     # 该段分配到的帧数（偶数）
                "resized_height": budget["resized_height"],   # 统一缩放高度
                "resized_width": budget["resized_width"],     # 统一缩放宽度
                "min_pixels": 3136,                 # 每帧像素下限（与全局口径一致）
                "max_pixels": budget["resized_height"] * budget["resized_width"],
            } for (start, end), cap in zip(spans, frame_caps)]

            # ---- 组装审计字段（记录教师侧用了什么口径，方便复查） ----------
            contract = dict(sft["sampling_contract"])   # 复制学生侧采样契约
            contract.update({
                "answer_label_in_model_row": bool(args.answer_conditioned or args.thinking_answer_conditioned),
                "teacher_view": "annotated-clue-intervals",  # 教师看的是标注证据段
                "teacher_frame_cap_total": sum(frame_caps),  # 教师总帧数
                "teacher_frame_caps": frame_caps,            # 每段帧数
                "teacher_dynamic_video_budget": budget["visual_budget_tokens"],
                "teacher_dynamic_audio_budget": budget["audio_budget_tokens"],
                "teacher_dynamic_reserved_tokens": budget["reserved_tokens"],
            })
            # ---- 最终输出行 -------------------------------------------------
            result = {
                "messages": ([{"role": "user", "content": student_prompt},
                              {"role": "assistant", "content": assistant_target}]
                             if args.thinking_answer_conditioned else
                             [{"role": "user", "content": student_prompt}]),
                "videos": sft["videos"],                                    # 学生看全片
                "teacher_prompt": teacher_prompt,                           # 教师 prompt（可含答案）
                "teacher_videos": teacher_videos,                           # 教师看证据段
                "clue_intervals": spans,                                    # 证据区间（秒）
                "prompt_id": case_id,
                "case_id": case_id,
                "video_id": sft["video_id"],
                "response_format": "reasoning",
                "experiment_arm": "clue_opsd",                              # 标记训练臂
                "sampling_contract": contract,
                "dynamic_student_budget": sft["dynamic_student_budget"],    # 学生预算原样保留
                "dynamic_teacher_budget": budget,                           # 教师预算（新算的）
                "supervision_contract": {
                    # 监督契约：说明这是哪种 CLUE-OPSD、答案是否进入教师 prompt 等。
                    "kind": ("gold-answer-conditioned-clue-thinking-gkd" if args.thinking_answer_conditioned
                             else "gold-answer-conditioned-clue-gkd" if args.answer_conditioned
                             else "reasoning-clue-on-policy-self-distillation"),
                    "gold_answer_in_student_prompt": False,
                    "gold_answer_in_student_target": args.thinking_answer_conditioned,
                    "gold_answer_in_teacher_prompt": args.answer_conditioned or args.thinking_answer_conditioned,
                    "gold_answer_in_auxiliary_loss": args.answer_conditioned or args.thinking_answer_conditioned,
                    "teacher_view": "annotated-clue-intervals",
                    "teacher_completion": "student_on_policy_completion_token_ids",
                    "analysis_target_truncated_to_120_words": target_truncated,
                },
            }
            if args.answer_conditioned or args.thinking_answer_conditioned:
                result["gold_answer"] = gold
                # ms-swift 的 RLHF 数据预处理会保留名为 `solution` 的字段，而会丢掉
                # 任意自定义字段（如 gold_answer）。所以这里额外写一份 solution，
                # 保证训练器能拿到标准答案用于辅助损失/统计。
                result["solution"] = gold
            # 写出一行（中文不转义）。
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} CLUE-OPSD rows to {args.output}")


# 直接运行时才执行 main()。
if __name__ == "__main__":
    main()
