#!/usr/bin/env bash
# =============================================================================
# npu24-watch.sh —— 当前 npu24 作业 worker-1/2 的“双列 NPU 实时监控表”
# =============================================================================
# 作用（只读监控，不启动/不停止训练）：
#   每隔几秒通过 SSH 读取 worker-1 和 worker-2 每张卡的利用率、显存、功率、温度，
#   在终端里“原位刷新”成一张表格；同时从 worker-1 日志里取最新 loss/step 显示进度。
#
# 用法：
#   bash npu24-watch.sh [刷新秒数] [刷新次数]
#     刷新秒数：默认 2 秒；
#     刷新次数：默认 0 = 无限刷新（按 Ctrl-C 退出）；写 1 表示只打印一次快照。
#
# 注意：Ctrl-C 只是停止监控，不会影响训练进程。
# =============================================================================

set -uo pipefail   # -u 未定义变量报错；-o pipefail 管道失败即失败（这里故意不加 -e，
                   # 因为监控脚本要允许个别 SSH 失败时继续显示）

# ---- 参数解析与校验 ------------------------------------------------------------
interval="${1:-2}"   # 第 1 个参数：刷新间隔（秒）
count="${2:-0}"      # 第 2 个参数：刷新次数（0=无限）
# 正则校验：间隔必须是非负数字（可带小数）；次数必须是非负整数。
[[ "${interval}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "interval_seconds must be nonnegative" >&2; exit 2; }
[[ "${count}" =~ ^[0-9]+$ ]] || { echo "count must be a nonnegative integer" >&2; exit 2; }

# ---- 路径与临时目录 ------------------------------------------------------------
repo="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd"
run_root="${CLUE_RUN_ROOT:-${repo}/training_runs/worldsense_openqa_20260929}" # 日志所在目录
# 复用的“取数脚本”：在远端节点上运行，输出每张卡一行指标。
snapshot="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/npu-snapshot.sh"
# 临时目录：存放两台机器的取数结果；mktemp -d 生成唯一目录名。
tmpdir="$(mktemp -d /tmp/npu24-watch.XXXXXX)"

# ---- 终端处理：判断是否在真实终端里运行（决定要不要用“清屏/隐藏光标”控制码） ----
tty=0
[[ -t 1 ]] && tty=1   # -t 1 = 标准输出是终端

# cleanup：退出时清理临时目录，并把光标恢复显示（\033[?25h）。
cleanup() {
  rm -rf -- "${tmpdir}"
  [[ "${tty}" == 1 ]] && printf '\033[?25h\n'
}
# trap：注册“退出时执行 cleanup”；收到 Ctrl-C（INT）时以 130 退出（约定俗成）。
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---- fetch：在指定机器上跑取数脚本，把结果写入本地临时文件 ----------------------
# $1 = IP；$2 = 输出文件。
# ssh 参数：
#   -F /dev/null：忽略系统 ssh 配置（避免本机配置里的权限问题）；
#   -p 2222     ：ModelArts 容器 SSH 端口；
#   BatchMode=yes：只用密钥不交互；
#   ConnectTimeout=5：5 秒连不上就失败，避免卡住整张表；
#   取数脚本路径用单引号包住，在远端由 bash 执行。
fetch() {
  local ip="$1" output="$2"
  ssh -F /dev/null -p 2222 -o BatchMode=yes -o ConnectTimeout=5 \
    -o StrictHostKeyChecking=accept-new "${ip}" "bash '${snapshot}'" \
    > "${output}" 2> "${output}.err"
}

# ---- format_card：从取数结果里挑出某张卡，格式化成表格单元格 -------------------
# 取数脚本每行格式：npu0 功率 温度 AI利用率 显存利用率
# awk 按 $1 == "npuN" 匹配；printf 输出 "AI% / HBM% / W / C"。
# found=1 标记找到了；END 里若没找到则打印 n/a（例如该卡取数失败）。
format_card() {
  local file="$1" card="$2"
  awk -v key="npu${card}" '$1 == key {printf "%3s / %3s / %4s / %2s", $4, $5, $2, $3; found=1}
       END {if (!found) printf "n/a"}' "${file}"
}

# ---- progress：从 worker-1 日志里提取最新一条训练指标 ---------------------------
# 日志里的指标行长这样：{'loss': '1.956', ..., 'global_step/max_steps': '16/135', ..., 'elapsed_time': '1h 7m 30s'}
# 1) grep -o 只保留 {...} 部分，tail -1 取最后一条；
# 2) sed 把 loss/step/耗时抽出来，重排成易读格式。
progress() {
  local line
  line="$(grep -a -o "{'loss'[^}]*}" "${run_root}/worker-1.log" 2>/dev/null | tail -1)"
  if [[ -z "${line}" ]]; then
    printf '等待首个训练步…'   # 还没有 loss 行
    return
  fi
  printf '%s\n' "${line}" | sed -E \
    "s/^\{'loss': '([^']*)'.*'global_step\/max_steps': '([^']*)'.*'elapsed_time': '([^']*)'.*\}$/loss \1  step \2  elapsed \3/"
}

# ---- 主循环 --------------------------------------------------------------------
if [[ "${tty}" == 1 ]]; then printf '\033[?25l'; fi   # 隐藏光标，避免刷新时闪烁
n=0
while true; do
  # 两台机器“并行”取数：各自放后台（&），记下 PID。
  fetch 172.16.12.237 "${tmpdir}/w1" & pid1=$!   # worker-1
  fetch 172.16.12.70 "${tmpdir}/w2" & pid2=$!    # worker-2
  wait "${pid1}"; ok1=$?   # 等待并记录退出码（0=成功）
  wait "${pid2}"; ok2=$?

  # \033[H\033[J：光标移到左上角并清屏 —— 这就是“动态表格”的实现方式。
  if [[ "${tty}" == 1 ]]; then printf '\033[H\033[J'; fi
  # 表头：时间、刷新间隔、退出提示。
  printf 'npu24 WorldSense CLUE-OPSD (worker-1 + worker-2)  %s  刷新 %ss  Ctrl-C 退出\n' \
    "$(date +%H:%M:%S)" "${interval}"
  printf '%-6s | %-32s | %-32s\n' 'NPU' 'worker-1 (AI% / HBM% / W / C)' 'worker-2 (AI% / HBM% / W / C)'
  printf -- '-------+----------------------------------+----------------------------------\n'
  # 每张卡一行：左列 worker-1，右列 worker-2。
  for card in 0 1 2 3 4 5 6 7; do
    w1="$(format_card "${tmpdir}/w1" "${card}")"
    w2="$(format_card "${tmpdir}/w2" "${card}")"
    printf 'npu%-3s | %-32s | %-32s\n' "${card}" "${w1}" "${w2}"
  done
  printf -- '-------+----------------------------------+----------------------------------\n'
  # 底部显示训练进度（从日志读取）。
  printf '训练进度(rank0): %s\n' "$(progress)"
  # 若某台 SSH 失败，打印其错误首行，方便排查（例如机器重启/网络不通）。
  [[ "${ok1}" == 0 ]] || printf 'worker-1 SSH 错误: %s\n' "$(head -1 "${tmpdir}/w1.err")"
  [[ "${ok2}" == 0 ]] || printf 'worker-2 SSH 错误: %s\n' "$(head -1 "${tmpdir}/w2.err")"

  # 计数；到达指定次数就退出（若期间有 SSH 失败则返回码 1）。
  n=$((n + 1))
  if [[ "${count}" -gt 0 && "${n}" -ge "${count}" ]]; then
    [[ "${ok1}" == 0 && "${ok2}" == 0 ]] || exit 1
    break
  fi
  sleep "${interval}"   # 等待下一次刷新
done
