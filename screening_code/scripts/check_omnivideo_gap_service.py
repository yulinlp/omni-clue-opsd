#!/usr/bin/env python3
"""Read-only service inspection plus tiny synthetic text/audio/video requests."""

import argparse
import base64
import io
import json
import urllib.request
import wave
from pathlib import Path

from PIL import Image

from scripts.score_omnivideo_gap import (
    complete_option_logprobs,
    option_probabilities,
    post,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="http://gpu02:8091/v1/chat/completions")
    p.add_argument("--model", default="Qwen2.5-Omni-7B")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/share/home/ylhu/models/Qwen2.5-Omni-7B"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--require-mode", choices=["text", "split", "joint"], default="joint"
    )
    a = p.parse_args()
    vocabulary = json.loads((a.model_dir / "tokenizer.json").read_text())["model"][
        "vocab"
    ]
    ids = {k: vocabulary[k] for k in "ABCD"}
    url = a.endpoint.rsplit("/v1/", 1)[0] + "/v1/models"
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        url, timeout=10
    ) as r:
        models = json.load(r)
    if not any(m["id"] == a.model for m in models["data"]):
        raise ValueError("requested model not served")
    audio = io.BytesIO()
    with wave.open(audio, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(bytes(16000 * 2 * 2))
    picture = io.BytesIO()
    Image.new("RGB", (112, 112), "gray").save(picture, format="JPEG")
    audio_item = {
        "type": "audio_url",
        "audio_url": {
            "url": "data:audio/wav;base64,"
            + base64.b64encode(audio.getvalue()).decode()
        },
    }
    video_item = {
        "type": "video_url",
        "video_url": {
            "url": "data:video/jpeg;base64,"
            + ",".join([base64.b64encode(picture.getvalue()).decode()] * 4)
        },
    }
    template = json.loads((a.model_dir / "chat_template.json").read_text())[
        "chat_template"
    ]
    text = {"type": "text", "text": "Choose one letter from A, B, C, D."}
    results = {}
    for mode, content in [
        ("text", [text]),
        ("audio", [audio_item, text]),
        ("split", [video_item, audio_item, text]),
        ("joint", [video_item, audio_item, text]),
        ("joint_multi", [video_item, audio_item] * 3 + [text]),
        ("joint_multi_repeat", [video_item, audio_item] * 3 + [text]),
    ]:
        payload = dict(
            model=a.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1,
            temperature=1.0,
            seed=20260906,
            logprobs=True,
            top_logprobs=20,
            return_tokens_as_token_ids=True,
            allowed_token_ids=list(ids.values()),
            mm_processor_kwargs=dict(
                use_audio_in_video=mode.startswith("joint"),
                fps=2.0,
                min_pixels=3136,
                max_pixels=28672,
                size={"shortest_edge": 3136, "longest_edge": 28672},
                do_sample_frames=False,
            ),
        )
        if mode.startswith("joint"):
            payload["chat_template"] = template.replace(
                "<|audio_bos|><|AUDIO|><|audio_eos|>", ""
            )
        try:
            response = post(a.endpoint, payload, timeout=120)
            response, extra = complete_option_logprobs(
                a.endpoint, payload, response, ids
            )
            probabilities, _ = option_probabilities(response, ids)
            results[mode] = dict(
                ok=True, probabilities=probabilities, extra_option_requests=extra
            )
        except Exception as e:
            results[mode] = dict(ok=False, error=f"{type(e).__name__}: {e}")
        print(json.dumps({mode: results[mode]}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            dict(models=models, synthetic_engineering_probe=True, results=results),
            indent=2,
        )
        + "\n"
    )
    required = [a.require_mode]
    if a.require_mode == "joint":
        required += ["joint_multi", "joint_multi_repeat"]
    if not all(results[mode]["ok"] for mode in required):
        raise SystemExit("Service not ready for all AV protocols; inspect output")


if __name__ == "__main__":
    main()
