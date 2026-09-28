#!/usr/bin/env python3
"""Does Qwen2.5-Omni really handle audio longer than 300 seconds?

Counts the audio placeholder tokens the official processor emits for growing
audio lengths.  If the count saturates at 7,500 (300 s x 25/s) the frontend is
truncating; if it keeps growing linearly, long audio is fully encoded.
"""

from __future__ import annotations

import io
import sys
import time
import warnings
import wave

import numpy as np

warnings.filterwarnings("ignore")

MODEL = "/share/home/ylhu/models/Qwen2.5-Omni-7B"
AUDIO_TOKEN_ID = 151646


def waveform(duration_s: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(duration_s * 16000)) / 16000.0
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main() -> None:
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    print(f"{'duration':>9s} {'audio_tokens':>12s} {'expected(25/s)':>14s} {'ratio':>7s} {'seconds':>8s}", flush=True)
    for duration in (60, 200, 300, 400, 600, 900):
        started = time.time()
        out = proc(
            text="<|audio_bos|><|AUDIO|><|audio_eos|>transcribe",
            audio=[waveform(duration)],
            return_tensors="pt",
        )
        count = out["input_ids"][0].tolist().count(AUDIO_TOKEN_ID)
        print(
            f"{duration:>8.0f}s {count:>12d} {duration * 25:>14d} "
            f"{count / (duration * 25):>7.3f} {time.time() - started:>7.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
