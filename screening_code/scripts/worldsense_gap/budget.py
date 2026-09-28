#!/usr/bin/env python3
"""Dynamic multimedia budget for Qwen2.5-Omni-7B WorldSense screening.

Implements the approved plan (docs/WORLDSENSE_EVIDENCE_GAP_SCREENING.zh-CN.md
section 3.2):

    audio_tokens  = ceil(duration * 25)              # position_id_per_seconds=25
    visual_budget = min(24000, 32768 - 2048 - audio) # text reserve 2048
    frames        = min(floor(duration * 2), 300)    # 2 FPS target, 300 cap
    groups        = ceil(frames / 2)                 # temporal_patch_size=2
    per_group     = visual_budget // groups
    grid          = largest 28-aligned (h, w) with h*w <= per_group,
                    aspect-preserving and never above the source resolution

The client uses `max_pixels = per_group * 784` for the frame resize and sends
the frame count explicitly, so the server-side processor never has to guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CONTEXT = 32768
TEXT_RESERVE = 2048
AUDIO_TOKENS_PER_SECOND = 25
VISUAL_CAP = 24000
MAX_FRAMES = 300
TARGET_FPS = 2.0
PATCH = 28  # 14 x 14 patch with 2 x 2 merge
SAFETY = 512  # keep this much headroom below the context limit


@dataclass(frozen=True)
class Budget:
    duration: float
    audio_tokens: int
    visual_budget: int
    frames: int
    groups: int
    per_group: int
    grid_h: int
    grid_w: int
    max_pixels: int
    visual_tokens: int
    total_tokens: int
    adjusted: bool = False

    @property
    def fps(self) -> float:
        return self.frames / self.duration if self.duration > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "audio_tokens": self.audio_tokens,
            "visual_budget": self.visual_budget,
            "frames": self.frames,
            "groups": self.groups,
            "per_group": self.per_group,
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "max_pixels": self.max_pixels,
            "visual_tokens": self.visual_tokens,
            "total_tokens": self.total_tokens,
            "fps": round(self.fps, 4),
            "adjusted": self.adjusted,
        }


def _grid(per_group: int, src_w: int, src_h: int) -> tuple[int, int]:
    """Largest 28-aligned grid (cells) that fits per_group, source aspect kept.

    Aspect-preserving: h/w tracks the source ratio (rounded to cells), so the
    model never sees a distorted frame; the search walks h down and takes the
    first (largest) grid that fits the token budget and the source resolution.
    """

    base_h = max(1, round(src_h / PATCH))
    base_w = max(1, round(src_w / PATCH))
    best = (1, 1)
    # Walk the source cell grid down at a constant aspect ratio; take the
    # largest scale whose token count still fits per_group.
    for h in range(base_h, 0, -1):
        w = max(1, round(h * base_w / base_h))
        if h * w <= per_group and h * w > best[0] * best[1]:
            best = (h, w)
    for w in range(base_w, 0, -1):
        h = max(1, round(w * base_h / base_w))
        if h * w <= per_group and h * w > best[0] * best[1]:
            best = (h, w)
    return best


def compute_budget(duration: float, src_w: int, src_h: int) -> Budget:
    """Resolution first, then frames: the frame count is the budget knob.

    The per-frame resolution is fixed by the design target (visual_budget /
    target groups), then the number of sampled frames is reduced until the
    total fits the context.  Shrinking frames without fixing the resolution
    would just let the grid grow and save nothing.
    """

    if duration <= 0:
        raise ValueError("duration must be positive")
    if src_w <= 0 or src_h <= 0:
        raise ValueError("source resolution must be positive")
    audio = int(math.ceil(duration * AUDIO_TOKENS_PER_SECOND))
    # Reserve the safety headroom up front so the final total always fits.
    visual_budget = min(VISUAL_CAP, CONTEXT - TEXT_RESERVE - SAFETY - audio)
    if visual_budget < 2 * PATCH * PATCH:
        raise ValueError(f"audio alone leaves no visual budget (duration={duration})")
    frames = min(int(math.floor(duration * TARGET_FPS)), MAX_FRAMES)
    frames = max(2, frames - frames % 2)  # even frame count for the 2-frame merge
    groups = math.ceil(frames / 2)
    per_group = visual_budget // groups
    grid_h, grid_w = _grid(per_group, src_w, src_h)
    grid_tokens = grid_h * grid_w
    adjusted = False
    # Reduce the frame count (never the resolution) until the total fits.
    max_groups = visual_budget // grid_tokens
    if max_groups < groups:
        adjusted = True
        groups = max(1, max_groups)
        frames = 2 * groups
    visual_tokens = groups * grid_tokens
    total = visual_tokens + audio + TEXT_RESERVE
    if total > CONTEXT - SAFETY:
        raise ValueError(f"cannot fit {duration}s video into the context budget")
    return Budget(
        duration=duration,
        audio_tokens=audio,
        visual_budget=visual_budget,
        frames=frames,
        groups=groups,
        per_group=per_group,
        grid_h=grid_h,
        grid_w=grid_w,
        max_pixels=per_group * PATCH * PATCH,
        visual_tokens=visual_tokens,
        total_tokens=total,
        adjusted=adjusted,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Show the dynamic budget for a duration")
    p.add_argument("duration", type=float)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    a = p.parse_args()
    print(compute_budget(a.duration, a.width, a.height).as_dict())
