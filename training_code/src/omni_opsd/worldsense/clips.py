"""Render agent media views into small MP4 clips for the cloud API backend.

The OpenAI-compatible Qwen3.8-Omni-Flash endpoint accepts a video only as a
public URL or as a base64 data URI (< 10 MB encoded).  Local cluster files are
not reachable from the cloud, so every view is transcoded with ffmpeg into a
small proxy clip whose frame rate and scale encode the agent's requested
sampling, then cached on disk.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFMPEG_CANDIDATES = (
    "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg",
    "/usr/bin/ffmpeg",
)


def ffprobe_path() -> str:
    found = shutil.which("ffprobe")
    if found:
        return found
    for candidate in (
        "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffprobe",
        "/usr/bin/ffprobe",
    ):
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("ffprobe not found; set PATH or install ffmpeg")


def audio_stream_info(path: str | Path, *, timeout: int = 60) -> dict[str, object]:
    """Return codec/sample_rate/channels of the first audio stream (if any)."""

    probe = ffprobe_path()
    command = [
        probe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        return {"present": False, "error": exc.stderr.decode(errors="replace")[-200:]}
    except subprocess.TimeoutExpired:
        return {"present": False, "error": "ffprobe timeout"}
    payload = json.loads(result.stdout.decode() or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return {"present": False}
    stream = streams[0]
    return {
        "present": True,
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def verify_audio_or_raise(path: str | Path, *, context: str = "") -> dict[str, object]:
    """Hard-fail when an audio-bearing view lost its audio.

    A silent proxy would silently degrade captions for speech/music questions,
    so a missing audio stream is treated as an explicit error instead of a
    warning.
    """

    info = audio_stream_info(path)
    if not info.get("present"):
        raise RuntimeError(
            f"audio stream missing after transcode{(' (' + context + ')') if context else ''}: {path} "
            f"| probe={info}"
        )
    if int(info.get("sample_rate") or 0) <= 0 or int(info.get("channels") or 0) <= 0:
        raise RuntimeError(f"audio stream looks invalid{(' (' + context + ')') if context else ''}: {path} | {info}")
    return info


def ffmpeg_path() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in FFMPEG_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("ffmpeg not found; set PATH or install ffmpeg")


@dataclass(frozen=True)
class ClipRequest:
    video_path: str
    start: float
    end: float
    fps: float
    max_pixels: int
    with_audio: bool
    tag: str = ""
    policy: str = "size_aware"

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "video": str(Path(self.video_path).resolve()),
                "start": round(self.start, 3),
                "end": round(self.end, 3),
                "fps": round(self.fps, 3),
                "max_pixels": int(self.max_pixels),
                "with_audio": bool(self.with_audio),
                "tag": self.tag,
                "policy": self.policy,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def target_scale(max_pixels: int, width: int, height: int) -> tuple[int, int]:
    """Largest even-sized frame whose area does not exceed ``max_pixels``."""

    if width <= 0 or height <= 0:
        width, height = 640, 360
    scale = math.sqrt(max(1, int(max_pixels)) / float(width * height))
    new_width = max(2, int(width * scale))
    new_height = max(2, int(height * scale))
    new_width -= new_width % 2
    new_height -= new_height % 2
    return max(2, new_width), max(2, new_height)


def _even(value: int) -> int:
    value = max(2, int(value))
    return value - value % 2


def plan_attempts(
    request: ClipRequest,
    source_size: tuple[int, int],
    *,
    policy: str | None = None,
) -> list[dict[str, object]]:
    """Ordered ffmpeg attempts for one view.

    ``size_aware`` maximises quality under the byte budget: it first tries the
    original resolution at the requested frame rate, then the same resolution at
    1 fps, then progressively smaller frames.  ``fixed`` keeps the historical
    behaviour of scaling straight to ``request.max_pixels``.  Audio is encoded
    at 96 kbps for the first attempts because audio tokens are billed by
    duration, so the bitrate is free.
    """

    width, height = source_size
    if width <= 0 or height <= 0:
        width, height = 640, 360
    chosen = str(policy or request.policy or "size_aware").lower()
    attempts: list[dict[str, object]] = []
    if chosen == "fixed":
        base_w, base_h = target_scale(request.max_pixels, width, height)
        attempts.append({"fps": request.fps, "width": base_w, "height": base_h,
                         "crf": "30", "audio_kbps": "96", "label": "fixed"})
        attempts.append({"fps": min(request.fps, 2.0), "width": base_w, "height": base_h,
                         "crf": "32", "audio_kbps": "64", "label": "fixed_lower_fps"})
        attempts.append({"fps": min(request.fps, 1.0), "width": _even(base_w // 2), "height": _even(base_h // 2),
                         "crf": "34", "audio_kbps": "64", "label": "fixed_half_scale"})
        return attempts

    # Respect the caller's spatial request but never upscale beyond the source:
    # the rendered frame is min(source pixels, requested max_pixels).
    requested_px = min(int(width) * int(height), max(1, int(request.max_pixels)))
    base_w, base_h = target_scale(requested_px, width, height)
    attempts.append({"fps": request.fps, "width": base_w, "height": base_h,
                     "crf": "30", "audio_kbps": "96", "label": "requested"})
    if request.fps > 1.0:
        attempts.append({"fps": 1.0, "width": base_w, "height": base_h,
                         "crf": "30", "audio_kbps": "96", "label": "requested_1fps"})
    for label, pixels, crf, kbps in (
        ("scale_156800", 156_800, "30", "96"),
        ("scale_31360", 31_360, "32", "64"),
        ("scale_15680", 15_680, "34", "64"),
    ):
        if pixels >= requested_px:
            continue  # would not be smaller than the requested frame
        scaled_w, scaled_h = target_scale(pixels, width, height)
        attempts.append({"fps": min(request.fps, 2.0), "width": scaled_w, "height": scaled_h,
                         "crf": crf, "audio_kbps": kbps, "label": label})
    # Final safety floor so a very long clip can never exceed the byte budget.
    floor_w, floor_h = target_scale(3_136, width, height)
    if (floor_w, floor_h) != (base_w, base_h):
        attempts.append({"fps": min(request.fps, 1.0), "width": floor_w, "height": floor_h,
                         "crf": "36", "audio_kbps": "48", "label": "safety_floor"})
    return attempts


def render_clip(
    request: ClipRequest,
    cache_dir: str | Path,
    *,
    source_size: tuple[int, int] = (640, 360),
    max_bytes: int = 7_000_000,
    policy: str | None = None,
    timeout: int = 300,
) -> Path:
    """Transcode one view into a cached MP4 clip under the size budget."""

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = request.cache_key()
    target = cache / f"{key}.mp4"
    meta = target.with_suffix(".json")
    if target.is_file() and meta.is_file():
        return target

    # Concurrent workers may request the same clip; serialise on a per-key lock
    # and publish the finished file with an atomic rename so nobody ever reads
    # or overwrites a half-written mp4.
    lock_path = cache / f"{key}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if target.is_file() and meta.is_file():
                return target
            return _render_clip_locked(request, cache, key, target, meta, source_size, max_bytes, policy, timeout)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _render_clip_locked(
    request: ClipRequest,
    cache: Path,
    key: str,
    target: Path,
    meta: Path,
    source_size: tuple[int, int],
    max_bytes: int,
    policy: str | None,
    timeout: int,
) -> Path:
    ffmpeg = ffmpeg_path()
    attempts = plan_attempts(request, source_size, policy=policy)
    last_error = ""
    tmp = cache / f"{key}.{os.getpid()}.tmp.mp4"
    for attempt in attempts:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, request.start):.3f}",
            "-to",
            f"{max(0.0, request.end):.3f}",
            "-i",
            str(request.video_path),
            "-r",
            f"{max(0.5, float(attempt['fps'])):g}",
            "-vf",
            f"scale={attempt['width']}:{attempt['height']}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            attempt["crf"],
            "-pix_fmt",
            "yuv420p",
        ]
        if request.with_audio:
            command += ["-c:a", "aac", "-b:a", f"{attempt['audio_kbps']}k"]
        else:
            command += ["-an"]
        command += ["-movflags", "+faststart", str(tmp)]
        try:
            subprocess.run(command, check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            last_error = exc.stderr.decode(errors="replace")[-400:]
            continue
        except subprocess.TimeoutExpired:
            last_error = "ffmpeg timeout"
            continue
        if request.with_audio:
            verify_audio_or_raise(
                tmp,
                context=f"{Path(request.video_path).name} {request.start:.1f}-{request.end:.1f}s",
            )
        size = tmp.stat().st_size
        if size <= max_bytes or attempt is attempts[-1]:
            os.replace(tmp, target)
            meta.write_text(
                json.dumps(
                    {
                        "video": request.video_path,
                        "start": request.start,
                        "end": request.end,
                        "fps": attempt["fps"],
                        "max_pixels": request.max_pixels,
                        "with_audio": request.with_audio,
                        "width": attempt["width"],
                        "height": attempt["height"],
                        "crf": attempt["crf"],
                        "policy": str(policy or request.policy),
                        "attempt_label": attempt["label"],
                        "bytes": size,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            return target
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f"clip render failed for {request.video_path}: {last_error}")


def clip_to_data_uri(path: str | Path) -> str:
    import base64

    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:;base64,{payload}"


def audio_to_data_uri(path: str | Path, fmt: str = "mp3") -> str:
    import base64

    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:;base64,{payload}"


def render_audio_clip(
    request: ClipRequest,
    cache_dir: str | Path,
    *,
    timeout: int = 180,
) -> Path:
    """Extract the requested audio range as a compact mp3."""

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = request.cache_key()
    target = cache / f"{key}.mp3"
    if target.is_file():
        return target
    lock_path = cache / f"{key}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if target.is_file():
                return target
            return _render_audio_locked(request, cache, key, target, timeout)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _render_audio_locked(request: ClipRequest, cache: Path, key: str, target: Path, timeout: int) -> Path:
    ffmpeg = ffmpeg_path()
    tmp = cache / f"{key}.{os.getpid()}.tmp.mp3"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, request.start):.3f}",
        "-to",
        f"{max(0.0, request.end):.3f}",
        "-i",
        str(request.video_path),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(tmp),
    ]
    subprocess.run(command, check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    verify_audio_or_raise(tmp, context=f"audio-only {Path(request.video_path).name}")
    os.replace(tmp, target)
    return target
