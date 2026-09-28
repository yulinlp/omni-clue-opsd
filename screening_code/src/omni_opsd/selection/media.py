"""PTS-based deterministic AV preparation; no answer-dependent selection."""

from __future__ import annotations

import base64
import io
import math
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image

AUDIO_UNCOVERED_MAX_FRACTION = 0.02
AUDIO_EDGE_UNCOVERED_MAX_SECONDS = 1.0


def frame_indices(timestamps, span, fps=2.0, max_frames=768):
    a, b = span
    times = np.asarray(timestamps)
    available = np.flatnonzero((times >= a) & (times <= b))
    if not len(available):
        raise ValueError("no source frames within evidence")
    n = min(len(available), max_frames, max(2, round((b - a) * fps / 2) * 2))
    return available[np.linspace(0, len(available) - 1, n).round().astype(int)].tolist()


def resize_shape(width, height, min_pixels=3136, max_pixels=28672):
    # Same 28-pixel spatial factor as Qwen2VL smart_resize.
    factor = 28
    h, w = (
        max(factor, round(height / factor) * factor),
        max(factor, round(width / factor) * factor),
    )
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = (
            max(factor, math.floor(height / beta / factor) * factor),
            max(factor, math.floor(width / beta / factor) * factor),
        )
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = (
            math.ceil(height * beta / factor) * factor,
            math.ceil(width * beta / factor) * factor,
        )
    if not min_pixels <= h * w <= max_pixels:
        raise ValueError("unsupported extreme video aspect ratio")
    return w, h


def decode_video(path, indices, cache=None, min_pixels=3136, max_pixels=28672):
    import av

    wanted = set(indices)
    frames = {}
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        for i in wanted:
            cached = cache / f"{i}.jpg"
            if cached.is_file():
                encoded = cached.read_bytes()
                frames[i] = (
                    encoded,
                    np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB")),
                )
    if len(frames) == len(wanted):
        return frames
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in wanted and i not in frames:
                img = frame.to_image()
                img = img.resize(
                    resize_shape(*img.size, min_pixels=min_pixels, max_pixels=max_pixels),
                    Image.Resampling.BICUBIC,
                )
                # Audit the JPEG-decoded pixels that the server will actually receive.
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                encoded = buf.getvalue()
                arr = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
                frames[i] = (encoded, arr)
                if cache:
                    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
                        f.write(encoded)
                        temp = f.name
                    os.replace(temp, cache / f"{i}.jpg")
            if len(frames) == len(wanted):
                break
    if len(frames) != len(wanted):
        raise ValueError("source video ended before planned frames")
    return frames


def decode_audio(path, duration):
    import av

    pcm = np.zeros(round(duration * 16000), dtype=np.float32)
    covered = np.zeros(len(pcm), dtype=bool)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    with av.open(path) as c:
        for frame in c.decode(audio=0):
            for audio in resampler.resample(frame):
                if audio.pts is None:
                    raise ValueError("resampled audio lacks timestamps")
                start = round(float(audio.pts * audio.time_base) * 16000)
                values = audio.to_ndarray().reshape(-1)
                lo, hi = max(0, start), min(len(pcm), start + len(values))
                if hi > lo:
                    pcm[lo:hi] = values[lo - start : hi - start]
                    covered[lo:hi] = True
        for audio in resampler.resample(None):
            start = round(float(audio.pts * audio.time_base) * 16000)
            values = audio.to_ndarray().reshape(-1)
            lo, hi = max(0, start), min(len(pcm), start + len(values))
            if hi > lo:
                pcm[lo:hi] = values[lo - start : hi - start]
                covered[lo:hi] = True
    if not covered.any():
        raise ValueError("no audio decoded")
    return pcm, covered


def wav_bytes(values):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((np.clip(values, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def validate_audio_coverage(
    covered,
    sample_rate=16000,
    max_fraction=AUDIO_UNCOVERED_MAX_FRACTION,
    max_edge_seconds=AUDIO_EDGE_UNCOVERED_MAX_SECONDS,
):
    """Allow bounded edge rounding, while rejecting interior audio holes."""
    covered = np.asarray(covered, dtype=bool)
    if not covered.size or not covered.any():
        raise ValueError("no audio decoded")
    present = np.flatnonzero(covered)
    first, last = int(present[0]), int(present[-1])
    interior_missing = int((~covered[first : last + 1]).sum())
    if interior_missing:
        raise ValueError(
            "audio has internal uncovered samples "
            f"({interior_missing / sample_rate:.3f}s)"
        )
    missing = 1.0 - float(covered.mean())
    edge_missing = first + len(covered) - last - 1
    allowed = max(
        math.ceil(max_fraction * len(covered)),
        math.ceil(max_edge_seconds * sample_rate),
    )
    if edge_missing > allowed:
        raise ValueError(
            f"audio has {missing:.2%} uncovered samples "
            f"({edge_missing / sample_rate:.3f}s at segment edges; "
            f"allowed up to {max_edge_seconds:g}s)"
        )
    return missing


def prepare(row, spans, audit, processor, cache_root=None):
    plans = [frame_indices(audit["timestamps"], s) for s in spans]
    cache = (
        Path(cache_root) / ("media_v1_" + audit["media_sha256"]) if cache_root else None
    )
    frames = decode_video(row["video_path"], {i for plan in plans for i in plan}, cache)
    audio_cache = cache / f"audio_{row['duration']}.npz" if cache else None
    if audio_cache and audio_cache.is_file():
        with np.load(audio_cache) as saved:
            pcm, covered = saved["pcm"], saved["covered"]
    else:
        pcm, covered = decode_audio(row["video_path"], row["duration"])
        if audio_cache:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
                np.savez(f, pcm=pcm, covered=covered)
                temp = f.name
            os.replace(temp, audio_cache)
    content, videos, audios = [], [], []
    for span, plan in zip(spans, plans):
        arrays = np.stack([frames[i][1] for i in plan])
        visual = processor.video_processor(
            videos=[arrays],
            return_tensors="pt",
            size={"shortest_edge": 3136, "longest_edge": 28672},
            do_sample_frames=False,
            device="cpu",
        )
        grid = visual["video_grid_thw"][0].tolist()
        tokens = int(np.prod(grid) // processor.video_processor.merge_size**2)
        if not 3136 <= grid[1] * grid[2] * 14**2 <= 28672:
            raise ValueError("processor changed the requested pixel budget")
        a, b = (round(t * 16000) for t in span)
        audio = pcm[a:b]
        missing = validate_audio_coverage(covered[a:b])
        video_url = "data:video/jpeg;base64," + ",".join(
            base64.b64encode(frames[i][0]).decode() for i in plan
        )
        audio_url = (
            "data:audio/wav;base64," + base64.b64encode(wav_bytes(audio)).decode()
        )
        content.extend(
            [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ]
        )
        videos.append(
            dict(
                sampled_timestamps=[audit["timestamps"][i] for i in plan],
                decoder_total_frames=audit["full_unique_timestamps"],
                source_fps=audit["source_average_fps"],
                num_frames=len(plan),
                processor_grid_thw=grid,
                visual_tokens=tokens,
                encoded_width=int(arrays.shape[2]),
                encoded_height=int(arrays.shape[1]),
            )
        )
        audios.append(
            dict(samples=len(audio), sampling_rate=16000, uncovered_fraction=missing)
        )
    return content, dict(
        input_spans=spans,
        videos=videos,
        audios=audios,
        total_visual_tokens=sum(v["visual_tokens"] for v in videos),
        processor_audit="client official processor on transmitted JPEG pixels",
        server_processor_grids="not exposed by OpenAI API",
        media_sha256=audit["media_sha256"],
    )
