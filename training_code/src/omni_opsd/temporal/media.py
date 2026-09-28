"""Video probing, timestamp planning, and optional frame materialisation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np

from .evidence import EvidenceSpan, FrameInfo, SampleEvidence, sample_frames_from_spans


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    duration: float
    native_fps: float
    frame_count: int
    width: int
    height: int
    has_audio: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration": float(self.duration),
            "native_fps": float(self.native_fps),
            "frame_count": int(self.frame_count),
            "width": int(self.width),
            "height": int(self.height),
            "has_audio": self.has_audio,
        }


def _ffprobe_json(path: str) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def probe_video(path: str | Path) -> VideoMetadata:
    """Probe source metadata without decoding the full video."""

    video_path = str(Path(path).resolve())
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    try:
        import cv2
    except ImportError:
        cv2 = None
    fps = frame_count = width = height = 0
    if cv2 is not None:
        capture = cv2.VideoCapture(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()
    ffprobe = _ffprobe_json(video_path)
    streams = (ffprobe or {}).get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    # Login nodes in this repository do not always expose ffprobe, while the
    # official Qwen runtime already depends on PyAV.  Use it for the missing
    # stream-level metadata instead of silently labelling an audio video as
    # visual-only.  The cv2/ffprobe values remain authoritative when present.
    if audio_stream is None or not video_stream:
        try:
            import av
            container = av.open(video_path)
            av_video = next((stream for stream in container.streams if stream.type == "video"), None)
            av_audio = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream is None and av_audio is not None:
                audio_stream = {"codec_type": "audio"}
            if not video_stream and av_video is not None:
                av_rate = float(av_video.average_rate or av_video.base_rate or 0.0)
                av_frames = int(av_video.frames or 0)
                av_duration = float(av_video.duration * av_video.time_base) if av_video.duration and av_video.time_base else 0.0
                video_stream = {
                    "codec_type": "video",
                    "width": int(av_video.width or 0),
                    "height": int(av_video.height or 0),
                    "avg_frame_rate": f"{av_rate}/1" if av_rate else "0/1",
                    "nb_frames": av_frames,
                    "duration": av_duration,
                }
            container.close()
        except Exception:
            # Metadata probing still has the cv2 path above; if PyAV is not
            # installed, preserve the prior behavior and fail only when no
            # usable video metadata can be obtained.
            pass
    def _duration(value: Any) -> float:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else 0.0
        except (TypeError, ValueError):
            return 0.0
    duration = _duration(video_stream.get("duration", (ffprobe or {}).get("format", {}).get("duration")))
    if not duration and fps > 0 and frame_count > 0:
        duration = frame_count / fps
    if not fps:
        rate = str(video_stream.get("avg_frame_rate", "0/1"))
        try:
            numerator, denominator = rate.split("/", 1)
            fps = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            fps = 0.0
    frame_count = frame_count or int(video_stream.get("nb_frames") or 0)
    width = width or int(video_stream.get("width") or 0)
    height = height or int(video_stream.get("height") or 0)
    if duration <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"could not obtain usable video metadata: {video_path}")
    return VideoMetadata(video_path, duration, fps, frame_count, width, height, audio_stream is not None)


def _timestamp_to_frame_index(metadata: VideoMetadata, timestamp: float) -> int:
    return min(max(int(round(timestamp * metadata.native_fps)), 0), metadata.frame_count - 1)


class FrameCache:
    """Persistent JPEG cache for query-independent decoded frames.

    The cache key includes source stat information, timestamp, and requested
    decode policy.  Query-aware scores are never put in this cache; callers
    should include the question hash in a separate score key.
    """

    def __init__(self, root: str | Path | None, enabled: bool = True) -> None:
        self.root = Path(root).expanduser() if root else None
        self.enabled = bool(enabled and self.root)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, path: str, timestamp: float, policy: str) -> str:
        source = Path(path)
        stat = source.stat()
        raw = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{timestamp:.6f}|{policy}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_or_decode(
        self,
        metadata: VideoMetadata,
        timestamp: float,
        *,
        policy: str = "source-rgb",
    ) -> Path:
        """Decode one RGB frame and return a cached JPEG path."""

        key = self._key(metadata.path, timestamp, policy)
        path = (self.root / f"{key}.jpg") if self.enabled and self.root else None
        if path is not None and path.is_file():
            return path
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to materialize high-resolution frames") from exc
        capture = cv2.VideoCapture(metadata.path)
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"failed to decode frame at {timestamp:.6f}s: {metadata.path}")
        if path is None:
            # A deterministic temporary path is preferable to returning an
            # in-memory object because the official qwen-omni-utils accepts a
            # regular image path and can perform its own processor resize.
            path = Path(tempfile.gettempdir()) / "golden_temporal" / f"{key}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        # OpenCV selects its encoder from the final suffix; keep `.jpg` on
        # the temporary path rather than using `.jpg.tmp`.
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        if not cv2.imwrite(str(temporary), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"failed to encode cached frame: {temporary}")
        temporary.replace(path)
        return path


def _frame_vector(path: Path, size: tuple[int, int] = (32, 32)) -> np.ndarray:
    try:
        from PIL import Image
        image = Image.open(path).convert("RGB").resize(size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise RuntimeError(f"failed to read cached frame {path}") from exc
    return array.reshape(-1)


def _allocate_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    return sorted({int(round(index)) for index in np.linspace(0, total - 1, count)})


def select_keyframes(
    metadata: VideoMetadata,
    evidence: SampleEvidence,
    k: int,
    *,
    policy: str = "uniform",
    cache: FrameCache | None = None,
    question: str | None = None,
) -> tuple[list[FrameInfo], dict[str, Any]]:
    """Select K frames inside the evidence union using lightweight policies.

    ``query-aware`` is an explicit optional backend.  Without a configured
    image-text embedding model it falls back to a documented deterministic
    baseline instead of silently pretending that question relevance was used.
    """

    if k < 0:
        raise ValueError("keyframe count must be non-negative")
    if k == 0:
        return [], {"policy": policy, "selected": 0}
    candidates = sample_frames_from_spans(
        evidence,
        fps=max(1.0, min(metadata.native_fps, 4.0)),
        native_fps=metadata.native_fps,
        video_duration=metadata.duration,
        max_frames=max(2 * k, 2),
        source="keyframe-candidate",
    )
    if not candidates:
        return [], {"policy": policy, "selected": 0}
    if cache is None:
        cache = FrameCache(None, enabled=False)
    frame_paths = [cache.get_or_decode(metadata, frame.original_timestamp) for frame in candidates]
    requested_policy = policy.lower().replace("_", "-")
    effective_policy = requested_policy
    fallback_reason = None
    if requested_policy == "query-aware":
        # The repository has no declared CLIP/SigLIP dependency.  Keep this
        # branch observable so a later embedding backend can be plugged in.
        effective_policy = "uniform"
        fallback_reason = "no_query_embedding_backend_configured"
    if requested_policy in {"uniform", "center"}:
        if requested_policy == "uniform":
            selected_indices = _allocate_indices(len(candidates), min(k, len(candidates)))
        else:
            centers = [(span.start + span.end) / 2.0 for span in evidence.spans]
            ranked = sorted(
                range(len(candidates)),
                key=lambda index: min(abs(candidates[index].original_timestamp - center) for center in centers),
            )
            selected_indices = sorted(ranked[: min(k, len(candidates))])
    elif requested_policy in {"motion", "diversity", "hybrid", "query-motion"}:
        vectors = [_frame_vector(path) for path in frame_paths]
        if requested_policy == "motion":
            scores = [0.0] + [float(np.mean(np.abs(vectors[i] - vectors[i - 1]))) for i in range(1, len(vectors))]
            selected_indices = sorted(np.argsort(scores)[-min(k, len(candidates)):].tolist())
        else:
            selected_indices = [0]
            while len(selected_indices) < min(k, len(candidates)):
                best = max(
                    (index for index in range(len(candidates)) if index not in selected_indices),
                    key=lambda index: min(float(np.linalg.norm(vectors[index] - vectors[j])) for j in selected_indices),
                )
                selected_indices.append(best)
            selected_indices.sort()
        effective_policy = requested_policy
    else:
        raise ValueError(f"unknown keyframe policy: {policy}")
    selected = [candidates[index] for index in selected_indices]
    selected = [FrameInfo(item.original_timestamp, item.local_timestamp, "high-resolution-keyframe") for item in selected]
    return selected, {
        "policy": policy,
        "effective_policy": effective_policy,
        "selected": len(selected),
        "candidate_count": len(candidates),
        "fallback_reason": fallback_reason,
        "question_hash_used": bool(question and requested_policy == "query-aware" and fallback_reason is None),
    }


def frame_info_lists(frames: Iterable[FrameInfo]) -> list[dict[str, Any]]:
    return [frame.as_dict() for frame in frames]
