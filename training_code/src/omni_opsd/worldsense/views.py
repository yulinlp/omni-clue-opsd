"""Harness-side rendering of agent media requests into Qwen-Omni chat entries.

The model never touches the processor: it emits ``ViewRequest`` parameters and
this module turns them into structured video/audio descriptors, enforces the
harness limits, and normalises mixed-modality conversations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import AgentConfig

MEDIA_KEY = "_worldsense_view"
MIN_PIXELS = 3_136
MODALITIES = ("av", "video", "audio")


@dataclass(frozen=True)
class ViewRequest:
    start: float
    end: float
    fps: float
    max_pixels: int
    modality: str
    focus: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def text(self) -> str:
        return (
            f"Requested view: {self.start:.2f}s-{self.end:.2f}s, fps={self.fps:g}, "
            f"max_pixels={self.max_pixels}, modality={self.modality}.\n"
            f"Focus: {self.focus or 'inspect this view for evidence of the asked fact.'}"
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_view(raw: dict[str, Any], duration: float, config: AgentConfig) -> ViewRequest:
    """Clamp one inspect request; raise ValueError when it is unusable."""

    if duration <= 0:
        raise ValueError("video duration must be positive")
    requested_start = float(raw.get("start", 0.0))
    requested_end = float(raw.get("end", 0.0))
    requested_len = requested_end - requested_start
    start = _clamp(requested_start, 0.0, duration)
    end = _clamp(requested_end, 0.0, duration)
    if end - start < config.min_view_seconds:
        # The window is empty, too short, or lies past the video end (a common
        # case: the model asks for [85, 92] on an 87.8s video, which clamps to
        # a 3ms sliver).  First try to extend forward; if the end is already at
        # the video end, slide the window back inside the video instead of
        # rejecting it -- a hard rejection here made the model retry the
        # identical request until the turn budget was gone.
        end = min(duration, start + config.min_view_seconds)
        if end - start < config.min_view_seconds - 1e-6:
            end = duration
            start = max(0.0, end - config.min_view_seconds)
        if end - start < config.min_view_seconds - 1e-6:
            raise ValueError(
                f"inspect window is empty: the video is {duration:.2f}s long but you asked for "
                f"{requested_start:.2f}s-{requested_end:.2f}s. Request a window inside "
                f"0-{duration:.2f}s, for example {max(0.0, duration - 15):.2f}-{duration:.2f}s"
            )
    if end - start > config.max_view_seconds:
        end = start + config.max_view_seconds
    fps = _clamp(float(raw.get("fps", 2.0)), config.fps_min, config.fps_max)
    max_pixels = int(
        _clamp(
            float(raw.get("max_pixels", config.default_max_pixels)),
            config.max_pixels_min,
            config.max_pixels_max,
        )
    )
    modality = str(raw.get("modality", "av")).strip().lower()
    if modality not in MODALITIES:
        raise ValueError(f"unsupported modality: {modality!r}")
    focus = str(raw.get("focus") or "").strip()
    return ViewRequest(
        start=round(start, 3),
        end=round(end, 3),
        fps=round(fps, 3),
        max_pixels=max_pixels,
        modality=modality,
        focus=focus[:800],
    )


def make_media_message(view: ViewRequest, video_path: str, duration: float) -> dict[str, Any]:
    return {
        "role": "user",
        "content": view.text(),
        MEDIA_KEY: {
            **view.as_dict(),
            "video_path": str(video_path),
            "duration_s": float(duration),
            "max_frames": 300,
        },
    }


def media_stub(view: dict[str, Any]) -> str:
    return (
        f"[earlier view {view['start']:.2f}s-{view['end']:.2f}s, fps={view['fps']:g}, "
        f"modality={view['modality']}: media released from context]"
    )


def trim_media_context(messages: list[dict[str, Any]], max_media: int) -> list[dict[str, Any]]:
    """Keep only the newest ``max_media`` media messages and stub the rest."""

    media_indices = [i for i, m in enumerate(messages) if m.get(MEDIA_KEY)]
    if len(media_indices) <= max_media:
        return messages
    for index in media_indices[: len(media_indices) - max_media]:
        view = messages[index][MEDIA_KEY]
        messages[index] = {
            "role": messages[index]["role"],
            "content": messages[index].get("content", "") + "\n" + media_stub(view),
        }
    return messages


def render_conversation(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Return (chat messages, use_audio_in_video) for the Qwen processor.

    When every media view is ``av`` we use the interleaved native layout; any
    silent/audio-only view forces explicit audio descriptors so each view keeps
    its requested modality.
    """

    views = [m[MEDIA_KEY] for m in messages if m.get(MEDIA_KEY)]
    interleaved = bool(views) and all(v["modality"] == "av" for v in views)
    rendered: list[dict[str, Any]] = []
    for message in messages:
        view = message.get(MEDIA_KEY)
        if not view:
            rendered.append({"role": message["role"], "content": message.get("content", "")})
            continue
        entries: list[dict[str, Any]] = []
        if view["modality"] in ("av", "video"):
            entries.append(
                {
                    "type": "video",
                    "video": view["video_path"],
                    "video_start": float(view["start"]),
                    "video_end": float(view["end"]),
                    "fps": float(view["fps"]),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": int(view["max_pixels"]),
                    "max_frames": int(view.get("max_frames", 300)),
                }
            )
        if view["modality"] == "audio" or (view["modality"] == "av" and not interleaved):
            entries.append(
                {
                    "type": "audio",
                    "audio": view["video_path"],
                    "audio_start": float(view["start"]),
                    "audio_end": float(view["end"]),
                }
            )
        entries.append({"type": "text", "text": message.get("content", "")})
        rendered.append({"role": message["role"], "content": entries})
    return rendered, interleaved
