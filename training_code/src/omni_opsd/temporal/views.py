"""Build controlled visual/audio/text views for E0--E18 and GEM-OPSD."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from .dataset import Sample
from .evidence import EvidenceSpan, FrameInfo, SampleEvidence, expand_evidence
from .media import FrameCache, VideoMetadata, frame_info_lists, probe_video, select_keyframes


RESOLUTION_PRESETS = {
    # These are processor max-pixel policies, not claims about actual output
    # dimensions.  The backend records actual grid H/W for every sample.
    # Qwen2.5-Omni's official video utility has a 128 visual-token/frame
    # lower bound (128 * 28^2 pixels), so low must not be below 100,352.
    "low": 102_400,
    "medium": 204_800,
    "high": 409_600,
    "very_high": 819_200,
}


@dataclass
class VideoInputSpec:
    path: str
    start: float
    end: float
    requested_fps: float
    effective_fps: float
    max_pixels: int
    max_frames: int
    label: str
    audio: bool = False

    def descriptor(self) -> dict[str, Any]:
        return {
            "video": self.path,
            "video_start": self.start,
            "video_end": self.end,
            "fps": self.effective_fps,
            "min_frames": 2,
            "max_frames": self.max_frames,
            "max_pixels": self.max_pixels,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "requested_fps": self.requested_fps,
            "effective_fps": self.effective_fps,
            "max_pixels": self.max_pixels,
            "max_frames": self.max_frames,
            "label": self.label,
            "audio": self.audio,
        }


@dataclass
class ImageInputSpec:
    frame: FrameInfo
    max_pixels: int
    path: str | None = None
    label: str = "high-resolution-keyframe"

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.frame.as_dict(),
            "max_pixels": self.max_pixels,
            "path": self.path,
            "label": self.label,
        }


@dataclass
class ViewSpec:
    name: str
    videos: list[VideoInputSpec] = field(default_factory=list)
    images: list[ImageInputSpec] = field(default_factory=list)
    visual_spans: list[EvidenceSpan] = field(default_factory=list)
    audio_spans: list[EvidenceSpan] = field(default_factory=list)
    audio_enabled: bool = False
    audio_mode: str = "none"
    subtitle_enabled: bool = False
    timestamp_mode: str = "none"
    timestamp_hint_spans: list[EvidenceSpan] = field(default_factory=list)
    prompt_note: str = ""
    teacher_role: str | None = None
    keyframe_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def actual_input_spans(self) -> list[EvidenceSpan]:
        return self.visual_spans

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "videos": [video.as_dict() for video in self.videos],
            "images": [image.as_dict() for image in self.images],
            "visual_spans": [span.as_list() for span in self.visual_spans],
            "audio_spans": [span.as_list() for span in self.audio_spans],
            "audio_enabled": self.audio_enabled,
            "audio_mode": self.audio_mode,
            "subtitle_enabled": self.subtitle_enabled,
            "timestamp_mode": self.timestamp_mode,
            "timestamp_hint_spans": [span.as_list() for span in self.timestamp_hint_spans],
            "prompt_note": self.prompt_note,
            "teacher_role": self.teacher_role,
            "keyframe_metadata": self.keyframe_metadata,
        }


def _get(config: dict[str, Any], *path: str, default: Any = None) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def resolution_pixels(config: dict[str, Any], policy: str | None = None) -> int:
    selected = policy or _get(config, "video", "resolution_policy", default="medium")
    value = _get(config, "video", "max_pixels", default=None)
    if value is not None:
        return int(value)
    if isinstance(selected, int):
        return int(selected)
    if selected not in RESOLUTION_PRESETS:
        raise ValueError(f"unknown resolution policy {selected!r}; choose {sorted(RESOLUTION_PRESETS)}")
    return RESOLUTION_PRESETS[selected]


def _span_union(spans: list[EvidenceSpan]) -> list[EvidenceSpan]:
    # Import lazily to keep the top-level planner dependency-light.
    from .evidence import merge_overlapping_spans
    return merge_overlapping_spans(spans)


def _make_videos(
    sample: Sample,
    metadata: VideoMetadata,
    spans: list[EvidenceSpan],
    *,
    fps: float,
    max_pixels: int,
    max_frames: int,
    label: str,
    audio: bool = False,
) -> list[VideoInputSpec]:
    return [
        VideoInputSpec(
            path=sample.video_path,
            start=max(0.0, float(span.start)),
            end=min(metadata.duration, float(span.end)),
            requested_fps=float(fps),
            effective_fps=min(float(fps), metadata.native_fps),
            max_pixels=int(max_pixels),
            max_frames=int(max_frames),
            label=f"{label}_{index + 1}",
            audio=audio,
        )
        for index, span in enumerate(spans)
        if span.end > span.start
    ]


def _full_span(metadata: VideoMetadata) -> list[EvidenceSpan]:
    return [EvidenceSpan(0.0, metadata.duration)]


def _gold_spans(sample: Sample, metadata: VideoMetadata) -> list[EvidenceSpan]:
    if not sample.evidence.spans:
        raise ValueError(f"sample {sample.sample_id} has no golden evidence")
    clamped = [
        EvidenceSpan(max(0.0, span.start), min(metadata.duration, span.end))
        for span in sample.evidence.spans
        if min(metadata.duration, span.end) > max(0.0, span.start)
    ]
    return _span_union(clamped)


def _apply_multi_span_policy(spans: list[EvidenceSpan], config: dict[str, Any]) -> list[EvidenceSpan]:
    """Keep separate evidence descriptors or make one continuous interval."""

    policy = str(_get(config, "evidence", "multi_span_policy", default="separate")).lower()
    if policy in {"separate", "concat", "independent"}:
        return list(spans)
    if policy in {"continuous", "bounding", "bounding_interval"}:
        if not spans:
            return []
        return [EvidenceSpan(spans[0].start, spans[-1].end)]
    raise ValueError(f"unknown evidence.multi_span_policy: {policy}")


def _audio_spans(sample: Sample, metadata: VideoMetadata, visual_spans: list[EvidenceSpan], config: dict[str, Any]) -> list[EvidenceSpan]:
    policy = _get(config, "audio", "span_policy", default="same_as_visual")
    if policy in {"full", "full_audio"}:
        return _full_span(metadata)
    if policy in {"same_as_visual", "visual"}:
        return list(visual_spans)
    if policy in {"golden", "evidence"}:
        return _gold_spans(sample, metadata)
    if policy in {"halo", "golden_halo"}:
        before = float(_get(config, "audio", "halo_before", default=_get(config, "evidence", "halo_before", default=0.0)))
        after = float(_get(config, "audio", "halo_after", default=_get(config, "evidence", "halo_after", default=before)))
        return expand_evidence(SampleEvidence(visual_spans, metadata.duration), before, after).spans
    raise ValueError(f"unknown audio span policy: {policy}")


def _base_view(
    name: str,
    sample: Sample,
    metadata: VideoMetadata,
    spans: list[EvidenceSpan],
    config: dict[str, Any],
    *,
    fps: float | None = None,
    resolution: str | None = None,
    audio: bool | None = None,
    timestamp_mode: str = "none",
    label: str = "evidence",
    teacher_role: str | None = None,
    prompt_note: str = "",
    timestamp_hint_spans: list[EvidenceSpan] | None = None,
) -> ViewSpec:
    use_audio = bool(_get(config, "audio", "enabled", default=False) if audio is None else audio)
    visual_fps = float(fps if fps is not None else _get(config, "video", "fps", default=1.0))
    max_pixels = resolution_pixels(config, resolution)
    max_frames = int(_get(config, "video", "max_frames", default=768))
    actual_audio_spans = _audio_spans(sample, metadata, spans, config) if use_audio else []
    audio_same_as_video = use_audio and actual_audio_spans == spans
    videos = _make_videos(
        sample, metadata, spans, fps=visual_fps, max_pixels=max_pixels,
        max_frames=max_frames, label=label, audio=audio_same_as_video,
    )
    return ViewSpec(
        name=name,
        videos=videos,
        visual_spans=spans,
        audio_spans=actual_audio_spans,
        audio_enabled=use_audio,
        audio_mode="native_video" if audio_same_as_video else ("explicit" if use_audio else "none"),
        subtitle_enabled=bool(_get(config, "subtitle", "enabled", default=False)),
        timestamp_mode=timestamp_mode,
        timestamp_hint_spans=list(timestamp_hint_spans or []),
        prompt_note=prompt_note,
        teacher_role=teacher_role,
    )


def build_gem_teacher_views(sample: Sample, metadata: VideoMetadata, config: dict[str, Any]) -> list[ViewSpec]:
    """Construct complementary GT-conditioned views for the improved teacher.

    The teacher may inspect these views during training/probing, while the
    deployable student receives only the configured full/global view.  No view
    uses the answer or any answer-derived frame score.
    """

    gold = _apply_multi_span_policy(_gold_spans(sample, metadata), config)
    halo_before = float(_get(config, "teacher", "halo_before", default=3.0))
    halo_after = float(_get(config, "teacher", "halo_after", default=3.0))
    context = expand_evidence(SampleEvidence(gold, metadata.duration), halo_before, halo_after).spans
    dense_fps = float(_get(config, "teacher", "dense_fps", default=8.0))
    context_fps = float(_get(config, "teacher", "context_fps", default=2.0))
    detail_fps = float(_get(config, "teacher", "detail_fps", default=2.0))
    medium = resolution_pixels(config, _get(config, "teacher", "dense_resolution", default="medium"))
    context_res = _get(config, "teacher", "context_resolution", default="low")
    detail_res = _get(config, "teacher", "detail_resolution", default="high")
    views = [
        _base_view(
            "tight_dense", sample, metadata, gold, config, fps=dense_fps,
            resolution=_get(config, "teacher", "dense_resolution", default="medium"),
            timestamp_mode="absolute", label="tight-evidence", teacher_role="temporal",
            prompt_note="This is the tight ground-truth evidence view for action and ordering.",
        ),
        _base_view(
            "context_halo", sample, metadata, context, config, fps=context_fps,
            resolution=context_res, timestamp_mode="absolute", label="context-halo", teacher_role="context",
            prompt_note="This view adds local before/after context around the ground-truth evidence.",
        ),
    ]
    if int(_get(config, "teacher", "num_hr_keyframes", default=4)) > 0:
        cache_root = _get(config, "cache", "frame_dir", default=".cache/golden_temporal/frames")
        frames, key_meta = select_keyframes(
            metadata, SampleEvidence(gold, metadata.duration),
            int(_get(config, "teacher", "num_hr_keyframes", default=4)),
            policy=_get(config, "teacher", "keyframe_policy", default="uniform"),
            cache=FrameCache(cache_root, enabled=bool(_get(config, "cache", "enabled", default=True))),
            question=sample.question,
        )
        detail = _base_view(
            "spatial_detail", sample, metadata, gold, config, fps=detail_fps,
            resolution=detail_res, timestamp_mode="per_frame", label="detail-video", teacher_role="spatial",
            prompt_note="This view is complemented by high-resolution stills selected only inside gold evidence.",
        )
        detail.images = [ImageInputSpec(frame, resolution_pixels(config, "very_high")) for frame in frames]
        detail.keyframe_metadata = key_meta
        views.append(detail)
    # Do not ask qwen-omni-utils to decode a nonexistent audio stream.  The
    # omission is explicit in the recorded teacher protocol; E13 itself
    # still fails loudly when a user explicitly requests audio on such data.
    if bool(_get(config, "teacher", "include_omni", default=True)) and metadata.has_audio:
        omni = _base_view(
            "omni_context", sample, metadata, context, config,
            fps=context_fps, resolution=context_res, audio=True, timestamp_mode="absolute",
            label="omni-context", teacher_role="audio-context",
            prompt_note="This view tests audio/subtitle evidence with a wider temporal window.",
        )
        views.append(omni)
    return views


def build_experiment_views(sample: Sample, config: dict[str, Any]) -> tuple[VideoMetadata, list[ViewSpec]]:
    metadata = probe_video(sample.video_path)
    mode = str(_get(config, "experiment", "mode", default="baseline")).lower()
    full = _full_span(metadata)
    gold = _apply_multi_span_policy(_gold_spans(sample, metadata), config) if mode not in {"baseline", "full"} else []
    fps = float(_get(config, "video", "fps", default=1.0))
    resolution = _get(config, "video", "resolution_policy", default="medium")
    if mode in {"baseline", "full", "e0"}:
        return metadata, [_base_view("full", sample, metadata, full, config, fps=fps, resolution=resolution, label="full")]
    if mode in {"full_timestamp", "e1"}:
        return metadata, [_base_view("full_timestamp_hint", sample, metadata, full, config, fps=fps, resolution=resolution, timestamp_mode="none", label="full", timestamp_hint_spans=gold)]
    if mode in {"golden", "golden_clip", "e2"}:
        return metadata, [_base_view("golden", sample, metadata, gold, config, fps=fps, resolution=resolution, label="golden")]
    if mode in {"golden_timestamp", "e3"}:
        return metadata, [_base_view("golden_timestamp", sample, metadata, gold, config, fps=fps, resolution=resolution, timestamp_mode="absolute", label="golden")]
    if mode in {"halo", "golden_halo", "e4"}:
        before = float(_get(config, "evidence", "halo_before", default=0.0))
        after = float(_get(config, "evidence", "halo_after", default=before))
        expanded = expand_evidence(SampleEvidence(gold, metadata.duration), before, after).spans
        return metadata, [_base_view("golden_halo", sample, metadata, expanded, config, fps=fps, resolution=resolution, timestamp_mode="absolute", label="golden-halo")]
    if mode in {"global_coarse_golden_dense", "e5"}:
        global_fps = float(_get(config, "global", "fps", default=0.25))
        global_res = _get(config, "global", "resolution_policy", default="low")
        dense = _base_view("global_coarse", sample, metadata, full, config, fps=global_fps, resolution=global_res, label="global-coarse")
        evidence_view = _base_view("golden_dense", sample, metadata, gold, config, fps=fps, resolution=resolution, timestamp_mode="absolute", label="golden-dense")
        return metadata, [ViewSpec(
            name="global_coarse_golden_dense",
            videos=dense.videos + evidence_view.videos,
            visual_spans=dense.visual_spans + evidence_view.visual_spans,
            timestamp_mode="absolute",
            prompt_note=(
                "Video 1 provides a coarse overview of the full video. "
                "The following video stream(s) provide the ground-truth evidence "
                "segment(s) most relevant to the question."
            ),
        )]
    if mode in {"fps", "resolution", "iso_budget", "e7", "e8", "e9"}:
        return metadata, [_base_view(mode, sample, metadata, gold, config, fps=fps, resolution=resolution, timestamp_mode="absolute", label="golden")]
    if mode in {"dense_hr", "e10"}:
        view = _base_view("dense_video_hr_images", sample, metadata, gold, config, fps=fps, resolution=resolution, timestamp_mode="per_frame", label="dense-evidence")
        k = int(_get(config, "keyframes", "num", default=0))
        if k > 0:
            cache_root = _get(config, "cache", "frame_dir", default=".cache/golden_temporal/frames")
            frames, key_meta = select_keyframes(
                metadata, SampleEvidence(gold, metadata.duration), k,
                policy=_get(config, "keyframes", "policy", default="uniform"),
                cache=FrameCache(cache_root, enabled=bool(_get(config, "cache", "enabled", default=True))),
                question=sample.question,
            )
            image_res = _get(config, "keyframes", "resolution_policy", default="very_high")
            view.images = [ImageInputSpec(frame, resolution_pixels(config, image_res)) for frame in frames]
            view.keyframe_metadata = key_meta
        return metadata, [view]
    if mode in {"audio", "e13", "audio_subtitle"}:
        return metadata, [_base_view("audio_factorial", sample, metadata, gold, config, fps=fps, resolution=resolution, audio=bool(_get(config, "audio", "enabled", default=True)), timestamp_mode="absolute", label="golden-audio")]
    if mode in {"gem_teacher", "teacher", "gem-opsd"}:
        return metadata, build_gem_teacher_views(sample, metadata, config)
    raise ValueError(f"unknown experiment mode: {mode}")


def frame_timestamps_for_view(view: ViewSpec, metadata: VideoMetadata) -> list[FrameInfo]:
    from .evidence import sample_frames_from_spans
    result: list[FrameInfo] = []
    for video in view.videos:
        result.extend(sample_frames_from_spans(
            [EvidenceSpan(video.start, video.end)],
            video.requested_fps,
            native_fps=metadata.native_fps,
            video_duration=metadata.duration,
            max_frames=video.max_frames,
            source=video.label,
        ))
    return result


def view_cache_key(sample: Sample, view: ViewSpec, processor_version: str = "unknown") -> str:
    question_hash = hashlib.sha256(sample.question.encode("utf-8")).hexdigest() if "query" in view.name else ""
    payload = {
        "video_id": sample.sample_id,
        "video_path": sample.video_path,
        "view": view.as_dict(),
        "processor_version": processor_version,
        "question_hash": question_hash,
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def view_debug_dict(view: ViewSpec, metadata: VideoMetadata) -> dict[str, Any]:
    frames = frame_timestamps_for_view(view, metadata)
    return {
        "view": view.as_dict(),
        "video_metadata": metadata.as_dict(),
        "sampled_frames": frame_info_lists(frames),
        "requested_fps": [video.requested_fps for video in view.videos],
        "effective_fps": [video.effective_fps for video in view.videos],
        "processed_resolution_policy": [video.max_pixels for video in view.videos],
    }
