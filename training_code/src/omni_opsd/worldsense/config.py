"""Runtime configuration for one evidence-localization episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Budgets and hard limits for the tool loop.

    Every value is validated by the harness, never trusted to the model.
    """

    max_turns: int = 12
    max_inspect: int = 6
    max_new_tokens: int = 1024
    temperature: float = 0.2
    max_media_in_context: int = 2

    max_intervals: int = 4
    min_interval_seconds: float = 1.0
    max_total_seconds: float = 90.0
    max_total_ratio: float = 0.5
    # Tasks whose evidence legitimately spans the video get a wider total
    # interval budget instead of being trimmed to half the duration.
    wide_budget_tasks: tuple[str, ...] = (
        "Temporal Localization",
        "Event Sorting",
        "Causal Reasoning",
        "Temporal Prediction",
        "Future Prediction",
        "Event Recognition",
        "Summarization",
        "Scene Transformation Detection",
        "Video Emotions",
        "Emotion Change",
    )
    wide_total_ratio: float = 0.8

    min_view_seconds: float = 1.0
    max_view_seconds: float = 120.0
    fps_min: float = 0.5
    fps_max: float = 8.0
    max_pixels_min: int = 3_136
    max_pixels_max: int = 313_600
    default_max_pixels: int = 156_800
    max_frames: int = 300

    # The agent must first watch the whole video at a coarse rate and build a
    # rough timeline before it may inspect anything.  The survey is injected by
    # the harness, does not consume the inspect budget, and is downgraded to
    # ``full_scan_fallback_fps`` when the first generation runs out of memory.
    force_full_scan: bool = True
    full_scan_fps: float = 2.0
    full_scan_fallback_fps: float = 1.0
    full_scan_max_pixels: int = 31_360
    # Long videos are sampled at ``full_scan_fps`` up to this many frames.  With
    # 600 frames a 300 s video keeps the full 2 fps; longer videos are thinned.
    full_scan_max_frames: int = 600
    # Optional chunked survey: >0 splits the full-video survey into chunks of
    # this many seconds so long videos keep the requested frame rate.  Chunking
    # costs extra context, so it is opt-in.
    survey_chunk_seconds: float = 0.0
    # API backend: the survey proxy keeps the full 2 fps up to this many frames
    # (1312 frames = 656 s, i.e. every WorldSense video).  The byte budget may
    # still thin longer clips.
    api_survey_max_frames: int = 1312

    # The dataset synopsis is shown as a weak hint, but the agent must still
    # watch the whole video first and verify everything it claims.
    include_synopsis: bool = True
    include_caption_in_metadata: bool = True

    # Hard cap on how many times an out-of-memory generate may be retried with
    # a cheaper view before the episode fails.
    max_oom_retries: int = 2

    # Evidence must be confirmed by watching: a submit is rejected until at
    # least one inspect view has been seen.
    require_inspect_before_submit: bool = True

    # API backend only: first ask the model for a timestamped full-video
    # caption, then use that caption to plan the inspect views.
    caption_stage: bool = True
    caption_max_tokens: int = 8_000
    # A caption must contain timestamps and at least this many characters,
    # otherwise it is treated as degenerate and retried once.
    caption_min_chars: int = 200

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def check_interval_budget(self, duration: float, task_type: str | None = None) -> float:
        ratio = self.max_total_ratio
        if task_type and task_type in self.wide_budget_tasks:
            ratio = max(ratio, self.wide_total_ratio)
        return min(self.max_total_seconds, ratio * float(duration))
