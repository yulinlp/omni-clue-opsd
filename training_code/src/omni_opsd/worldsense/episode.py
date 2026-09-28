"""The evidence-localization tool loop for one WorldSense question."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .clients import OmniClient
from .config import AgentConfig
from .media import MediaInfo
from .prompts import build_initial_user_prompt, build_system_prompt
from .protocol import parse_action
from .schema import Interval, QuestionRecord
from .views import MEDIA_KEY, ViewRequest, clamp_view, make_media_message, trim_media_context

OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
    "cudaerrormemoryallocation",
    "cuda out of memory",
)

MEDIA_MARKERS = (
    "video_reader_backend",
    "blockingioerror",
    "swscaler",
    "failed initializing scaling graph",
    "decord error",
    "resource temporarily unavailable",
)


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in OOM_MARKERS)


def _is_media_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in MEDIA_MARKERS)


def _stub_oldest_media(messages: list[dict[str, Any]]) -> bool:
    """Release the oldest media message so a retry decodes less."""

    for message in messages:
        view = message.get(MEDIA_KEY)
        if not view:
            continue
        message.pop(MEDIA_KEY, None)
        message["content"] = (
            str(message.get("content", ""))
            + "\n[media released to recover from a decoding error]"
        )
        return True
    return False


def _downgrade_scan_fps(messages: list[dict[str, Any]], fallback_fps: float) -> bool:
    """Re-sample the injected full-video survey at a lower frame rate."""

    changed = False
    for message in messages:
        view = message.get(MEDIA_KEY)
        if not view or not view.get("scan"):
            continue
        if float(view.get("fps", 0.0)) <= fallback_fps:
            continue
        view["fps"] = float(fallback_fps)
        # Frame count, not fps, drives the visual token count: halving fps keeps
        # the same 600 frames and therefore the same memory.  Halve the frames
        # (down to a floor) so the retry actually needs less memory.
        current_frames = int(view.get("max_frames", 600) or 600)
        new_frames = max(120, current_frames // 2)
        view["max_frames"] = new_frames
        view["downgraded"] = True
        view["downgrade_reason"] = f"oom_survey_frames_{current_frames}_to_{new_frames}"
        message["content"] = (
            f"Requested view: 0.00s-{float(view['end']):.2f}s, fps={float(fallback_fps):g}, "
            f"max_pixels={int(view['max_pixels'])}, modality={view['modality']} "
            f"(the survey was re-sampled at a lower frame rate to fit in memory).\n"
            f"Focus: {view.get('focus', '')}"
        )
        changed = True
    return changed


def _downgrade_latest_view(messages: list[dict[str, Any]], config: AgentConfig) -> bool:
    """Halve the newest inspect view's pixels/fps after an OOM."""

    for message in reversed(messages):
        view = message.get(MEDIA_KEY)
        if not view or view.get("scan"):
            continue
        changed = False
        pixels = int(view.get("max_pixels", config.default_max_pixels))
        if pixels > config.max_pixels_min:
            view["max_pixels"] = max(config.max_pixels_min, pixels // 2)
            changed = True
        fps = float(view.get("fps", 2.0))
        if fps > config.fps_min:
            view["fps"] = max(config.fps_min, round(fps / 2, 3))
            changed = True
        if changed:
            view["downgraded"] = True
            view["downgrade_reason"] = "oom_view_resample"
            message["content"] = (
                f"Requested view: {float(view['start']):.2f}s-{float(view['end']):.2f}s, "
                f"fps={float(view['fps']):g}, max_pixels={int(view['max_pixels'])}, "
                f"modality={view['modality']} (re-sampled at lower cost to fit in memory).\n"
                f"Focus: {view.get('focus', '')}"
            )
        return changed
    return False


def _generate_with_oom_recovery(
    client: OmniClient,
    messages: list[dict[str, Any]],
    config: AgentConfig,
    scan_state: dict[str, Any],
) -> str:
    """Generate one reply, recovering from CUDA OOM by re-sampling media.

    Recovery order: first lower the injected full-video survey to the fallback
    fps, then halve the newest inspect view's pixels and fps.  Each downgrade
    is tried once and counted in ``scan_state['oom_retries']``.
    """

    attempts = 0
    media_retry_done = False
    while True:
        try:
            return client.generate(
                messages,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise unless recoverable
            if _is_media_error(exc) and not media_retry_done:
                # decord failed on this file and the torchvision/ffmpeg fallback
                # ran out of resources; release media and retry once.
                media_retry_done = True
                scan_state["media_retries"] = int(scan_state.get("media_retries", 0)) + 1
                _stub_oldest_media(messages)
                empty_cache = getattr(client, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
                continue
            if not _is_oom(exc) or attempts >= config.max_oom_retries:
                raise
            attempts += 1
            recovered = False
            if (
                config.force_full_scan
                and not scan_state.get("survey_downgraded")
                and _downgrade_scan_fps(messages, config.full_scan_fallback_fps)
            ):
                scan_state["survey_downgraded"] = True
                recovered = True
            elif _downgrade_latest_view(messages, config):
                recovered = True
            if not recovered:
                raise
            scan_state["oom_retries"] = int(scan_state.get("oom_retries", 0)) + 1
            empty_cache = getattr(client, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()


@dataclass
class EpisodeResult:
    question_id: str
    video_id: str
    task_type: str
    status: str
    intervals: list[dict[str, Any]] = field(default_factory=list)
    observation: str = ""
    confidence: float | None = None
    turns: int = 0
    inspect_calls: int = 0
    media_views: list[dict[str, Any]] = field(default_factory=list)
    oom_retries: int = 0
    interval_adjustments: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "video_id": self.video_id,
            "task_type": self.task_type,
            "status": self.status,
            "clue_intervals": [[item["start"], item["end"]] for item in self.intervals],
            "intervals": list(self.intervals),
            "observation": self.observation,
            "confidence": self.confidence,
            "turns": self.turns,
            "inspect_calls": self.inspect_calls,
            "media_views": list(self.media_views),
            "oom_retries": self.oom_retries,
            "interval_adjustments": list(self.interval_adjustments),
            "error": self.error,
        }


def normalize_intervals(
    raw: Any, duration: float, config: AgentConfig, task_type: str | None = None
) -> tuple[list[Interval] | None, list[dict[str, Any]], str | None]:
    """Validate submitted intervals and auto-repair length problems.

    Structural problems (not a list, missing observation, non-numeric bounds)
    still reject the submit.  Length problems are repaired instead of rejected
    -- over-budget windows are trimmed, too-short windows are expanded, extra
    intervals are dropped -- because rejecting them wastes the turn budget and
    was the main source of ``budget_exhausted`` episodes.  Every repair is
    recorded so the annotation stays auditable.
    """

    if raw is None or not isinstance(raw, list):
        return None, [], 'submit requires an "intervals" list (it may be empty if no evidence exists)'
    parsed: list[tuple[float, float, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None, [], "each interval must be an object with start/end/observation"
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            return None, [], "interval start/end must be numbers"
        observation = str(item.get("observation") or "").strip()
        if not observation:
            return None, [], "every interval needs an observation explaining the evidence"
        parsed.append((start, end, observation))

    adjustments: list[dict[str, Any]] = []
    fixed: list[list[Any]] = []
    for start, end, observation in parsed:
        low = max(0.0, min(duration, start))
        high = max(0.0, min(duration, end))
        if high <= low:
            adjustments.append({"kind": "dropped_empty", "from": [start, end]})
            continue
        if high - low < config.min_interval_seconds - 1e-6:
            pad = (config.min_interval_seconds - (high - low)) / 2.0
            low2 = max(0.0, low - pad)
            high2 = min(duration, low2 + config.min_interval_seconds)
            low2 = max(0.0, high2 - config.min_interval_seconds)
            adjustments.append(
                {"kind": "expanded_short", "from": [low, high], "to": [low2, high2]}
            )
            low, high = low2, high2
        if abs(low - start) > 1e-9 or abs(high - end) > 1e-9:
            adjustments.append({"kind": "clamped_range", "from": [start, end], "to": [low, high]})
        fixed.append([low, high, observation])

    if len(fixed) > config.max_intervals:
        adjustments.append(
            {
                "kind": "dropped_extra_intervals",
                "kept": config.max_intervals,
                "dropped": len(fixed) - config.max_intervals,
            }
        )
        fixed = fixed[: config.max_intervals]

    budget = config.check_interval_budget(duration, task_type)
    total = sum(high - low for low, high, _ in fixed)
    if total > budget + 1e-6:
        # A single interval that alone exceeds the budget (typical for
        # "throughout the video" evidence) is split into pieces spread across
        # the original range, so the head AND the tail stay covered instead of
        # truncating away the end.
        split: list[list[Any]] = []
        for low, high, observation in fixed:
            length = high - low
            if length > budget + 1e-6 and len(fixed) + len(split) < config.max_intervals:
                pieces = max(2, min(config.max_intervals - len(fixed) + 1, int(math.ceil(length / budget))))
                piece_len = budget / pieces
                for index in range(pieces):
                    start = low if pieces == 1 else low + (length - piece_len) * index / (pieces - 1)
                    split.append([start, start + piece_len, observation])
                adjustments.append(
                    {
                        "kind": "split_to_cover_range",
                        "from": [low, high],
                        "pieces": pieces,
                        "piece_seconds": round(piece_len, 2),
                    }
                )
            else:
                split.append([low, high, observation])
        fixed = split[: config.max_intervals]
        total = sum(high - low for low, high, _ in fixed)
    if total > budget + 1e-6:
        kept: list[list[Any]] = []
        remaining = budget
        for low, high, observation in fixed:
            length = high - low
            if length <= remaining + 1e-6:
                kept.append([low, high, observation])
                remaining -= length
            elif remaining >= config.min_interval_seconds - 1e-6:
                kept.append([low, low + remaining, observation])
                adjustments.append(
                    {"kind": "trimmed_to_budget", "from": [low, high], "to": [low, low + remaining]}
                )
                remaining = 0.0
                break
            else:
                break
        adjustments.append(
            {
                "kind": "enforced_total_budget",
                "budget_s": round(budget, 3),
                "before_s": round(total, 3),
                "after_s": round(sum(high - low for low, high, _ in kept), 3),
            }
        )
        fixed = kept

    if not fixed and parsed:
        detail = "; ".join(
            f"{item['kind']} {item.get('from')}" for item in adjustments[:3]
        ) or "no interval was inside the video range"
        return None, adjustments, (
            f"all submitted intervals were empty or outside 0..{duration:.2f}s "
            f"({detail}); submit intervals inside the video range"
        )
    intervals = []
    for low, high, observation in fixed:
        start, end = _round_inside(low, high, duration)
        intervals.append(Interval(start=start, end=end, observation=observation))
    return intervals, adjustments, None


def _round_inside(low: float, high: float, duration: float) -> tuple[float, float]:
    """Round to milliseconds while staying inside [0, duration].

    Plain round() can push the end past the duration (e.g. 124.25759 -> 124.258),
    which leaks out-of-range intervals into the dataset.  Floor the end and ceil
    the start instead.
    """

    start = math.ceil((low - 1e-9) * 1000.0) / 1000.0
    end = math.floor((high + 1e-9) * 1000.0) / 1000.0
    end = min(end, math.floor((duration + 1e-9) * 1000.0) / 1000.0)
    return max(0.0, start), max(start, end)


def validate_intervals(
    raw: Any, duration: float, config: AgentConfig
) -> tuple[list[Interval] | None, str | None]:
    """Strict validator retained for tests and callers that reject deviations."""

    if raw is None or not isinstance(raw, list):
        return None, 'submit requires an "intervals" list (it may be empty if no evidence exists)'
    if len(raw) > config.max_intervals:
        return None, f"at most {config.max_intervals} intervals are allowed, got {len(raw)}"
    intervals: list[Interval] = []
    for item in raw:
        if not isinstance(item, dict):
            return None, "each interval must be an object with start/end/observation"
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            return None, "interval start/end must be numbers"
        if start < -1e-6 or end > duration + 1e-6 or end <= start:
            return None, (
                f"interval [{start}, {end}] is outside 0..{duration:.3f} or not increasing"
            )
        if end - start < config.min_interval_seconds - 1e-6:
            return None, (
                f"interval [{start:.3f}, {end:.3f}] is shorter than "
                f"{config.min_interval_seconds}s"
            )
        observation = str(item.get("observation") or "").strip()
        if not observation:
            return None, "every interval needs an observation explaining the evidence"
        start, end = _round_inside(start, end, duration)
        intervals.append(Interval(start=start, end=end, observation=observation))
    total = sum(interval.duration for interval in intervals)
    total_budget = config.check_interval_budget(duration)
    if total > total_budget + 1e-6:
        return None, (
            f"total interval length {total:.1f}s exceeds the budget {total_budget:.1f}s"
        )
    return intervals, None


def _event(turn: int, kind: str, **payload: Any) -> dict[str, Any]:
    return {"turn": turn, "kind": kind, **payload}


def run_episode(
    record: QuestionRecord,
    client: OmniClient,
    config: AgentConfig,
    media_info: MediaInfo,
) -> tuple[EpisodeResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one bounded tool loop and return (result, stored_messages, events)."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(record, config)},
        {"role": "user", "content": build_initial_user_prompt(record, config)},
    ]
    events: list[dict[str, Any]] = []
    media_views: list[dict[str, Any]] = []
    inspect_calls = 0
    seen_views: set[tuple[float, float, float, int, str]] = set()
    rejected_inspects = 0
    last_inspect_payload: str | None = None
    repeat_count = 0
    scan_state: dict[str, Any] = {"survey_downgraded": False, "oom_retries": 0}
    result = EpisodeResult(
        question_id=record.question_id,
        video_id=record.video_id,
        task_type=record.task_type,
        status="budget_exhausted",
    )

    media_payload = {
        "duration_s": round(media_info.duration_s, 3),
        "fps": round(media_info.fps, 3),
        "n_frames": media_info.n_frames,
        "has_audio": media_info.has_audio,
        "width": media_info.width,
        "height": media_info.height,
    }

    bootstrap = None
    bootstrap_fn = getattr(client, "bootstrap", None)
    if callable(bootstrap_fn):
        bootstrap = bootstrap_fn(record, media_info, config)
    if bootstrap is not None:
        events.append(_event(0, "bootstrap", **dict(bootstrap.metadata)))
        messages.extend(bootstrap.messages)

    if (
        config.force_full_scan
        and media_info.duration_s > 0
        and not (bootstrap is not None and bootstrap.skip_survey)
    ):
        survey = ViewRequest(
            start=0.0,
            end=round(float(media_info.duration_s), 3),
            fps=float(config.full_scan_fps),
            max_pixels=int(config.full_scan_max_pixels),
            modality="av",
            focus="initial full-video survey: build a rough timeline of what happens and when",
        )
        survey_message = make_media_message(survey, record.video_path, record.duration_s)
        survey_message[MEDIA_KEY]["scan"] = True
        survey_message[MEDIA_KEY]["max_frames"] = int(config.full_scan_max_frames)
        messages.append(survey_message)
        survey_message[MEDIA_KEY]["turn"] = 0
        media_views.append(survey_message[MEDIA_KEY])
        events.append(
            _event(
                0,
                "survey",
                view={key: value for key, value in survey_message[MEDIA_KEY].items() if key != "video_path"},
            )
        )

    for turn in range(1, config.max_turns + 1):
        result.turns = turn
        try:
            raw = _generate_with_oom_recovery(client, messages, config, scan_state)
        except Exception as exc:  # noqa: BLE001 - keep the shard alive
            is_block = getattr(client, "is_content_block", None)
            if callable(is_block) and is_block(exc):
                result.status = "blocked"
            else:
                result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"[:500]
            result.inspect_calls = inspect_calls
            result.media_views = media_views
            events.append(_event(turn, "generation_error", error=result.error))
            return result, messages, events
        messages.append({"role": "assistant", "content": raw})
        action, error = parse_action(raw)
        if error is not None:
            events.append(_event(turn, "protocol_error", error=error, raw=raw[:1000]))
            messages.append({"role": "user", "content": f"Protocol error: {error}"})
            continue

        events.append(
            _event(turn, "action", action=action.action, observation=action.observation)
        )

        if action.action == "submit":
            if (
                config.require_inspect_before_submit
                and inspect_calls == 0
                and rejected_inspects < 2
            ):
                events.append(_event(turn, "submit_rejected", error="no inspect before submit"))
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Submit rejected: you must inspect at least one view to confirm "
                            "the evidence before submitting."
                        ),
                    }
                )
                continue
            intervals, adjustments, interval_error = normalize_intervals(
                action.payload.get("intervals"), record.duration_s, config, record.task_type
            )
            if interval_error is not None:
                events.append(_event(turn, "submit_rejected", error=interval_error))
                messages.append({"role": "user", "content": f"Submit rejected: {interval_error}"})
                continue
            if adjustments:
                events.append(_event(turn, "intervals_adjusted", adjustments=adjustments))
            confidence = action.payload.get("confidence")
            try:
                confidence_value = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence_value = None
            result.status = "submitted" if intervals else "no_evidence"
            result.intervals = [interval.as_dict() for interval in intervals]
            result.interval_adjustments = adjustments
            result.observation = action.observation
            result.confidence = confidence_value
            result.inspect_calls = inspect_calls
            result.media_views = media_views
            result.oom_retries = int(scan_state.get("oom_retries", 0))
            events.append(_event(turn, "submitted", status=result.status, intervals=result.intervals))
            return result, messages, events

        if action.action == "get_media_info":
            messages.append(
                {"role": "user", "content": "Media info: " + json.dumps(media_payload, ensure_ascii=False)}
            )
            continue

        if inspect_calls >= config.max_inspect:
            rejected_inspects += 1
            events.append(_event(turn, "inspect_rejected", error="inspect budget exhausted"))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Inspect budget exhausted ({config.max_inspect} views used). "
                        "Submit your best evidence intervals now." + repeat_nudge
                    ),
                }
            )
            continue

        payload_key = json.dumps(action.payload, sort_keys=True, ensure_ascii=False)
        if payload_key == last_inspect_payload:
            repeat_count += 1
        else:
            repeat_count = 0
        last_inspect_payload = payload_key
        repeat_nudge = ""
        if repeat_count >= 1:
            repeat_nudge = (
                f" You have made this exact inspect request {repeat_count + 1} times in a row and it "
                "will keep being rejected."
            )
            if repeat_count >= 2:
                repeat_nudge += (
                    " You MUST use action=submit now with your best evidence intervals (the survey "
                    "already shows the video timeline)."
                )

        try:
            view = clamp_view(action.payload, record.duration_s, config)
        except (TypeError, ValueError) as exc:
            rejected_inspects += 1
            events.append(_event(turn, "inspect_rejected", error=str(exc)))
            messages.append(
                {"role": "user", "content": f"Inspect rejected: {exc}. Fix the parameters." + repeat_nudge}
            )
            continue

        coverage = (view.end - view.start) / max(1e-6, float(record.duration_s))
        if (
            config.force_full_scan
            and coverage >= 0.9
            and view.max_pixels < 2 * int(config.full_scan_max_pixels)
        ):
            # A full-range request is served as a SHARPER full pass instead of
            # being rejected: rejecting it just made the model repeat itself
            # until the turn budget was exhausted.
            upgraded = max(view.max_pixels, 2 * int(config.full_scan_max_pixels))
            events.append(
                _event(
                    turn,
                    "inspect_upgraded",
                    reason="full_range_request_rendered_sharper",
                    from_max_pixels=view.max_pixels,
                    to_max_pixels=upgraded,
                )
            )
            view = ViewRequest(
                start=view.start,
                end=view.end,
                fps=view.fps,
                max_pixels=min(upgraded, config.max_pixels_max),
                modality=view.modality,
                focus=view.focus,
            )
        view_key = (view.start, view.end, view.fps, view.max_pixels, view.modality)
        if view_key in seen_views:
            rejected_inspects += 1
            events.append(_event(turn, "inspect_rejected", error="duplicate view"))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Inspect rejected: you already requested this exact view "
                        f"({view.start:.1f}s-{view.end:.1f}s, fps={view.fps:g}, "
                        f"modality={view.modality}). Its content is in your context (or was "
                        "already observed above). Choose a DIFFERENT time range or submit now."
                        + repeat_nudge
                    ),
                }
            )
            continue
        seen_views.add(view_key)
        media_message = make_media_message(view, record.video_path, record.duration_s)
        media_message[MEDIA_KEY]["turn"] = turn
        media_views.append(media_message[MEDIA_KEY])
        messages.append(media_message)
        stored_view = media_message[MEDIA_KEY]
        messages = trim_media_context(messages, config.max_media_in_context)
        inspect_calls += 1
        events.append(_event(turn, "inspect", view=stored_view))

    result.inspect_calls = inspect_calls
    result.media_views = media_views
    result.oom_retries = int(scan_state.get("oom_retries", 0))
    result.error = f"no submit within {config.max_turns} turns"
    events.append(_event(result.turns, "budget_exhausted", error=result.error))
    return result, messages, events
