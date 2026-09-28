"""Unit tests for the WorldSense evidence-localization agent (no GPU needed)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omni_opsd.worldsense.clients import MockOmniClient
from omni_opsd.worldsense.config import AgentConfig
from omni_opsd.worldsense.episode import normalize_intervals, run_episode, validate_intervals
from omni_opsd.worldsense.media import MediaInfo, MediaIndex
from omni_opsd.worldsense.prompts import build_initial_user_prompt, build_system_prompt
from omni_opsd.worldsense.protocol import parse_action
from omni_opsd.worldsense.runner import run_batch
from omni_opsd.worldsense.schema import BootstrapResult, QuestionRecord
from omni_opsd.worldsense.views import (
    MEDIA_KEY,
    clamp_view,
    make_media_message,
    render_conversation,
    trim_media_context,
)


def make_record(**overrides) -> QuestionRecord:
    data = dict(
        question_id="TESTVID::task0",
        video_id="TESTVID",
        task_index=0,
        task_domain="Understanding",
        task_type="Spatial Relation",
        question="What is the position of the metal door?",
        candidates=["A. left", "B. right", "C. front", "D. behind"],
        video_caption="A woman plays an instrument; a metal door is behind her.",
        domain="Music",
        sub_category="Covers",
        audio_class=["Music", "Speech"],
        duration_s=60.0,
        video_path="/tmp/does-not-need-to-exist/TESTVID.mp4",
        answer_letter="B",
    )
    data.update(overrides)
    return QuestionRecord(**data)


def make_media_info(duration: float = 60.0) -> MediaInfo:
    return MediaInfo(duration_s=duration, fps=30.0, n_frames=int(duration * 30), has_audio=True, width=640, height=360)


def test_prompt_contains_requested_metadata() -> None:
    config = AgentConfig()
    prompt = build_system_prompt(make_record(), config)
    for needle in (
        "video_duration_seconds",
        "domain",
        "sub_category",
        "audio_class",
        "task_domain",
        "task_type",
        "observation",
        "inspect",
        "submit",
    ):
        assert needle in prompt, f"missing {needle!r} in system prompt"
    assert "A. left" in prompt
    assert "never" in prompt.lower()
    assert "video_synopsis" in prompt, "synopsis is shown as a weak hint by default"
    assert "full-video survey" in prompt
    user_prompt = build_initial_user_prompt(make_record(), config)
    assert "answer" in user_prompt.lower()
    assert "survey" in user_prompt.lower()


def test_synopsis_can_be_disabled_for_ablation() -> None:
    disabled = AgentConfig(include_synopsis=False, include_caption_in_metadata=False)
    prompt = build_system_prompt(make_record(), disabled)
    assert "video_synopsis" not in prompt


def test_forced_survey_is_injected_before_first_turn() -> None:
    config = AgentConfig(require_inspect_before_submit=False)
    client = MockOmniClient(
        [
            json.dumps(
                {
                    "observation": "surveyed the whole video; the door appears mid-clip",
                    "action": "submit",
                    "intervals": [{"start": 2.0, "end": 8.0, "observation": "door visible"}],
                    "confidence": 0.5,
                }
            )
        ]
    )
    record = make_record(duration_s=60.0)
    result, messages, events = run_episode(record, client, config, make_media_info(60.0))
    survey_views = [m[MEDIA_KEY] for m in messages if m.get(MEDIA_KEY)]
    assert survey_views, "the harness must inject a full-video survey"
    survey = survey_views[0]
    assert survey.get("scan") is True
    assert survey["start"] == 0.0 and abs(survey["end"] - 60.0) < 1e-6
    assert survey["fps"] == config.full_scan_fps
    assert survey["modality"] == "av"
    assert events[0]["kind"] == "survey"
    assert result.inspect_calls == 0, "the survey must not consume the inspect budget"
    assert result.status == "submitted"


def test_survey_can_be_disabled() -> None:
    config = AgentConfig(force_full_scan=False, require_inspect_before_submit=False)
    client = MockOmniClient(
        [
            json.dumps(
                {
                    "observation": "inspect directly",
                    "action": "submit",
                    "intervals": [{"start": 1.0, "end": 3.0, "observation": "x"}],
                    "confidence": 0.3,
                }
            )
        ]
    )
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    assert not any(m.get(MEDIA_KEY) for m in messages)
    assert not any(e["kind"] == "survey" for e in events)
    assert result.status == "submitted"


class OomOnceClient(MockOmniClient):
    """Raises a CUDA-OOM-like error on the first call, then behaves normally."""

    def __init__(self, scripted_responses):
        super().__init__(scripted_responses)
        self.oom_calls = 0

    def generate(self, messages, *, max_new_tokens, temperature):
        if self.oom_calls == 0:
            self.oom_calls += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 12.00 GiB")
        return super().generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)


def test_latest_view_downgrades_after_survey_downgraded() -> None:
    """After the survey is already at the fallback fps, the newest view shrinks."""

    class TwoOomsClient(MockOmniClient):
        def __init__(self, scripted_responses):
            super().__init__(scripted_responses)
            self.oom_calls = 0

        def generate(self, messages, *, max_new_tokens, temperature):
            # OOM on the first call (survey only) and again on the call after the
            # first inspect, so both recovery branches are exercised.
            self.oom_calls += 1
            if self.oom_calls in (1, 3):
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
            return super().generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    config = AgentConfig(
        full_scan_fps=2.0,
        full_scan_fallback_fps=1.0,
        max_oom_retries=2,
        require_inspect_before_submit=False,
    )
    inspect_action = json.dumps(
        {
            "observation": "zoom in",
            "action": "inspect",
            "start": 5,
            "end": 25,
            "fps": 4,
            "max_pixels": 313600,
            "modality": "av",
            "focus": "detail",
        }
    )
    submit = json.dumps(
        {
            "observation": "found it",
            "action": "submit",
            "intervals": [{"start": 5.0, "end": 20.0, "observation": "evidence"}],
            "confidence": 0.5,
        }
    )
    client = TwoOomsClient([inspect_action, submit])
    result, messages, events = run_episode(make_record(duration_s=60.0), client, config, make_media_info(60.0))
    assert client.oom_calls == 4, client.oom_calls
    views = [m[MEDIA_KEY] for m in messages if m.get(MEDIA_KEY)]
    survey = views[0]
    assert survey["fps"] == 1.0, "survey downgraded first"
    inspect_view = views[-1]
    assert inspect_view["max_pixels"] < 313600, "the newest inspect view must shrink"
    assert inspect_view["fps"] < 4
    assert result.status == "submitted"
    assert result.oom_retries == 2


def test_survey_downgrades_fps_on_oom() -> None:
    config = AgentConfig(full_scan_fps=2.0, full_scan_fallback_fps=1.0, require_inspect_before_submit=False)
    client = OomOnceClient(
        [
            json.dumps(
                {
                    "observation": "the survey at 1 fps still shows the door",
                    "action": "submit",
                    "intervals": [{"start": 1.0, "end": 6.0, "observation": "door"}],
                    "confidence": 0.5,
                }
            )
        ]
    )
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    assert client.oom_calls == 1, "the OOM path must be exercised"
    survey = [m[MEDIA_KEY] for m in messages if m.get(MEDIA_KEY)][0]
    assert survey["fps"] == 1.0, "the survey must be re-sampled at the fallback fps"
    assert "lower frame rate" in [m for m in messages if m.get(MEDIA_KEY)][0]["content"]
    assert survey.get("downgraded") is True, "downgrades must be auditable on the view"
    assert result.status == "submitted"


def test_parse_action_accepts_fenced_json_with_prose() -> None:
    text = 'Sure, here is my action:\n```json\n{"observation": "checked start", "action": "inspect", "start": 1, "end": 5}\n```\n'
    action, error = parse_action(text)
    assert error is None and action is not None
    assert action.action == "inspect"
    assert action.observation == "checked start"


def test_parse_action_rejects_missing_observation() -> None:
    action, error = parse_action(json.dumps({"action": "submit", "intervals": []}))
    assert action is None and error is not None and "observation" in error


def test_clamp_view_enforces_limits() -> None:
    config = AgentConfig()
    view = clamp_view(
        {"start": -5, "end": 1000, "fps": 99, "max_pixels": 99, "modality": "AV", "focus": "x"},
        duration=60.0,
        config=config,
    )
    assert view.start == 0.0 and view.end == 60.0
    assert view.fps == config.fps_max
    assert view.max_pixels == config.max_pixels_min
    assert view.modality == "av"
    try:
        clamp_view({"start": 1, "end": 2, "modality": "telepathy"}, 60.0, config)
    except ValueError as exc:
        assert "modality" in str(exc)
    else:
        raise AssertionError("invalid modality must raise")


def test_clamp_view_slides_past_end_window_back_inside() -> None:
    """A window past the video end must be slid back, not rejected."""

    config = AgentConfig()
    view = clamp_view({"start": 87.83, "end": 90.0, "fps": 2}, 87.8333, config)
    assert view.end <= 87.8333 + 1e-6
    assert view.end - view.start >= config.min_view_seconds - 1e-6
    assert view.start >= 0.0 and view.start < view.end

    fully_past = clamp_view({"start": 90.0, "end": 95.0}, 87.8333, config)
    assert abs(fully_past.end - 87.8333) < 1e-2
    assert fully_past.end - fully_past.start >= config.min_view_seconds - 1e-6

    try:
        clamp_view({"start": 0.5, "end": 0.5}, 0.4, config)
    except ValueError as exc:
        assert "0.40" in str(exc)
    else:
        raise AssertionError("a 0.4s video cannot hold a min-length window")


def test_mixed_modality_renderer_keeps_silent_view_silent() -> None:
    config = AgentConfig()
    record = make_record()
    av_view = clamp_view({"start": 0, "end": 6, "fps": 2, "modality": "av"}, 60.0, config)
    silent_view = clamp_view({"start": 10, "end": 14, "fps": 4, "modality": "video"}, 60.0, config)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        make_media_message(av_view, record.video_path, record.duration_s),
        make_media_message(silent_view, record.video_path, record.duration_s),
    ]
    rendered, interleaved = render_conversation(messages)
    assert interleaved is False
    first_entries = rendered[2]["content"]
    assert [entry["type"] for entry in first_entries] == ["video", "audio", "text"]
    second_entries = rendered[3]["content"]
    assert [entry["type"] for entry in second_entries] == ["video", "text"]
    assert first_entries[0]["min_pixels"] == 3136


def test_trim_media_context_stubs_old_views() -> None:
    config = AgentConfig(max_media_in_context=1)
    record = make_record()
    view = clamp_view({"start": 0, "end": 6, "fps": 2, "modality": "av"}, 60.0, config)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        make_media_message(view, record.video_path, record.duration_s),
        make_media_message(view, record.video_path, record.duration_s),
    ]
    trimmed = trim_media_context(messages, config.max_media_in_context)
    assert MEDIA_KEY not in trimmed[2]
    assert MEDIA_KEY in trimmed[3]
    assert "media released from context" in trimmed[2]["content"]


def _scripted_episode(script: list[str]):
    client = MockOmniClient(script)
    record = make_record()
    config = AgentConfig(max_turns=6, max_inspect=3, temperature=0.0)
    result, messages, events = run_episode(record, client, config, make_media_info())
    return client, result, messages, events


def test_episode_loop_inspect_then_submit() -> None:
    script = [
        json.dumps(
            {
                "observation": "scan opening",
                "action": "inspect",
                "start": 2,
                "end": 12,
                "fps": 2,
                "max_pixels": 156800,
                "modality": "av",
                "focus": "door position",
            }
        ),
        json.dumps({"observation": "need metadata", "action": "get_media_info"}),
        json.dumps(
            {
                "observation": "evidence is in the first window",
                "action": "submit",
                "intervals": [
                    {"start": 2.0, "end": 8.0, "observation": "metal door visible behind the woman"}
                ],
                "confidence": 0.7,
            }
        ),
    ]
    client, result, messages, events = _scripted_episode(script)
    assert result.status == "submitted"
    assert result.inspect_calls == 1
    assert result.intervals[0]["start"] == 2.0 and result.intervals[0]["end"] == 8.0
    assert result.observation == "evidence is in the first window"
    assert result.confidence == 0.7
    kinds = [event["kind"] for event in events]
    assert kinds[-1] == "submitted"
    assert kinds.count("action") == 3
    assert "inspect" in kinds
    assert any(message.get(MEDIA_KEY) for message in messages)


def test_episode_rejects_then_retries_submit() -> None:
    script = [
        json.dumps(
            {
                "observation": "look first",
                "action": "inspect",
                "start": 1,
                "end": 6,
                "fps": 2,
                "max_pixels": 3136,
                "modality": "av",
                "focus": "check",
            }
        ),
        json.dumps(
            {
                "observation": "submit an interval without an observation",
                "action": "submit",
                "intervals": [{"start": 1.0, "end": 5.0}],
                "confidence": 0.5,
            }
        ),
        json.dumps(
            {
                "observation": "fixed the interval",
                "action": "submit",
                "intervals": [{"start": 1.0, "end": 5.0, "observation": "valid window"}],
                "confidence": 0.5,
            }
        ),
    ]
    client, result, messages, events = _scripted_episode(script)
    assert result.status == "submitted"
    kinds = [event["kind"] for event in events if event["kind"] != "survey"]
    assert kinds == ["action", "inspect", "action", "submit_rejected", "action", "submitted"]
    assert any("observation" in str(event.get("error")) for event in events)


def test_episode_budget_exhausted_without_submit() -> None:
    # Distinct views each turn so the duplicate guard does not fire and the
    # inspect budget itself is what gets exhausted.
    script = [
        json.dumps(
            {
                "observation": f"keep looking {index}",
                "action": "inspect",
                "start": 1 + index * 3,
                "end": 5 + index * 3,
                "fps": 2,
                "max_pixels": 3136,
                "modality": "av",
                "focus": f"window {index}",
            }
        )
        for index in range(6)
    ]
    client, result, messages, events = _scripted_episode(script)
    assert result.status == "budget_exhausted"
    assert result.inspect_calls == 3
    assert any(event["kind"] == "inspect_rejected" for event in events)


def test_submit_requires_an_inspect_first() -> None:
    config = AgentConfig()
    client = MockOmniClient(
        [
            json.dumps(
                {
                    "observation": "submitting straight away",
                    "action": "submit",
                    "intervals": [{"start": 1.0, "end": 5.0, "observation": "guess"}],
                    "confidence": 0.5,
                }
            ),
            json.dumps(
                {
                    "observation": "ok, inspected",
                    "action": "inspect",
                    "start": 1,
                    "end": 6,
                    "fps": 2,
                    "max_pixels": 3136,
                    "modality": "av",
                    "focus": "check",
                }
            ),
            json.dumps(
                {
                    "observation": "confirmed",
                    "action": "submit",
                    "intervals": [{"start": 1.0, "end": 5.0, "observation": "seen"}],
                    "confidence": 0.5,
                }
            ),
        ]
    )
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    assert result.status == "submitted"
    assert result.inspect_calls == 1
    assert any(
        event["kind"] == "submit_rejected" and "inspect" in str(event.get("error"))
        for event in events
    )


def test_normalize_intervals_error_mentions_duration() -> None:
    config = AgentConfig()
    intervals, adjustments, error = normalize_intervals(
        [{"start": 500.0, "end": 520.0, "observation": "way outside"}],
        duration=131.36,
        config=config,
    )
    assert intervals is None and error is not None
    assert "131.36" in error


def test_oom_downgrade_is_marked_on_the_view() -> None:
    class OomClient(MockOmniClient):
        def __init__(self, scripted):
            super().__init__(scripted)
            self.n_calls = 0

        def generate(self, messages, *, max_new_tokens, temperature):
            self.n_calls += 1
            if self.n_calls == 1:
                raise RuntimeError("CUDA out of memory")
            return super().generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    client = OomClient(
        [
            json.dumps(
                {
                    "observation": "ok",
                    "action": "submit",
                    "intervals": [{"start": 1.0, "end": 5.0, "observation": "e"}],
                    "confidence": 0.5,
                }
            )
        ]
    )
    config = AgentConfig(require_inspect_before_submit=False)
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    survey = [m[MEDIA_KEY] for m in messages if m.get(MEDIA_KEY)][0]
    assert survey.get("downgraded") is True
    assert str(survey.get("downgrade_reason")).startswith("oom_survey_frames_")
    assert int(survey.get("max_frames", 0)) == 300, "frames must halve to actually save memory"
    assert result.media_views[0].get("downgraded") is True, "media_views must reflect the downgrade"
    assert result.oom_retries == 1


def test_submit_is_allowed_after_rejected_inspects() -> None:
    """Two rejected inspects must not deadlock an episode that has seen media."""

    config = AgentConfig()
    full_range = json.dumps(
        {
            "observation": "watch all again",
            "action": "inspect",
            "start": 0,
            "end": 60,
            "fps": 2,
            "max_pixels": 3136,
            "modality": "av",
            "focus": "whole video",
        }
    )
    submit = json.dumps(
        {
            "observation": "submit with what the survey showed",
            "action": "submit",
            "intervals": [{"start": 1.0, "end": 5.0, "observation": "survey evidence"}],
            "confidence": 0.4,
        }
    )
    client = MockOmniClient([full_range, full_range, submit])
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    # The first full-range request is upgraded (not rejected); the second is an
    # exact duplicate of the upgraded view and is rejected, after which submit
    # is still allowed because the survey was already seen.
    assert result.status == "submitted", (result.status, result.error)
    assert any(event["kind"] in ("inspect_upgraded", "inspect_rejected") for event in events)


def test_duplicate_inspect_is_rejected_without_cost() -> None:
    config = AgentConfig()
    same_view = json.dumps(
        {
            "observation": "look again",
            "action": "inspect",
            "start": 10,
            "end": 20,
            "fps": 2,
            "max_pixels": 3136,
            "modality": "av",
            "focus": "same as before",
        }
    )
    submit = json.dumps(
        {
            "observation": "done",
            "action": "submit",
            "intervals": [{"start": 1.0, "end": 5.0, "observation": "evidence"}],
            "confidence": 0.5,
        }
    )
    client = MockOmniClient([same_view, same_view, submit])
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    assert result.status == "submitted"
    assert result.inspect_calls == 1, "the duplicate must not consume the inspect budget"
    assert any(
        event["kind"] == "inspect_rejected" and event.get("error") == "duplicate view"
        for event in events
    )


class BootstrappingClient(MockOmniClient):
    """Mock backend that injects a caption and asks the harness to skip the survey."""

    def bootstrap(self, record, media_info, config):
        return BootstrapResult(
            messages=[{"role": "user", "content": "VIDEO DESCRIPTION (timestamped): [00:00-00:05] a door."}],
            skip_survey=True,
            metadata={"caption_chars": 50, "caption": "captioned"},
        )


def test_bootstrap_replaces_the_survey() -> None:
    config = AgentConfig()
    client = BootstrappingClient(
        [
            json.dumps(
                {
                    "observation": "the caption points at 2-6s",
                    "action": "inspect",
                    "start": 2,
                    "end": 6,
                    "fps": 2,
                    "max_pixels": 156800,
                    "modality": "av",
                    "focus": "door",
                }
            ),
            json.dumps(
                {
                    "observation": "confirmed",
                    "action": "submit",
                    "intervals": [{"start": 2.0, "end": 6.0, "observation": "door visible"}],
                    "confidence": 0.6,
                }
            ),
        ]
    )
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    assert result.status == "submitted"
    kinds = [event["kind"] for event in events]
    assert "bootstrap" in kinds and "survey" not in kinds
    assert any("VIDEO DESCRIPTION" in str(m.get("content")) for m in messages)
    assert not any(m.get(MEDIA_KEY) and m[MEDIA_KEY].get("scan") for m in messages)
    assert result.inspect_calls == 1


class MediaErrorOnceClient(MockOmniClient):
    """Raises a decord/ffmpeg-style media error once, then succeeds."""

    def __init__(self, scripted):
        super().__init__(scripted)
        self.failed = 0

    def generate(self, messages, *, max_new_tokens, temperature):
        if self.failed == 0:
            self.failed += 1
            raise RuntimeError(
                "video_reader_backend decord error, use torchvision as default; "
                "av.error.BlockingIOError: swscaler Failed initializing scaling graph"
            )
        return super().generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)


def test_media_error_is_recovered_by_releasing_media() -> None:
    config = AgentConfig(require_inspect_before_submit=False)
    client = MediaErrorOnceClient(
        [
            json.dumps(
                {
                    "observation": "retry ok",
                    "action": "submit",
                    "intervals": [{"start": 1.0, "end": 5.0, "observation": "evidence"}],
                    "confidence": 0.5,
                }
            )
        ]
    )
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    assert client.failed == 1
    assert result.status == "submitted", result.error
    assert not any(m.get(MEDIA_KEY) for m in messages), "media must be released for the retry"


def test_persistent_error_does_not_crash_the_episode() -> None:
    class AlwaysFailing(MockOmniClient):
        def generate(self, messages, *, max_new_tokens, temperature):
            raise RuntimeError("video_reader_backend decord error; swscaler failed")

    result, messages, events = run_episode(
        make_record(), AlwaysFailing([]), AgentConfig(), make_media_info()
    )
    assert result.status == "error"
    assert result.error and "decord" in result.error
    assert any(event["kind"] == "generation_error" for event in events)


def test_full_range_reinspect_is_upgraded_not_rejected() -> None:
    config = AgentConfig(full_scan_max_pixels=31360)
    full_range = json.dumps(
        {
            "observation": "watch everything again",
            "action": "inspect",
            "start": 0,
            "end": 60,
            "fps": 2,
            "max_pixels": 15680,
            "modality": "av",
            "focus": "whole video",
        }
    )
    sharper_full = json.dumps(
        {
            "observation": "sharper full pass",
            "action": "inspect",
            "start": 0,
            "end": 60,
            "fps": 2,
            "max_pixels": 62720,
            "modality": "av",
            "focus": "sharp full pass",
        }
    )
    submit = json.dumps(
        {
            "observation": "done",
            "action": "submit",
            "intervals": [{"start": 1.0, "end": 5.0, "observation": "evidence"}],
            "confidence": 0.5,
        }
    )
    client = MockOmniClient([full_range, sharper_full, submit])
    result, messages, events = run_episode(make_record(duration_s=60.0), client, config, make_media_info(60.0))
    assert result.status == "submitted"
    # The low-resolution full-range request is served as a sharper full pass
    # instead of being rejected; the follow-up sharper request then duplicates
    # it and is the one rejected.
    assert any(event["kind"] == "inspect_upgraded" for event in events)
    assert result.inspect_calls == 1, "only the upgraded full pass should count"


def test_survey_stays_in_context_with_two_media_slots() -> None:
    config = AgentConfig(max_media_in_context=2)
    inspect_action = json.dumps(
        {
            "observation": "zoom",
            "action": "inspect",
            "start": 1,
            "end": 6,
            "fps": 2,
            "max_pixels": 3136,
            "modality": "av",
            "focus": "detail",
        }
    )
    submit = json.dumps(
        {
            "observation": "done",
            "action": "submit",
            "intervals": [{"start": 1.0, "end": 5.0, "observation": "evidence"}],
            "confidence": 0.5,
        }
    )
    client = MockOmniClient([inspect_action, submit])
    result, messages, events = run_episode(make_record(), client, config, make_media_info())
    media = [m for m in messages if m.get(MEDIA_KEY)]
    assert len(media) == 2, "survey + latest inspect must both keep their media"
    assert media[0][MEDIA_KEY].get("scan") is True
    assert result.status == "submitted"


def test_normalize_intervals_splits_oversized_interval_across_range() -> None:
    config = AgentConfig()
    intervals, adjustments, error = normalize_intervals(
        [{"start": 0.0, "end": 120.0, "observation": "whole video"}],
        duration=131.36,
        config=config,
    )
    assert error is None and intervals is not None
    budget = config.check_interval_budget(131.36)
    total = sum(item.duration for item in intervals)
    assert abs(total - budget) < 1e-6, "the pieces must fit the budget"
    assert len(intervals) >= 2, "a single over-budget interval is split"
    assert intervals[0].start == 0.0, "the head of the range stays covered"
    assert abs(intervals[-1].end - 120.0) < 1e-6, "the tail of the range stays covered"
    assert any(item["kind"] == "split_to_cover_range" for item in adjustments)


def test_wide_budget_tasks_get_a_larger_ratio() -> None:
    config = AgentConfig()
    assert config.check_interval_budget(100.0, "Temporal Localization") > config.check_interval_budget(100.0, "Spatial Relation")
    intervals, _, error = normalize_intervals(
        [{"start": 0.0, "end": 80.0, "observation": "spans the video"}],
        duration=100.0,
        config=config,
        task_type="Temporal Localization",
    )
    assert error is None and intervals is not None
    assert abs(sum(item.duration for item in intervals) - 80.0) < 1e-6, "80% ratio keeps it intact"


def test_normalize_intervals_expands_short_and_caps_count() -> None:
    config = AgentConfig()
    intervals, adjustments, error = normalize_intervals(
        [{"start": float(i), "end": float(i) + 0.2, "observation": f"i{i}"} for i in range(6)],
        duration=60.0,
        config=config,
    )
    assert error is None and intervals is not None
    assert len(intervals) == config.max_intervals
    assert all(item.duration >= config.min_interval_seconds - 1e-6 for item in intervals)
    assert any(item["kind"] == "expanded_short" for item in adjustments)
    assert any(item["kind"] == "dropped_extra_intervals" for item in adjustments)


def test_validate_intervals_rejects_oversized_total() -> None:
    config = AgentConfig()
    intervals, error = validate_intervals(
        [
            {"start": 0, "end": 50, "observation": "a"},
            {"start": 50, "end": 100, "observation": "b"},
        ],
        duration=200.0,
        config=config,
    )
    assert intervals is None and error is not None and "exceeds the budget" in error


def test_runner_resume_and_trace_roundtrip() -> None:
    from omni_opsd.worldsense.dataset import load_questions

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        video_root = root / "videos"
        video_root.mkdir()
        (video_root / "TESTVID.mp4").write_bytes(b"not a real video")
        qa = root / "qa.json"
        qa.write_text(
            json.dumps(
                {
                    "TESTVID": {
                        "video_id": "TESTVID",
                        "video_duration": "60s",
                        "domain": "Music",
                        "sub_category": "Covers",
                        "audio_class": ["Music"],
                        "video_caption": "synopsis",
                        "task0": {
                            "task_domain": "Understanding",
                            "task_type": "Spatial Relation",
                            "question": "q?",
                            "answer": "B",
                            "candidates": ["A. a", "B. b"],
                        },
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        index = MediaIndex({str((video_root / "TESTVID.mp4").resolve()): make_media_info()})
        records = load_questions(qa, video_root, index)
        assert len(records) == 1 and records[0].duration_s == 60.0

        config = AgentConfig()
        output = root / "out.jsonl"
        trace = root / "out.trace.jsonl"
        script = [
            json.dumps(
                {
                    "observation": "scan",
                    "action": "inspect",
                    "start": 1,
                    "end": 6,
                    "fps": 2,
                    "max_pixels": 3136,
                    "modality": "av",
                    "focus": "x",
                }
            ),
            json.dumps(
                {
                    "observation": "done",
                    "action": "submit",
                    "intervals": [{"start": 1, "end": 6, "observation": "evidence here"}],
                    "confidence": 0.4,
                }
            ),
        ]
        summary = run_batch(
            records,
            None,
            config,
            index,
            output,
            trace,
            client_factory=lambda record: MockOmniClient(list(script)),
            log=lambda message: None,
        )
        assert summary["processed"] == 1
        assert summary["by_status"] == {"submitted": 1}
        assert output.read_text(encoding="utf-8").strip()
        trace_row = json.loads(trace.read_text(encoding="utf-8").strip())
        assert trace_row["status"] == "submitted"
        assert trace_row["events"][-1]["kind"] == "submitted"

        summary_again = run_batch(
            records,
            None,
            config,
            index,
            output,
            trace,
            client_factory=lambda record: MockOmniClient(list(script)),
            log=lambda message: None,
        )
        assert summary_again["processed"] == 0
        assert summary_again["skipped_resume"] == 1


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - test runner reports all failures
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
