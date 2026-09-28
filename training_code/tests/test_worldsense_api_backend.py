"""Unit tests for the cloud API backend (no network, no ffmpeg needed)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omni_opsd.worldsense import api_client as api_mod
from omni_opsd.worldsense.clips import ClipRequest, plan_attempts, target_scale
from omni_opsd.worldsense.schema import QuestionRecord
from omni_opsd.worldsense.views import MEDIA_KEY, ViewRequest, make_media_message


def make_record() -> QuestionRecord:
    return QuestionRecord(
        question_id="TESTVID::task0",
        video_id="TESTVID",
        task_index=0,
        task_domain="Understanding",
        task_type="Spatial Relation",
        question="q?",
        candidates=["A. a", "B. b", "C. c", "D. d"],
        video_caption="synopsis",
        domain="Music",
        sub_category="Covers",
        audio_class=["Music"],
        duration_s=60.0,
        video_path="/tmp/TESTVID.mp4",
        answer_letter="B",
    )


class _FakeCompletions:
    def __init__(self, captured):
        self.captured = captured

    def _text_for(self, kwargs) -> str:
        import json as _json

        blob = _json.dumps(kwargs.get("messages"), ensure_ascii=False)
        if "VIDEO DESCRIPTION" in blob:
            return '{"observation":"plan","action":"inspect","start":10,"end":20,"fps":2,"max_pixels":156800,"modality":"av","focus":"check"}'
        if "audio-visual description expert" in blob:
            return (
                "[00:00:00:000-00:00:04:000] The video opens with a wide shot of a field of "
                "large light grey stones under a cloudy sky, with wind noise in the background. "
                "[00:00:04:000-00:00:09:000] The camera pans right and a person walks into the "
                "frame carrying a basket; the narration says the stones were moved by farmers."
            )
        return '{"observation":"ok","action":"submit","intervals":[]}'

    def create(self, **kwargs):
        self.captured.append(kwargs)
        text = self._text_for(kwargs)
        if kwargs.get("stream"):
            def chunks():
                for index in range(0, len(text), 64):
                    delta = types.SimpleNamespace(content=text[index:index + 64], reasoning_content=None)
                    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
                yield types.SimpleNamespace(choices=[], usage=types.SimpleNamespace(total_tokens=1))

            return chunks()
        message = types.SimpleNamespace(content=text)
        choice = types.SimpleNamespace(message=message, logprobs=None)
        return types.SimpleNamespace(choices=[choice], usage=None)


def make_client(monkeypatch_render: bool = True):
    captured: list[dict] = []
    client = api_mod.QwenAPIOmniClient.__new__(api_mod.QwenAPIOmniClient)
    client.model = "qwen3.8-omni-flash"
    client.clip_cache_dir = Path("/tmp/ws_api_test_clips")
    client.max_clip_bytes = 7 * 1024 * 1024
    client.caption_store = None
    client.caption_reuse = True
    client.proxy_policy = "size_aware"
    client.media_index = None
    client.enable_thinking = False
    client.max_retries = 1
    client.clip_renders = 0
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_FakeCompletions(captured)))
    if monkeypatch_render:
        api_mod.render_clip = lambda *a, **k: Path("/tmp/ws_api_test_clips/fake.mp4")  # type: ignore[assignment]
        api_mod.render_audio_clip = lambda *a, **k: Path("/tmp/ws_api_test_clips/fake.mp3")  # type: ignore[assignment]
        api_mod.clip_to_data_uri = lambda path: "data:;base64,AAAA"  # type: ignore[assignment]
        api_mod.audio_to_data_uri = lambda path, fmt="mp3": "data:;base64,BBBB"  # type: ignore[assignment]
    return client, captured


def test_target_scale_keeps_area_under_budget() -> None:
    width, height = target_scale(156_800, 640, 360)
    assert width % 2 == 0 and height % 2 == 0
    assert width * height <= 156_800
    small = target_scale(3_136, 640, 360)
    assert small[0] * small[1] <= 3_136


def test_clip_cache_key_is_stable_and_sensitive() -> None:
    first = ClipRequest("/tmp/a.mp4", 1.0, 5.0, 2.0, 156_800, True)
    same = ClipRequest("/tmp/a.mp4", 1.0, 5.0, 2.0, 156_800, True)
    other = ClipRequest("/tmp/a.mp4", 1.0, 5.0, 4.0, 156_800, True)
    assert first.cache_key() == same.cache_key()
    assert first.cache_key() != other.cache_key()


def test_size_aware_respects_requested_cap_without_upscaling() -> None:
    # The caller asked for a small frame: the first attempt must honour it.
    small = plan_attempts(
        ClipRequest("/tmp/a.mp4", 0.0, 60.0, 2.0, 31_360, True, policy="size_aware"), (640, 360)
    )
    assert small[0]["label"] == "requested"
    assert small[0]["width"] * small[0]["height"] <= 31_360
    assert small[0]["audio_kbps"] == "96", "audio bitrate is free in token terms"
    assert small[1]["label"] == "requested_1fps"
    sizes = [attempt["width"] * attempt["height"] for attempt in small]
    assert sizes == sorted(sizes, reverse=True)

    # The caller asked for more than the source: never upscale beyond the source.
    big = plan_attempts(
        ClipRequest("/tmp/a.mp4", 0.0, 60.0, 4.0, 313_600, True, policy="size_aware"), (640, 360)
    )
    assert big[0]["label"] == "requested"
    assert (big[0]["width"], big[0]["height"]) == (640, 360), "capped at the source frame"
    labels = [attempt["label"] for attempt in big]
    assert "requested_1fps" in labels and "scale_156800" in labels and "safety_floor" in labels
    sizes = [attempt["width"] * attempt["height"] for attempt in big]
    assert sizes == sorted(sizes, reverse=True)


def test_fixed_policy_keeps_historical_behaviour() -> None:
    request = ClipRequest("/tmp/a.mp4", 0.0, 60.0, 4.0, 31_360, True, policy="fixed")
    attempts = plan_attempts(request, (640, 360))
    expected_w, expected_h = target_scale(31_360, 640, 360)
    assert attempts[0]["width"] == expected_w and attempts[0]["height"] == expected_h
    assert attempts[0]["fps"] == 4.0
    assert attempts[-1]["label"] == "fixed_half_scale"


def test_cache_key_includes_proxy_policy() -> None:
    size_aware = ClipRequest("/tmp/a.mp4", 0.0, 60.0, 2.0, 31_360, True, policy="size_aware")
    fixed = ClipRequest("/tmp/a.mp4", 0.0, 60.0, 2.0, 31_360, True, policy="fixed")
    assert size_aware.cache_key() != fixed.cache_key()


def test_api_messages_convert_modalities() -> None:
    client, _ = make_client()
    record = make_record()
    av = make_media_message(ViewRequest(0.0, 10.0, 2.0, 156_800, "av", "look"), record.video_path, 60.0)
    audio = make_media_message(ViewRequest(10.0, 14.0, 1.0, 3_136, "audio", "listen"), record.video_path, 60.0)
    silent = make_media_message(ViewRequest(20.0, 24.0, 4.0, 313_600, "video", "read"), record.video_path, 60.0)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        av,
        audio,
        silent,
    ]
    converted = client._api_messages(messages)
    assert converted[0] == {"role": "system", "content": "sys"}
    assert converted[1] == {"role": "user", "content": "q"}
    av_parts = converted[2]["content"]
    assert av_parts[0]["type"] == "video_url" and av_parts[0]["video_url"]["url"].startswith("data:")
    assert av_parts[-1]["type"] == "text"
    assert converted[3]["content"][0]["type"] == "input_audio"
    assert converted[3]["content"][0]["input_audio"]["format"] == "mp3"
    assert converted[4]["content"][0]["type"] == "video_url"


def test_api_generate_sends_text_modality_and_returns_text() -> None:
    client, captured = make_client()
    record = make_record()
    view = make_media_message(ViewRequest(0.0, 5.0, 2.0, 3_136, "av", "x"), record.video_path, 60.0)
    text = client.generate([{"role": "user", "content": "hi"}, view], max_new_tokens=64, temperature=0.2)
    assert "observation" in text
    assert captured, "the API must be called"
    payload = captured[0]
    assert payload["model"] == "qwen3.8-omni-flash"
    assert payload["modalities"] == ["text"]
    assert payload["stream"] is True
    assert payload["extra_body"] == {"enable_thinking": False}


def test_letter_probabilities_from_logprobs() -> None:
    candidates = [
        types.SimpleNamespace(token=" A", logprob=-0.2),
        types.SimpleNamespace(token="B", logprob=-2.5),
        types.SimpleNamespace(token="the", logprob=-0.1),
    ]
    token_info = types.SimpleNamespace(top_logprobs=candidates)
    choice = types.SimpleNamespace(logprobs=types.SimpleNamespace(content=[token_info]))
    probabilities = api_mod.QwenAPIOmniClient._letter_probabilities_from_logprobs(choice)
    assert probabilities is not None
    assert probabilities["A"] > probabilities["B"]
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    assert api_mod.QwenAPIOmniClient._letter_probabilities_from_logprobs(
        types.SimpleNamespace(logprobs=None)
    ) is None


def test_bootstrap_produces_caption_and_skips_survey() -> None:
    from omni_opsd.worldsense.config import AgentConfig

    client, captured = make_client()
    config = AgentConfig()
    media_info = types.SimpleNamespace(duration_s=131.36)
    result = client.bootstrap(make_record(), media_info, config)
    assert result is not None
    assert result.skip_survey is True
    assert result.messages and result.messages[0]["role"] == "user"
    assert "VIDEO DESCRIPTION" in result.messages[0]["content"]
    assert result.metadata["caption_chars"] > 0
    assert "00:00:00:000" in result.metadata["caption"]
    # bootstrap 只发一次调用（生成 caption）；规划消息注入后由 episode 循环接手
    assert len(captured) == 1
    first_blob = str(captured[0]["messages"])
    assert "audio-visual description expert" in first_blob
    assert "video_url" in first_blob, "caption pass must include the full video"
    assert captured[0]["max_tokens"] == config.caption_max_tokens


def test_bootstrap_can_be_disabled_and_falls_back_on_error() -> None:
    from omni_opsd.worldsense.config import AgentConfig

    client, _ = make_client()
    assert client.bootstrap(make_record(), types.SimpleNamespace(duration_s=60.0),
                            AgentConfig(caption_stage=False)) is None

    class Failing:
        def create(self, **kwargs):
            raise RuntimeError("network down")

    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Failing()))
    fallback = client.bootstrap(make_record(), types.SimpleNamespace(duration_s=60.0), AgentConfig())
    assert fallback is not None
    assert fallback.skip_survey is False
    assert "caption_error" in fallback.metadata


def test_caption_store_reuses_captions() -> None:
    import tempfile

    from omni_opsd.worldsense.caption_store import CaptionStore, caption_key
    from omni_opsd.worldsense.config import AgentConfig

    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "captions.jsonl"
        client, captured = make_client()
        client.caption_store = CaptionStore(store_path)
        config = AgentConfig()
        media_info = types.SimpleNamespace(duration_s=60.0)

        first = client.bootstrap(make_record(), media_info, config)
        assert first is not None and first.metadata["caption_source"] == "generated"
        assert len(captured) == 1
        assert store_path.is_file()

        second = client.bootstrap(make_record(), media_info, config)
        assert second is not None and second.metadata["caption_source"] == "reused"
        assert len(captured) == 1, "reuse must not call the API again"
        assert second.metadata["caption"] == first.metadata["caption"]

        key = caption_key("TESTVID::task0", fps=config.full_scan_fps, max_pixels=config.full_scan_max_pixels)
        assert client.caption_store.get(key) is not None
        assert caption_key("TESTVID::task1", fps=config.full_scan_fps,
                           max_pixels=config.full_scan_max_pixels) != key, \
            "captions are question-conditioned, so each question needs its own key"
        assert client.caption_store.get(caption_key("TESTVID::task0", fps=1.0,
                                                    max_pixels=config.full_scan_max_pixels)) is None


def test_caption_truncation_is_recorded() -> None:
    from omni_opsd.worldsense.config import AgentConfig

    class TruncatingCompletions:
        def create(self, **kwargs):
            if kwargs.get("stream"):
                def chunks():
                    delta = types.SimpleNamespace(content="[00:00:00:000-00:00:05:000] partial", reasoning_content=None)
                    yield types.SimpleNamespace(
                        choices=[types.SimpleNamespace(delta=delta, finish_reason=None)], usage=None
                    )
                    yield types.SimpleNamespace(
                        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=""), finish_reason="length")],
                        usage=types.SimpleNamespace(total_tokens=1),
                    )

                return chunks()
            return types.SimpleNamespace(choices=[], usage=None)

    client, _ = make_client()
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=TruncatingCompletions()))
    result = client.bootstrap(make_record(), types.SimpleNamespace(duration_s=60.0), AgentConfig())
    assert result is not None
    assert result.metadata["caption_truncated"] is True
    assert result.metadata["caption_finish_reason"] == "length"
    assert result.metadata["caption_valid"] is False
    assert result.metadata["caption_attempts"] == 2, "an invalid caption is retried once"


def test_audio_verification_raises_on_silent_clip() -> None:
    from omni_opsd.worldsense import clips as clips_mod

    original = clips_mod.audio_stream_info
    try:
        clips_mod.audio_stream_info = lambda path, timeout=60: {"present": False}
        try:
            clips_mod.verify_audio_or_raise("/tmp/fake.mp4", context="unit-test")
        except RuntimeError as exc:
            assert "audio stream missing" in str(exc)
        else:
            raise AssertionError("a silent clip must raise explicitly")
        clips_mod.audio_stream_info = lambda path, timeout=60: {
            "present": True, "codec": "aac", "sample_rate": 44100, "channels": 2
        }
        info = clips_mod.verify_audio_or_raise("/tmp/fake.mp4")
        assert info["codec"] == "aac"
    finally:
        clips_mod.audio_stream_info = original


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
