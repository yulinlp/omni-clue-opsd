"""Cloud API backend for the evidence-localization agent.

Talks to the OpenAI-compatible Qwen3.8-Omni-Flash endpoint.  Each stored media
view is transcoded on demand into a small proxy clip (frame rate and scale
encode the agent's requested sampling) and sent as a base64 data URI, so the
cloud model sees exactly the range the agent asked for.
用于证据定位代理的云 API 后端。

与兼容 OpenAI 的 Qwen3.8-Omni-Flash 端点进行通信。每个存储的媒体
视图都会根据需求转码为一个小型代理片段（帧率和缩放比例
反映了代理请求的采样范围），并作为 base64 数据 URI 发送，因此
云模型能精确识别代理所请求的范围。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from .caption_store import CaptionStore, caption_key
from .clips import (
    ClipRequest,
    audio_to_data_uri,
    clip_to_data_uri,
    render_audio_clip,
    render_clip,
)
from .media import MediaIndex
from .prompts import (
    CAPTION_RETRY_NOTE,
    build_caption_planning_message,
    build_caption_prompt,
    validate_caption,
)
from .schema import BootstrapResult, QuestionRecord
from .views import MEDIA_KEY

DEFAULT_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
DEFAULT_KEY_ENV = "DASHSCOPE_API_KEY"
LETTERS = ("A", "B", "C", "D")


class QwenAPIOmniClient:
    """Qwen3.8-Omni-Flash over Chat Completions (video/audio via data URIs)."""

    def __init__(
        self,
        *,
        model: str = "qwen3.8-omni-flash",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_KEY_ENV,
        clip_cache_dir: str | Path = "output/worldsense_api_clips",
        caption_store_path: str | Path = "output/worldsense_captions/captions.jsonl",
        proxy_policy: str = "size_aware",
        caption_reuse: bool = True,
        max_clip_mb: float = 7.0,
        media_index: MediaIndex | None = None,
        enable_thinking: bool = False,
        request_timeout: int = 600,
        max_retries: int = 3,
    ):
        from openai import OpenAI

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise SystemExit(
                f"API key not found: pass --api-key or set {api_key_env} "
                "(see docs/Instruction-For-Qwen3.8-Omni-Flash.md)"
            )
        resolved_base = base_url or os.environ.get(DEFAULT_BASE_URL_ENV)
        if not resolved_base:
            raise SystemExit(
                f"API base URL not found: pass --api-base-url or set {DEFAULT_BASE_URL_ENV}"
            )
        self.model = model
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base, timeout=request_timeout)
        self.clip_cache_dir = Path(clip_cache_dir)
        self.caption_store = CaptionStore(caption_store_path) if caption_reuse else None
        self.caption_reuse = bool(caption_reuse)
        self.proxy_policy = str(proxy_policy)
        self.max_clip_bytes = int(max_clip_mb * 1024 * 1024)
        self.media_index = media_index
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.clip_renders = 0

    # ------------------------------------------------------------------ media

    def _source_size(self, video_path: str) -> tuple[int, int]:
        if self.media_index is not None:
            try:
                info = self.media_index.get(video_path)
                if info.width and info.height:
                    return int(info.width), int(info.height)
            except Exception:
                pass
        return 640, 360

    def _media_part(self, view: dict[str, Any]) -> dict[str, Any]:
        modality = str(view.get("modality", "av"))
        request = ClipRequest(
            video_path=str(view["video_path"]),
            start=float(view["start"]),
            end=float(view["end"]),
            fps=float(view.get("fps", 2.0)),
            max_pixels=int(view.get("max_pixels", 156_800)),
            with_audio=modality in ("av",),
            tag=str(view.get("focus", ""))[:40],
        )
        if modality == "audio":
            path = render_audio_clip(request, self.clip_cache_dir)
            self.clip_renders += 1
            return {
                "type": "input_audio",
                "input_audio": {"data": audio_to_data_uri(path, "mp3"), "format": "mp3"},
            }
        path = render_clip(
            request,
            self.clip_cache_dir,
            source_size=self._source_size(request.video_path),
            max_bytes=self.max_clip_bytes,
            policy=self.proxy_policy,
        )
        self.clip_renders += 1
        return {"type": "video_url", "video_url": {"url": clip_to_data_uri(path)}}

    def _api_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            view = message.get(MEDIA_KEY)
            role = message["role"]
            text = str(message.get("content") or "")
            if not view:
                converted.append({"role": role, "content": text})
                continue
            content: list[dict[str, Any]] = [self._media_part(view)]
            if text:
                content.append({"type": "text", "text": text})
            converted.append({"role": role, "content": content})
        return converted

    # ------------------------------------------------------------- generation

    @staticmethod
    def is_content_block(exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}"
        return "data_inspection_failed" in text or "inappropriate content" in text.lower()

    def _create_raw(self, api_messages: Sequence[dict[str, Any]], **kwargs: Any):
        payload = {
            "model": self.model,
            "messages": list(api_messages),
            "modalities": ["text"],
            **kwargs,
        }
        if not self.enable_thinking:
            payload["extra_body"] = {"enable_thinking": False}
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**payload)
            except TypeError:
                payload.pop("extra_body", None)
            except Exception as exc:  # noqa: BLE001 - retry transient API failures
                last_error = exc
                time.sleep(min(30, 3 * attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("API request failed")

    def _create(self, messages: Sequence[dict[str, Any]], **kwargs: Any):
        return self._create_raw(self._api_messages(messages), **kwargs)

    def _generate_text(
        self, api_messages: Sequence[dict[str, Any]], *, max_tokens: int, temperature: float
    ) -> tuple[str, str | None]:
        """Return (text, finish_reason); ``finish_reason == 'length'`` means truncated."""

        completion = self._create_raw(
            api_messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: list[str] = []
        finish_reason: str | None = None
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            piece = getattr(choice.delta, "content", None)
            if piece:
                chunks.append(piece)
            if getattr(choice, "finish_reason", None):
                finish_reason = str(choice.finish_reason)
        return "".join(chunks).strip(), finish_reason

    # -------------------------------------------------------------- bootstrap

    def bootstrap(
        self,
        record: QuestionRecord,
        media_info: Any,
        config: Any,
    ) -> BootstrapResult | None:
        """Two-stage API prelude: timestamped caption -> first inspect plan.

        Stage 1 watches the whole video once and writes a chronological,
        timestamped description that is extra detailed around moments relevant
        to the question.  Stage 2 injects that description and asks the model
        to plan the first detailed look.  The full-video survey is skipped
        because the caption already summarises it.
        """

        if not getattr(config, "caption_stage", False):
            return None
        duration = float(getattr(media_info, "duration_s", 0.0) or 0.0)
        if duration <= 0:
            return None
        store_key = caption_key(
            record.question_id,
            fps=float(config.full_scan_fps),
            max_pixels=int(config.full_scan_max_pixels),
        )
        reused = self.caption_store.get(store_key) if self.caption_store is not None else None
        if reused is not None and reused.get("caption"):
            caption = str(reused["caption"])
            return BootstrapResult(
                messages=[{"role": "user", "content": build_caption_planning_message(record, caption)}],
                skip_survey=True,
                metadata={
                    "caption_source": "reused",
                    "caption_chars": len(caption),
                    "caption_seconds": 0.0,
                    "caption_key": store_key,
                    "caption": caption,
                },
            )
        # The survey asks for the source resolution (quality-first captioning);
        # the frame rate keeps the full 2 fps up to api_survey_max_frames.
        source_pixels = 640 * 360
        if self.media_index is not None:
            try:
                info = self.media_index.get(record.video_path)
                if info.width and info.height:
                    source_pixels = int(info.width) * int(info.height)
            except Exception:
                pass
        survey_fps = float(config.full_scan_fps)
        cap_frames = int(getattr(config, "api_survey_max_frames", 1312))
        if cap_frames > 0 and duration > 0:
            survey_fps = min(survey_fps, cap_frames / duration)
        survey_view = {
            "video_path": record.video_path,
            "start": 0.0,
            "end": round(duration, 3),
            "fps": round(max(0.5, survey_fps), 3),
            "max_pixels": source_pixels,
            "modality": "av",
            "focus": "full-video caption pass",
        }
        caption_request = [
            {
                "role": "user",
                "content": [
                    self._media_part(survey_view),
                    {"type": "text", "text": build_caption_prompt(record)},
                ],
            }
        ]
        started = time.time()
        caption = ""
        finish_reason: str | None = None
        valid = False
        reason = "not_attempted"
        attempts = 0
        try:
            for attempts in range(1, 3):
                request_messages = list(caption_request)
                if attempts > 1:
                    # retry with an explicit format reminder appended
                    request_messages = [
                        {
                            "role": "user",
                            "content": list(caption_request[0]["content"])
                            + [{"type": "text", "text": CAPTION_RETRY_NOTE}],
                        }
                    ]
                caption, finish_reason = self._generate_text(
                    request_messages,
                    max_tokens=int(config.caption_max_tokens),
                    temperature=0.2,
                )
                valid, reason = validate_caption(
                    caption, int(getattr(config, "caption_min_chars", 200))
                )
                if valid and finish_reason != "length":
                    break
        except Exception as exc:  # noqa: BLE001 - fall back to the plain survey flow
            return BootstrapResult(
                messages=[],
                skip_survey=False,
                metadata={"caption_error": f"{type(exc).__name__}: {exc}"},
            )
        elapsed = time.time() - started
        truncated = finish_reason == "length"
        if self.caption_store is not None and caption and valid and not truncated:
            self.caption_store.put(
                store_key,
                {
                    "video_id": record.video_id,
                    "question_id": record.question_id,
                    "caption": caption,
                    "model": self.model,
                    "fps": float(config.full_scan_fps),
                    "max_pixels": int(config.full_scan_max_pixels),
                    "caption_seconds": round(elapsed, 1),
                    "question_id": record.question_id,
                },
            )
        planning_message = {
            "role": "user",
            "content": build_caption_planning_message(record, caption),
        }
        return BootstrapResult(
            messages=[planning_message],
            skip_survey=True,
            metadata={
                "caption_source": "generated",
                "caption_chars": len(caption),
                "caption_seconds": round(elapsed, 1),
                "caption_max_tokens": int(config.caption_max_tokens),
                "caption_key": store_key,
                "caption_truncated": bool(truncated),
                "caption_finish_reason": finish_reason,
                "caption_valid": bool(valid),
                "caption_validation": reason,
                "caption_attempts": attempts,
                "caption": caption,
            },
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        completion = self._create(
            messages,
            max_tokens=max(16, int(max_new_tokens)),
            temperature=float(temperature),
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: list[str] = []
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                chunks.append(piece)
        return "".join(chunks).strip()

    def generate_with_scores(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 4,
    ) -> tuple[str, dict[str, float] | None]:
        """Greedy answer plus letter probabilities derived from API logprobs."""

        try:
            completion = self._create(
                messages,
                max_tokens=max(1, int(max_new_tokens)),
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
                stream=False,
            )
        except Exception:
            text = self.generate(messages, max_new_tokens=max_new_tokens, temperature=0.0)
            return text, None
        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        probabilities = self._letter_probabilities_from_logprobs(choice)
        return text, probabilities

    @staticmethod
    def _letter_probabilities_from_logprobs(choice: Any) -> dict[str, float] | None:
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None or not getattr(logprobs, "content", None):
            return None
        best: dict[str, float] = {}
        for token_info in logprobs.content[:3]:
            for candidate in getattr(token_info, "top_logprobs", None) or []:
                token = str(getattr(candidate, "token", "")).strip()
                if token and token[0].upper() in LETTERS:
                    letter = token[0].upper()
                    value = float(getattr(candidate, "logprob", -30.0))
                    if letter not in best or value > best[letter]:
                        best[letter] = value
        if not best:
            return None
        import math

        peak = max(best.values())
        weights = {letter: math.exp(value - peak) for letter, value in best.items()}
        total = sum(weights.values())
        return {letter: weight / total for letter, weight in weights.items()}

    def option_token_ids(self, letters: list[str]) -> dict[str, list[int]]:
        return {}

    def empty_cache(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "api",
            "model": self.model,
            "clip_renders": self.clip_renders,
            "clip_cache_dir": str(self.clip_cache_dir),
            "proxy_policy": self.proxy_policy,
        }
