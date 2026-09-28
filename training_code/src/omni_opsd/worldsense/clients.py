"""Model clients for the controller: a scripted mock and the Qwen-Omni runtime."""

from __future__ import annotations

import copy
import json
from typing import Any, Protocol

from .views import render_conversation


class OmniClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str: ...


class MockOmniClient:
    """Scripted controller for tests and CPU dry-runs."""

    def __init__(self, scripted_responses: list[str] | None = None):
        self.scripted = list(scripted_responses or [])
        self.calls: list[list[dict[str, Any]]] = []
        self.rendered: list[tuple[list[dict[str, Any]], bool]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append(copy.deepcopy(messages))
        self.rendered.append(render_conversation(messages))
        if self.scripted:
            return self.scripted.pop(0)
        return json.dumps(
            {
                "observation": "script exhausted; no evidence found",
                "action": "submit",
                "intervals": [],
                "confidence": 0.0,
            },
            ensure_ascii=False,
        )

    def empty_cache(self) -> None:
        return None


class TransformersOmniClient:
    """Qwen3-Omni / Qwen2.5-Omni controller over the Transformers processor.

    The harness renders each stored media view into the processor's structured
    video/audio descriptors right before generation, so the model's requested
    fps / max_pixels / modality are honoured exactly.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        talker: bool = False,
    ):
        import torch
        from transformers import AutoConfig, AutoProcessor

        self.torch = torch
        self.device = device
        self.talker = talker
        config = AutoConfig.from_pretrained(model_path)
        self.model_type = str(getattr(config, "model_type", ""))
        if self.model_type == "qwen3_omni_moe":
            from transformers import Qwen3OmniMoeForConditionalGeneration as ModelCls
        elif self.model_type == "qwen2_5_omni":
            from transformers import Qwen2_5OmniForConditionalGeneration as ModelCls
        else:
            raise ValueError(f"unsupported model_type for the omni client: {self.model_type}")
        torch_dtype = getattr(torch, dtype)
        device_map = "auto" if str(device).lower() == "auto" else device
        self.model = ModelCls.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        if self.model_type == "qwen3_omni_moe" and not talker and hasattr(self.model, "disable_talker"):
            # Qwen3-Omni's talker intermittently crashes the native stack when it
            # stays resident; the localization/verification stages only need text.
            try:
                self.model.disable_talker()
            except Exception:
                pass
        self.processor = AutoProcessor.from_pretrained(model_path)
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except Exception:
            self.input_device = next(self.model.parameters()).device

    def _prepare_inputs(self, messages: list[dict[str, Any]]):
        from qwen_omni_utils import process_mm_info

        rendered, use_audio_in_video = render_conversation(messages)
        text = self.processor.apply_chat_template(
            rendered, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            rendered, use_audio_in_video=use_audio_in_video
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.input_device).to(self.model.dtype)
        return inputs, use_audio_in_video

    def _generate_ids(self, inputs, use_audio_in_video: bool, max_new_tokens: int, temperature: float):
        do_sample = bool(temperature and temperature > 0)
        base: dict[str, Any] = {
            "use_audio_in_video": bool(use_audio_in_video),
            "thinker_max_new_tokens": int(max_new_tokens),
            "max_new_tokens": int(max_new_tokens),
        }
        if self.model_type == "qwen3_omni_moe":
            base["return_audio"] = bool(self.talker)
        elif self.model_type == "qwen2_5_omni":
            base["return_audio"] = bool(self.talker)
        if do_sample:
            base["do_sample"] = True
            base["temperature"] = float(temperature)
        else:
            base["do_sample"] = False
        with self.torch.inference_mode():
            try:
                out = self.model.generate(**inputs, **base)
            except TypeError:
                base.pop("max_new_tokens", None)
                base.pop("thinker_max_new_tokens", None)
                base["max_new_tokens"] = int(max_new_tokens)
                out = self.model.generate(**inputs, **base)
        return out[0] if isinstance(out, tuple) else out

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        inputs, use_audio_in_video = self._prepare_inputs(messages)
        output_ids = self._generate_ids(inputs, use_audio_in_video, max_new_tokens, temperature)
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = output_ids[:, prompt_length:]
        if generated.numel() == 0:
            generated = output_ids
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip()

    def generate_with_scores(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 4,
    ) -> tuple[str, Any]:
        """Greedy generation that also returns the first-step logits.

        The scores are used by the sufficiency verifier to obtain calibrated
        per-option probabilities without a second forward pass.  When the model
        runtime does not expose scores the text result is still returned.
        """

        inputs, use_audio_in_video = self._prepare_inputs(messages)
        do_sample = False
        base: dict[str, Any] = {
            "use_audio_in_video": bool(use_audio_in_video),
            "thinker_max_new_tokens": int(max_new_tokens),
            "max_new_tokens": int(max_new_tokens),
            "return_audio": bool(self.talker),
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        with self.torch.inference_mode():
            try:
                output = self.model.generate(**inputs, **base)
            except TypeError:
                base.pop("return_dict_in_generate", None)
                base.pop("output_scores", None)
                base.pop("max_new_tokens", None)
                output = self.model.generate(**inputs, **base)
        sequences = getattr(output, "sequences", None)
        scores = getattr(output, "scores", None)
        if sequences is None:
            sequences = output[0] if isinstance(output, tuple) else output
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = sequences[:, prompt_length:]
        if generated.numel() == 0:
            generated = sequences
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        first_scores = None
        if scores:
            first_scores = scores[0][0]
        return decoded[0].strip(), first_scores

    def option_token_ids(self, letters: list[str]) -> dict[str, list[int]]:
        """Token ids whose next-token probability represents an option letter."""

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        mapping: dict[str, list[int]] = {}
        for letter in letters:
            ids: set[int] = set()
            for text in (letter, " " + letter, letter.lower(), " " + letter.lower()):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
            mapping[letter] = sorted(ids)
        return mapping

    def empty_cache(self) -> None:
        """Release cached CUDA blocks so a downgraded retry has room to run."""

        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass
