"""Process-start compatibility hooks for the Ascend training runners.

Python imports ``sitecustomize`` automatically when this directory is on
``PYTHONPATH``.  The full-video GKD arms deliberately disable gradient
clipping on Ascend 910B1 because the platform's ``LpNormV2`` kernel can run
out of HBM even when no clipping is requested.  Recent Transformers still
compute an unclipped norm (``max_norm=inf``) solely for logging, so skip that
diagnostic call in the two safe-mode GKD arms.
"""

from __future__ import annotations

import os
import weakref


def _patch_qwen_omni_logits_to_keep() -> None:
    """Avoid materialising vocabulary logits for masked multimodal tokens.

    Qwen2.5-Omni's Transformers 4.x thinker computes the lm-head over every
    visual/prompt position even when Swift has already reduced ``labels`` to
    the supervised assistant span.  A 24k-token video therefore creates a
    second ~16 GiB allocation during cross entropy in full-parameter SFT.
    Swift's ``--use_logits_to_keep`` supplies a bool/index mask; slice the
    hidden states immediately before the lm-head so the loss and labels stay
    aligned while the full multimodal context still participates in attention.
    The patch is opt-in for SFT/GKD arms.  GKD has separate student and
    answer-privileged teacher encodings, so both forwards need the same
    multimodal-safe lm-head slicing to avoid materialising full-vocabulary
    logits for visual/prompt positions.
    """
    if os.environ.get("OMNI_OPSD_ARM") not in {"sft", "opsd", "clue_opsd"}:
        return
    if os.environ.get("OMNI_OPSD_USE_LOGITS_TO_KEEP", "0") != "1":
        return

    classes = []
    for module_name in (
            "transformers.models.qwen2_5_omni.modular_qwen2_5_omni",
            "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"):
        try:
            module = __import__(module_name, fromlist=["Qwen2_5OmniThinkerForConditionalGeneration"])
            cls = getattr(module, "Qwen2_5OmniThinkerForConditionalGeneration")
        except Exception:
            continue
        if cls not in classes:
            classes.append(cls)

    # Keep dependency probes and non-Omni commands resilient.
    if not classes:
        return

    def _patch_class(cls):
        origin_init = cls.__init__
        origin_forward = cls.forward
        if getattr(origin_forward, "_omni_opsd_logits_to_keep_patch", False):
            return

        def _lm_head_pre_hook(module, args):
            if not args:
                return args
            parent_ref = getattr(module, "_omni_opsd_parent_ref", None)
            parent = parent_ref() if parent_ref is not None else None
            keep = getattr(parent, "_omni_opsd_logits_to_keep", None) if parent is not None else None
            if keep is None:
                return args
            hidden_states = args[0]
            if isinstance(keep, int):
                index = slice(-keep, None) if keep > 0 else slice(None)
            else:
                index = keep
            return (hidden_states[:, index, :], *args[1:])

        def _patched_init(self, *args, **kwargs):
            origin_init(self, *args, **kwargs)
            # The loader constructs one thinker per rank, so an instance-local
            # hook avoids changing evaluation or unrelated model instances.
            if not getattr(self.lm_head, "_omni_opsd_logits_hook", False):
                # A normal Module attribute would register ``self`` as a
                # child of lm_head and create a recursive module graph.
                self.lm_head._omni_opsd_parent_ref = weakref.ref(self)
                self.lm_head.register_forward_pre_hook(_lm_head_pre_hook)
                self.lm_head._omni_opsd_logits_hook = True

        def _patched_forward(self, *args, **kwargs):
            keep = kwargs.pop("logits_to_keep", None)
            if keep is None:
                return origin_forward(self, *args, **kwargs)
            # Swift trims labels to the supervised suffix for a single sample.
            # Using that suffix width is robust across Transformers 4.x Omni
            # wrappers, whose bool-mask convention differs between releases.
            labels = kwargs.get("labels")
            if labels is not None and getattr(labels, "ndim", 0) >= 2:
                keep = int(labels.shape[-1])
            self._omni_opsd_logits_to_keep = keep
            try:
                return origin_forward(self, *args, **kwargs)
            finally:
                self._omni_opsd_logits_to_keep = None

        _patched_forward._omni_opsd_logits_to_keep_patch = True
        cls.__init__ = _patched_init
        cls.forward = _patched_forward

    for cls in classes:
        _patch_class(cls)


def _patch_gkd_grad_norm_logging() -> None:
    if os.environ.get("OMNI_OPSD_GKD_SAFE_MODE", "1") != "1":
        return
    if os.environ.get("OMNI_OPSD_ARM") not in {"opsd", "clue_opsd"}:
        return

    try:
        from transformers.trainer import Trainer
    except Exception:
        # Keep interpreter startup resilient in dependency-validation commands.
        return

    origin = getattr(Trainer, "_get_grad_norm", None)
    if origin is None:
        # Older Transformers builds do not expose the private helper.  There
        # is nothing to patch in that case, and this must not prevent the
        # independent logits-to-keep memory hook from being installed below.
        return
    if getattr(origin, "_omni_opsd_gkd_safe_patch", False):
        return

    def _get_grad_norm_without_ascend_vector_norm(self, model, grad_norm=None):
        try:
            no_clipping = float(getattr(self.args, "max_grad_norm", 1.0)) <= 0.0
        except (TypeError, ValueError):
            no_clipping = False
        if grad_norm is None and no_clipping:
            # This value is used only for the trainer's logging dictionary.
            return 0.0
        return origin(self, model, grad_norm=grad_norm)

    _get_grad_norm_without_ascend_vector_norm._omni_opsd_gkd_safe_patch = True
    Trainer._get_grad_norm = _get_grad_norm_without_ascend_vector_norm


_patch_gkd_grad_norm_logging()
_patch_qwen_omni_logits_to_keep()
