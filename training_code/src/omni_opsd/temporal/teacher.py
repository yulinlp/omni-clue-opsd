"""GT-conditioned multi-view teacher probes for the improved OPSD design."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np

from .dataset import Sample
from .media import VideoMetadata
from .opsd import PrefixTeacherTarget, aggregate_prefix_logits
from .views import ViewSpec, build_gem_teacher_views, view_debug_dict


class ViewRunner(Protocol):
    def run_view(self, sample: Sample, view: ViewSpec, metadata: VideoMetadata): ...

    def score_prefix(self, sample: Sample, view: ViewSpec, metadata: VideoMetadata, response_ids: Any, **kwargs: Any): ...


@dataclass
class TeacherViewResult:
    view_name: str
    teacher_role: str | None
    prediction: str
    option_probs: dict[str, float] | None
    confidence: float
    entropy: float | None
    margin: float | None
    stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "view_name": self.view_name,
            "teacher_role": self.teacher_role,
            "prediction": self.prediction,
            "option_probs": self.option_probs,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "margin": self.margin,
            "stats": self.stats,
        }


def _distribution_stats(option_probs: dict[str, float] | None) -> tuple[float, float | None, float | None]:
    if not option_probs:
        return 0.0, None, None
    values = np.asarray(list(option_probs.values()), dtype=np.float64)
    values = values / max(values.sum(), 1e-12)
    entropy = float(-(values * np.log(np.maximum(values, 1e-12))).sum() / max(math.log(len(values)), 1e-12)) if len(values) > 1 else 0.0
    ordered = np.sort(values)[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
    # High probability, low entropy, and a decisive margin are useful teacher
    # confidence signals.  The output remains a distribution, not an answer
    # label or a hard-coded correctness target.
    confidence = float(max(0.0, min(1.0, 0.5 * ordered[0] + 0.3 * (1.0 - entropy) + 0.2 * margin)))
    return confidence, entropy, margin


def _parse_fallback_prediction(text: str, choices: list[str]) -> str | None:
    upper = str(text).upper()
    for index in range(len(choices)):
        letter = chr(ord("A") + index)
        if letter in upper:
            return letter
    return None


def _teacher_weight(result: TeacherViewResult, floor: float = 0.05) -> float:
    if result.option_probs:
        return max(floor, result.confidence)
    return floor


def aggregate_teacher_views(
    results: list[TeacherViewResult],
    *,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Aggregate option distributions with confidence/agreement gating.

    This is an inference-time teacher probe.  It intentionally does not
    consume ``Sample.answer``.  A future differentiable trainer can use the
    returned ``teacher_distribution`` as a target for the student's on-policy
    prefix, while retaining the per-view confidence and agreement masks.
    """

    usable = [item for item in results if item.option_probs and item.confidence >= min_confidence]
    if not usable:
        votes: dict[str, int] = {}
        for item in results:
            if item.prediction:
                votes[item.prediction] = votes.get(item.prediction, 0) + 1
        prediction = max(votes, key=votes.get) if votes else ""
        return {
            "teacher_distribution": None,
            "teacher_prediction": prediction,
            "teacher_confidence": None,
            "view_agreement": None,
            "usable_views": 0,
            "gating": "no_option_logits",
        }
    keys = sorted({key for item in usable for key in item.option_probs or {}})
    aggregate = {key: 0.0 for key in keys}
    total_weight = 0.0
    for item in usable:
        weight = _teacher_weight(item)
        total_weight += weight
        for key in keys:
            aggregate[key] += weight * float((item.option_probs or {}).get(key, 0.0))
    if total_weight:
        aggregate = {key: value / total_weight for key, value in aggregate.items()}
    prediction = max(aggregate, key=aggregate.get)
    top_predictions = [max(item.option_probs or {}, key=(item.option_probs or {}).get) for item in usable]
    agreement = float(sum(pred == prediction for pred in top_predictions) / len(top_predictions))
    aggregate_confidence, entropy, margin = _distribution_stats(aggregate)
    return {
        "teacher_distribution": aggregate,
        "teacher_prediction": prediction,
        "teacher_confidence": aggregate_confidence,
        "teacher_entropy": entropy,
        "teacher_margin": margin,
        "view_agreement": agreement,
        "usable_views": len(usable),
        "total_views": len(results),
        "gating": "confidence_weighted_consensus",
        "distillation_allowed": bool(agreement >= 0.5 and aggregate_confidence > 0.0),
    }


def run_teacher_probe(
    sample: Sample,
    metadata: VideoMetadata,
    config: dict[str, Any],
    runner: ViewRunner,
) -> dict[str, Any]:
    """Run all complementary oracle views and return a training-ready record."""

    views = build_gem_teacher_views(sample, metadata, config)
    outputs: list[TeacherViewResult] = []
    for view in views:
        result = runner.run_view(sample, view, metadata)
        prediction = ""
        if result.option_probs:
            prediction = max(result.option_probs, key=result.option_probs.get)
        if not prediction:
            prediction = _parse_fallback_prediction(result.text, sample.choices) or result.text
        confidence, entropy, margin = _distribution_stats(result.option_probs)
        outputs.append(TeacherViewResult(
            view_name=view.name,
            teacher_role=view.teacher_role,
            prediction=prediction,
            option_probs=result.option_probs,
            confidence=confidence,
            entropy=entropy,
            margin=margin,
            stats={**result.stats, "view_plan": view_debug_dict(view, metadata)},
        ))
    aggregate = aggregate_teacher_views(
        outputs,
        min_confidence=float(config.get("teacher", {}).get("min_confidence", 0.0)),
    )
    return {
        "teacher_views": [item.as_dict() for item in outputs],
        "teacher_ensemble": aggregate,
        "teacher_protocol": {
            "name": "GEM-OPSD",
            "uses_answer": False,
            "uses_golden_temporal_evidence": True,
            "view_count": len(outputs),
            "roles": [view.teacher_role for view in views],
        },
    }


def build_gem_prefix_target(
    sample: Sample,
    metadata: VideoMetadata,
    config: dict[str, Any],
    runner: ViewRunner,
    response_ids: Any,
) -> tuple[PrefixTeacherTarget, list[dict[str, Any]]]:
    """Score one student's on-policy prefix under all GEM teacher views.

    This is the bridge from the inference probe to OPSD training.  The caller
    supplies only the sampled response token IDs; the function never consumes
    ``sample.answer``.  A future optimizer can pass the returned target and
    the student's prefix logits to :func:`gem_generalized_jsd`.
    """
    views = build_gem_teacher_views(sample, metadata, config)
    prefix_logits = []
    audits: list[dict[str, Any]] = []
    configured_weights = config.get("teacher", {}).get("view_weights", {}) or {}
    default_weights = {
        "temporal": 1.0,
        "context": 0.8,
        "spatial": 1.0,
        "audio-context": 0.8,
    }
    weights: list[float] = []
    for view in views:
        logits = runner.score_prefix(
            sample,
            view,
            metadata,
            response_ids,
            adapter=config.get("teacher", {}).get("inference_adapter"),
            no_grad=True,
        )
        prefix_logits.append(logits)
        role = view.teacher_role or "context"
        weights.append(float(configured_weights.get(role, default_weights.get(role, 1.0))))
        audits.append({
            "view_name": view.name,
            "teacher_role": view.teacher_role,
            "view_plan": view_debug_dict(view, metadata),
        })
    target = aggregate_prefix_logits(
        prefix_logits,
        view_weights=weights,
        min_confidence=float(config.get("teacher", {}).get("min_confidence", 0.15)),
        min_agreement=float(config.get("teacher", {}).get("min_agreement", 0.5)),
        temperature=float(config.get("teacher", {}).get("distillation_temperature", 1.0)),
    )
    return target, audits
