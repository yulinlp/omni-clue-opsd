"""Portable manifest schema used by every active benchmark adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from omni_opsd.temporal.evidence import EvidenceSpan


@dataclass
class ModalityEvidence:
    visual: list[EvidenceSpan] = field(default_factory=list)
    audio: list[EvidenceSpan] = field(default_factory=list)
    subtitle: list[EvidenceSpan] = field(default_factory=list)
    provenance: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual": [span.as_list() for span in self.visual],
            "audio": [span.as_list() for span in self.audio],
            "subtitle": [span.as_list() for span in self.subtitle],
            "provenance": self.provenance,
        }


@dataclass
class CanonicalSample:
    sample_id: str
    benchmark: str
    video_id: str
    video_path: str
    question: str
    choices: list[str]
    answer: str | None
    duration: float | None = None
    question_type: str = "unknown"
    category_provenance: str = "dataset"
    subtitle_path: str | None = None
    evidence: ModalityEvidence = field(default_factory=ModalityEvidence)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        row = asdict(self)
        row["evidence"] = self.evidence.to_dict()
        # Compatibility alias consumed by the temporal experiment planner.
        row["evidence_spans"] = row["evidence"]["visual"]
        return row
