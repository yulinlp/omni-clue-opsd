#!/usr/bin/env python3
"""Grade an explicit MCQ option decision separately from strict output format.

Only a clearly marked A-D decision is scored. Explanatory text is neither
rewarded nor used to infer the choice. Legacy parsing stays available as an
unchanged diagnostic; training rewards and old evaluation files are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training_code/src"))
from omni_opsd.evaluation import _input_signature, scored_mcq_rows  # noqa: E402
from omni_opsd.rewards import completion_text, extract_mcq_answer  # noqa: E402

SCORING = "mcq-v2: conservative explicit option decision; explanation content is not graded"
FINAL_PREFIX = re.compile(r"\b(?:final\s+answer|answer)\b\s*(?:is\s+|[:=]\s*|(?=[A-Da-d](?!\w)))", re.I)


def _leading_choice(text: str) -> tuple[str, str] | None:
    text = text.strip()
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper(), "bare-letter"
    # Uppercase option markers prevent lower-case event labels such as (a)
    # from being interpreted as a final choice.
    match = re.match(r"^([A-D])[.):](?=\s|$|[^\w])", text)
    if match:
        return match.group(1), "leading-option-marker"
    match = re.match(r"^(?:\(([A-D])\)|\[([A-D])\])(?=\s|$|[^\w])", text)
    if match:
        return match.group(1) or match.group(2), "leading-bracketed-option"
    match = re.match(r"^([A-D])\s+(\(.*)$", text, re.S)
    if match:
        return match.group(1), "leading-option-parenthesized-content"
    match = re.match(r"^([A-D])\s+([^\s].*)$", text, re.S)
    if match and unicodedata.category(match.group(2)[0]) == "So":
        return match.group(1), "leading-option-emoji"
    return None


def _outside_analysis(text: str) -> str:
    # Completed analysis blocks never contribute option decisions.
    text = re.sub(r"<analysis\s*>.*?</analysis\s*>", "\n", text, flags=re.S | re.I)
    opening = re.search(r"<analysis\s*>", text, re.I)
    if opening:
        tail = text[opening.end():]
        boundaries = [match.start() + opening.end() for match in
                      list(re.finditer(r"<answer\s*>", tail, re.I)) + list(FINAL_PREFIX.finditer(tail))]
        if boundaries:
            text = text[:opening.start()] + "\n" + text[min(boundaries):]
        else:
            text = text[:opening.start()]
    # Plain "Analysis:" followed by a final-answer declaration is also
    # reasoning, so a mentioned option there is not a competing final choice.
    plain = re.match(r"^\s*(?:analysis|thinking)\s*[:\n]", text, re.I)
    if plain:
        tail = text[plain.end():]
        boundaries = [match.start() + plain.end() for match in
                      list(re.finditer(r"<answer\s*>", tail, re.I)) + list(FINAL_PREFIX.finditer(tail))]
        text = text[min(boundaries):] if boundaries else ""
    return text.strip()


def parse_mcq_choice(value) -> dict:
    raw = completion_text(value).strip()
    text = _outside_analysis(unicodedata.normalize("NFKC", raw))
    candidates = []
    leading = _leading_choice(text)
    if leading:
        candidates.append({"letter": leading[0], "source": leading[1], "offset": 0})
    # Explicit answer tags can contain a marked choice plus explanation. A
    # complete or uniquely unfinished answer is accepted, without looking into
    # analysis for a missing final choice.
    tag_openings = list(re.finditer(r"<answer\s*>", text, re.I))
    for opening in tag_openings:
        close = re.search(r"</answer\s*>", text[opening.end():], re.I)
        end = opening.end() + close.start() if close else len(text)
        body = text[opening.end():end].strip()
        if re.search(r"<\s*/?\s*(?:analysis|answer)\b", body, re.I):
            continue
        result = _leading_choice(body)
        if result:
            candidates.append({"letter": result[0], "source": "answer-tag", "offset": opening.start()})
    for match in FINAL_PREFIX.finditer(text):
        # Keep the declaration through its line; the leading marker reader
        # rejects article-like "A dog" and word fragments such as "can".
        body = text[match.end():].split("\n", 1)[0].strip()
        result = _leading_choice(body)
        if result:
            candidates.append({"letter": result[0], "source": "explicit-answer-declaration", "offset": match.start()})
    # A second plainly marked choice on another line or after an explicit
    # alternative connector makes the final decision ambiguous. Parenthesized
    # event sequences inside option content are deliberately not scanned here.
    competing = re.compile(r"(?:^|\n|[,;]\s*|\b(?:or|and|but\s+(?:also\s+)?)\s+)([A-D])[.):](?=\s|$|[^\w])")
    for match in competing.finditer(text):
        candidates.append({"letter": match.group(1), "source": "marked-choice-list", "offset": match.start(1)})
    # Two bare alternatives ('A or B') do not designate one final choice.
    bare_alternatives = re.fullmatch(r"\s*([A-D])\s+(?:or|and)\s+([A-D])\s*[.!?]?\s*", text)
    if bare_alternatives:
        for letter in bare_alternatives.groups():
            candidates.append({"letter": letter, "source": "bare-choice-alternatives", "offset": bare_alternatives.start()})
    letters = sorted({candidate["letter"] for candidate in candidates})
    prediction = letters[0] if len(letters) == 1 else None
    reason = "explicit-option-decision" if prediction else ("conflicting-explicit-options" if len(letters) > 1 else "no-explicit-option-decision")
    return {"prediction": prediction, "parsed": prediction is not None, "parse_reason": reason,
            "choice_candidates": candidates, "conflicting_letters": letters if len(letters) > 1 else [],
            "strict_format_compliant": bool(re.fullmatch(r"[A-Da-d]", raw)),
            "legacy_prediction": extract_mcq_answer(value)}


def scored_mcq_v2_rows(results: list[dict], labels: list[dict], sources: list[dict]) -> list[dict]:
    # Reuse the unchanged exact-ID/signature checks, without adopting legacy
    # parsing as the primary prediction.
    legacy = {row["sample_id"]: row for row in scored_mcq_rows(results, labels, sources)}
    signatures = {_input_signature(row): row.get("case_id") or row.get("prompt_id") for row in sources}
    mapped = {(row.get("case_id") or row.get("prompt_id") or signatures.get(_input_signature(row))): row for row in results}
    labelmap = {row["sample_id"]: row for row in labels}
    scored = []
    for key in sorted(labelmap):
        answer = str(labelmap[key]["answer"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError("WorldSense MCQ label must be A-D: " + key)
        response = completion_text(mapped[key].get("response", ""))
        parsed = parse_mcq_choice(response)
        record = {"sample_id": key, "response": response, "gold_answer": answer, **parsed,
                  "correct": parsed["prediction"] == answer if parsed["prediction"] is not None else False,
                  "legacy_correct": legacy[key]["correct"],
                  "legacy_parsed": legacy[key]["prediction"] is not None}
        for field in ("video_id", "question_type", "tier"):
            if field in labelmap[key]:
                record[field] = labelmap[key][field]
        scored.append(record)
    return scored


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        raise ValueError("No prediction rows")
    parsed = sum(row["parsed"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    legacy_parsed = sum(row["legacy_parsed"] for row in rows)
    legacy_correct = sum(row["legacy_correct"] for row in rows)
    compliance = sum(row["strict_format_compliant"] for row in rows)
    return {"scoring": SCORING, "total": n, "parsed": parsed, "parse_failures": n - parsed,
            "parse_failure_ids": [row["sample_id"] for row in rows if not row["parsed"]],
            "correct": correct, "accuracy": correct / n, "parse_rate": parsed / n,
            "strict_format_compliant": compliance, "strict_format_rate": compliance / n,
            "legacy_strict_parsed": legacy_parsed, "legacy_strict_parse_failures": n - legacy_parsed,
            "legacy_strict_correct": legacy_correct, "legacy_strict_accuracy": legacy_correct / n,
            "legacy_strict_parse_rate": legacy_parsed / n,
            "v2_recovered_from_legacy_unparsed": sum(row["parsed"] and not row["legacy_parsed"] for row in rows),
            "v2_rejected_legacy_parsed": sum(not row["parsed"] and row["legacy_parsed"] for row in rows),
            "prediction_counts": dict(sorted(Counter(row["prediction"] or "UNPARSED" for row in rows).items())),
            "parse_reason_counts": dict(sorted(Counter(row["parse_reason"] for row in rows).items())),
            "strict_unparsed_counted_wrong": True, "identical_complete_sample_ids": True,
            "primary_definition": "Correctness of one unambiguous explicitly selected A-D option; explanation content receives no score.",
            "format_definition": "Exactly one option letter and no other text, case insensitive after surrounding whitespace is stripped.",
            "legacy_note": "Legacy strict parser is preserved unchanged as a diagnostic and may accept unsafe word fragments; it is not the primary v2 decision."}


def read(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--adapter", default="")
    args = parser.parse_args()
    rows = scored_mcq_v2_rows(read(args.results), read(args.labels), read(args.dataset))
    if len(rows) != 518:
        raise ValueError(f"This WorldSense evaluation requires 518 rows, got {len(rows)}")
    summary = summarize(rows)
    summary.update(arm=args.arm, adapter=args.adapter or None,
                   parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name, path in (("results", args.results), ("labels", args.labels), ("dataset", args.dataset)):
        summary[name] = str(path.resolve())
        summary[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored_path = args.output.parent / "mcq_scored_v2.jsonl"
    scored_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary = args.output.with_name(args.output.name + ".mcq_v2.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
