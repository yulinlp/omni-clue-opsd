"""Unit tests for V1 sufficiency verification (no GPU needed)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omni_opsd.worldsense.schema import QuestionRecord
from omni_opsd.worldsense.verify import (
    VerifierConfig,
    build_sufficiency_messages,
    letter_probabilities,
    parse_option_letter,
    verify_interval,
    verify_question,
)


def make_record(**overrides) -> QuestionRecord:
    data = dict(
        question_id="TESTVID::task0",
        video_id="TESTVID",
        task_index=0,
        task_domain="Recognition",
        task_type="Object Counting",
        question="How many countries are mentioned?",
        candidates=["A. Three.", "B. Five.", "C. Four.", "D. One."],
        video_caption="synopsis",
        domain="Performance",
        sub_category="Talks",
        audio_class=["Speech"],
        duration_s=60.0,
        video_path="/tmp/TESTVID.mp4",
        answer_letter="C",
    )
    data.update(overrides)
    return QuestionRecord(**data)


class FakeClient:
    def __init__(self, text: str, scores=None):
        self.text = text
        self.scores = scores
        self.calls = []

    def generate_with_scores(self, messages, *, max_new_tokens: int = 4):
        self.calls.append(messages)
        return self.text, self.scores

    def option_token_ids(self, letters):
        return {letter: [index] for index, letter in enumerate(letters)}


def test_parse_option_letter_variants() -> None:
    assert parse_option_letter("<answer>B</answer>") == "B"
    assert parse_option_letter("B") == "B"
    assert parse_option_letter("The answer is C.") == "C"
    assert parse_option_letter("answer: d") == "D"
    assert parse_option_letter("I cannot tell") is None
    assert parse_option_letter("") is None


def test_letter_probabilities_softmax() -> None:
    scores = [0.0, 3.0, 1.0, 1.0]  # A, B, C, D
    ids = {"A": [0], "B": [1], "C": [2], "D": [3]}
    probabilities = letter_probabilities(scores, ids)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    assert probabilities["B"] == max(probabilities.values())
    assert probabilities["B"] > probabilities["A"]
    assert letter_probabilities(None, ids) == {}


def test_letter_probabilities_sums_token_variants() -> None:
    scores = [0.0, 5.0, 0.0, 0.0, 5.0]
    ids = {"A": [0], "B": [1, 4], "C": [2], "D": [3]}
    probabilities = letter_probabilities(scores, ids)
    assert probabilities["B"] > 0.98


def test_build_sufficiency_messages_has_media_and_no_answer() -> None:
    record = make_record()
    verifier = VerifierConfig()
    messages = build_sufficiency_messages(record, {"start": 1.0, "end": 6.0}, verifier)
    assert messages[0]["role"] == "system"
    assert "_worldsense_view" in messages[1]
    assert "introduced" not in messages[0]["content"].lower()
    user_text = messages[2]["content"]
    assert "C. Four." in user_text
    assert "correct option letter is" in user_text
    assert "<answer>" not in user_text
    assert "Task type: Object Counting" in user_text
    assert record.answer_letter not in ("", None)


def test_verify_interval_marks_correct_and_scores() -> None:
    record = make_record()
    verifier = VerifierConfig()
    client = FakeClient("<answer>C</answer>", scores=[0.0, 0.0, 4.0, 0.0])
    result = verify_interval(client, record, {"start": 2.0, "end": 8.0}, verifier)
    assert result.error is None
    assert result.answer_letter == "C"
    assert result.correct is True
    assert result.p_true is not None and result.p_true > 0.9
    assert result.p_max == result.p_true
    assert result.margin is not None and result.margin > 0.8
    assert result.entropy is not None and result.entropy >= 0.0


def test_verify_interval_reports_wrong_answer() -> None:
    record = make_record()
    verifier = VerifierConfig()
    client = FakeClient("A", scores=[4.0, 0.0, 0.0, 0.0])
    result = verify_interval(client, record, {"start": 2.0, "end": 8.0}, verifier)
    assert result.correct is False
    assert result.predicted_letter == "A"
    assert result.p_true is not None and result.p_true < 0.2


def test_verify_question_selects_best_interval() -> None:
    record = make_record()
    verifier = VerifierConfig()

    class Router(FakeClient):
        def generate_with_scores(self, messages, *, max_new_tokens: int = 4):
            view = messages[1]["_worldsense_view"]
            if view["start"] == 10.0:
                return "<answer>C</answer>", [0.0, 0.0, 2.0, 0.0]
            return "<answer>A</answer>", [2.0, 0.0, 0.0, 0.0]

    summary = verify_question(
        Router("", None),
        record,
        [{"start": 1.0, "end": 5.0}, {"start": 10.0, "end": 15.0}],
        verifier,
    )
    assert summary["n_intervals"] == 2
    assert summary["n_correct"] == 1
    assert summary["any_correct"] is True
    assert summary["best_interval"] == {"start": 10.0, "end": 15.0}


def test_verify_question_without_gold_letter() -> None:
    record = make_record(answer_letter=None)
    verifier = VerifierConfig()
    summary = verify_question(
        FakeClient("<answer>B</answer>", [0.0, 2.0, 0.0, 0.0]),
        record,
        [{"start": 1.0, "end": 5.0}],
        verifier,
    )
    assert summary["any_correct"] is None
    assert summary["results"][0]["correct"] is None
    assert summary["results"][0]["predicted_letter"] == "B"


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
