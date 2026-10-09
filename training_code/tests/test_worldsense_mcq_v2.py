"""Regress real omitted option decisions and ambiguous final-choice replies."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_worldsense_mcq_v2.py"
spec = importlib.util.spec_from_file_location("mcq_v2", SCRIPT)
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


class MCQParserRegressionTests(unittest.TestCase):
    def test_real_formatted_option_decisions_previously_omitted(self):
        examples = {
            "D. Thrilled.": "D", "C. Two astronauts.": "C", "C. No, they do not.": "C",
            "B. Step three.": "B", "D (a), (d), (": "D", "D. Two.": "D",
            "D. From straight to curved.": "D", "C. December 4th.": "C", "C 🎉": "C",
            "A. Three times. But also...": "A",
        }
        for response, expected in examples.items():
            with self.subTest(response=response):
                parsed = grader.parse_mcq_choice(response)
                self.assertEqual(parsed["prediction"], expected)
                self.assertFalse(parsed["strict_format_compliant"])

    def test_unmarked_articles_and_unmarked_numbers_are_not_choices(self):
        for response in ("A dog is standing near the door.", "A Nineteen", "The answer can be inferred from the video.",
                         "Answer: A dog.", "The final answer is Because the man left."):
            with self.subTest(response=response):
                self.assertIsNone(grader.parse_mcq_choice(response)["prediction"])

    def test_conflicting_explicit_final_decisions_are_rejected(self):
        for response in ("A. Four.\nFinal answer: B.", "<answer>A</answer><answer>B</answer>",
                         "A. Four.\nB. Six.", "A or B", "C. Two, or D. Three"):
            with self.subTest(response=response):
                parsed = grader.parse_mcq_choice(response)
                self.assertIsNone(parsed["prediction"])
                self.assertEqual(parsed["parse_reason"], "conflicting-explicit-options")

    def test_analysis_event_letters_do_not_compete_with_final_choice(self):
        examples = {
            "<analysis>(a), (b), (c), (d)</analysis><answer>D</answer>": "D",
            "<analysis>A. seems plausible. B. also occurs.</analysis>Final answer: C.": "C",
            "Analysis: A. could describe the opening event.\nFinal answer: B": "B",
            "<analysis>A. is discussed, B. another event. Final answer: C.": "C",
            "<analysis>The events are (a), (c), (b).": None,
            "D. (A), (B), (C)": "D",
            "The final answer is B": "B",
        }
        for response, expected in examples.items():
            with self.subTest(response=response):
                self.assertEqual(grader.parse_mcq_choice(response)["prediction"], expected)

    def test_explanation_content_does_not_change_option_grade(self):
        labels = [{"sample_id": "q", "answer": "C"}]
        sources = [{"case_id": "q", "messages": [{"role": "user", "content": "Choose A-D"}], "videos": []}]
        results = [{"case_id": "q", "response": "C. No, they do not."}]
        rows = grader.scored_mcq_v2_rows(results, labels, sources)
        summary = grader.summarize(rows)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["legacy_strict_accuracy"], 0.0)
        self.assertEqual(summary["strict_format_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
