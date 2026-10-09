"""CPU regressions for unresolved semantic grades and complete-ID aggregation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/aggregate_worldsense_training_matched_eval.py"
spec = importlib.util.spec_from_file_location("matched_aggregate", SCRIPT)
aggregation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregation)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class UnresolvedGradeTests(unittest.TestCase):
    def test_null_is_an_interval_not_an_incorrect_grade(self):
        metrics = aggregation.score_metrics([True, None, False, None])
        self.assertEqual(metrics["confirmed_correct"], 1)
        self.assertEqual(metrics["confirmed_incorrect"], 1)
        self.assertEqual(metrics["undecided"], 2)
        self.assertIsNone(metrics["accuracy"])
        self.assertEqual(metrics["accuracy_lower_bound"], 0.25)
        self.assertEqual(metrics["accuracy_upper_bound"], 0.75)

    def test_non_boolean_score_is_rejected(self):
        for bad in (0, 1, "YES", "NO"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                aggregation.score_metrics([True, bad])

    def test_paired_bounds_and_identical_unresolved_model(self):
        ids = ["q0", "q1", "q2", "q3"]
        labels = {key: {"video_id": f"v{i // 2}"} for i, key in enumerate(ids)}
        bootstrap = aggregation.VideoBootstrap(ids, labels)
        base = [True, None, False, True]
        model = [True, False, None, False]
        comparison = aggregation.paired_comparison(model, base, bootstrap)
        self.assertIsNone(comparison["delta_accuracy"])
        self.assertIsNone(comparison["ci95_video_bootstrap"])
        self.assertEqual(comparison["delta_accuracy_lower_bound"], -0.5)
        self.assertEqual(comparison["delta_accuracy_upper_bound"], 0.0)
        self.assertEqual(comparison["resolved_pairs"], 2)
        self.assertEqual(comparison["known_resolved_pair_difference_contribution_full_denominator"], -0.25)
        self.assertEqual(bootstrap.draws.shape, (5000, 2))
        same = aggregation.paired_comparison(base, base, bootstrap, identical_model=True)
        self.assertEqual(same["delta_accuracy"], 0.0)
        self.assertEqual(same["ci95_video_bootstrap"], [0.0, 0.0])

    def test_strict_summary_cannot_publish_null_as_zero(self):
        metrics = aggregation.score_metrics([True, None])
        with self.assertRaises(ValueError):
            aggregation.check_summary({"total": 2, "correct": 1, "accuracy": 0.5}, metrics, "openqa")

    def test_old_mcq_is_reparsed_without_replacing_its_original_grade(self):
        # This actual observed response was omitted by the old parser. The
        # comparison must use the new option decision while keeping the old
        # stored summary intact, even when the two accuracies differ.
        with tempfile.TemporaryDirectory(prefix="old-mcq-reparse-", dir="/tmp") as temporary:
            root = Path(temporary)
            labels = [{"sample_id": "q", "answer": "C", "video_id": "v"}]
            sources = [{"case_id": "q", "messages": [{"role": "user", "content": "Does the event occur?"}], "videos": []}]
            write_jsonl(root / "mcq/base/results.jsonl", [{"case_id": "q", "response": "C. No, they do not."}])
            summary_path = root / "mcq/base/summary.json"
            write_json(summary_path, {"total": 1, "correct": 0, "accuracy": 0.0})
            before = summary_path.read_bytes()
            loaded = aggregation.load_model(root, "mcq", "base", ["q"], {"q": {"video_id": "v"}},
                                            labels, sources, None, require_v2=False)
            self.assertEqual(loaded[2]["accuracy"], 1.0)
            self.assertEqual(loaded[2]["legacy_strict_accuracy"], 0.0)
            self.assertEqual(summary_path.read_bytes(), before)

    def test_partial_table_retains_pending_tasks_and_real_mcq_schema(self):
        repo = SCRIPT.parents[2]
        source = repo / "training_runs/worldsense_training_matched_eval_20261003"
        mc_input = aggregation.read_rows(source / "data/worldsense.answer_free.jsonl")[:4]
        ids = {row["case_id"] for row in mc_input}
        open_labels = [row for row in aggregation.read_rows(source / "data/worldsense.openqa.labels.jsonl") if row["sample_id"] in ids]
        mc_labels = [row for row in aggregation.read_rows(source / "data/worldsense.labels.jsonl") if row["sample_id"] in ids]
        tasks = json.loads((source / "tasks.json").read_text())
        with tempfile.TemporaryDirectory(prefix="matched-aggregation-", dir="/tmp") as temporary:
            root = Path(temporary)
            write_json(root / "tasks.json", tasks)
            write_jsonl(root / "data/worldsense.answer_free.jsonl", mc_input)
            write_jsonl(root / "data/worldsense.labels.jsonl", mc_labels)
            write_jsonl(root / "data/worldsense.openqa.labels.jsonl", open_labels)
            write_json(root / "data/openqa_manifest.json", {"generation_max_new_tokens": 512})
            scores = [True, None, False, None]
            rows = []
            for label, score in zip(open_labels, scores):
                response = "<analysis>Evidence supports this answer.</analysis><answer>Six.</answer>"
                rows.append(dict(sample_id=label["sample_id"], video_id=label["video_id"], tier=label["tier"],
                                 question_type=label["question_type"], response=response, final_answer="Six.",
                                 analysis="Evidence supports this answer.", analysis_words=5,
                                 format_ok=True, valid_answer=True, semantic_correct=score))
            write_jsonl(root / "openqa/base/scored.jsonl", rows)
            write_json(root / "openqa/base/summary.json", {"total": 4, "correct": 1, "incorrect": 1,
                       "uncertain": 2, "accuracy": None, "scoring": "v2: synthetic fixture", "format_rate": 1.0,
                       "judge_calibration": {"evaluated": True}, "scoring_pipeline_reliable_on_calibration": True})
            # Swift results omit custom IDs and append an assistant message.
            # The real parser must join them by question and video signature.
            lettermap = {row["sample_id"]: row["answer"] for row in mc_labels}
            results = []
            for i, row in enumerate(mc_input):
                response = lettermap[row["case_id"]] if i < 3 else "no usable option"
                results.append({"response": response, "labels": None, "logprobs": None,
                                "messages": row["messages"] + [{"role": "assistant", "content": response}],
                                "videos": row["videos"], "dataset": "synthetic/shards/card_0/input.jsonl"})
            write_jsonl(root / "mcq/base/results.jsonl", results)
            write_json(root / "mcq/base/summary.json", {"total": 4, "correct": 3, "accuracy": 0.75, "parse_rate": 0.75,
                                                       "scoring": "mcq-v2: synthetic fixture"})
            report, table, _ = aggregation.build_report(root, None, True, None, expected_rows=4)
            self.assertEqual(len(table), 26)
            self.assertEqual(report["completed_tasks"], 2)
            self.assertEqual(report["pending_tasks"], 24)
            self.assertEqual(report["unresolved_semantic_grades"], 2)
            self.assertFalse(report["all_semantic_grades_resolved"])
            opened = next(row for row in table if row["mode"] == "openqa" and row["model"] == "base")
            self.assertIsNone(opened["accuracy"])
            self.assertEqual(opened["accuracy_lower_bound"], 0.25)
            self.assertEqual(opened["accuracy_upper_bound"], 0.75)
            self.assertEqual(opened["status"], "provisional-unresolved")
            mcq = next(row for row in table if row["mode"] == "mcq" and row["model"] == "base")
            self.assertEqual(mcq["accuracy"], 0.75)
            self.assertEqual(mcq["parse_or_format_rate"], 0.75)
            with self.assertRaises(ValueError):
                aggregation.build_report(root, None, False, None, expected_rows=4)


class UniformAutomaticOpenQATests(unittest.TestCase):
    def make_row(self, key, grade, reviewed=False):
        row = {"sample_id": key, "video_id": "v", "semantic_correct": grade,
               "scoring_method": "dual-local-judge",
               "response": "<analysis>Evidence is clear.</analysis><answer>Six.</answer>",
               "final_answer": "Six.", "valid_answer": True}
        if reviewed:
            row.update(scoring_method="assistant-review",
                       assistant_adjudication={"semantic_correct": grade, "reviewer_type": "assistant"},
                       judge_reviews=[{"verdict": "YES"}, {"verdict": "NO"}])
        return row

    def prepare_model(self, root, name, rows, summary_updates=None):
        metrics = aggregation.score_metrics([row["semantic_correct"] for row in rows])
        summary = {"total": metrics["total"], "correct": metrics["confirmed_correct"],
                   "incorrect": metrics["confirmed_incorrect"], "uncertain": metrics["undecided"],
                   "accuracy": metrics["accuracy"], "accuracy_lower_bound": metrics["accuracy_lower_bound"],
                   "accuracy_upper_bound": metrics["accuracy_upper_bound"], "scoring": "v2: synthetic fixture",
                   "judge_calibration": {"evaluated": True}, "scoring_pipeline_reliable_on_calibration": True}
        summary.update(summary_updates or {})
        write_jsonl(root / "openqa" / name / "scored.jsonl", rows)
        write_json(root / "openqa" / name / "summary.json", summary)

    def load(self, root, name, rows, require_v2=True):
        ids = [row["sample_id"] for row in rows]
        return aggregation.load_model(root, "openqa", name, ids,
                                      {key: {"video_id": "v"} for key in ids}, [], [], None, require_v2)

    def test_base_seven_positive_three_negative_reviews_recover_ten_nulls(self):
        with tempfile.TemporaryDirectory(prefix="uniform-openqa-", dir="/tmp") as temporary:
            root = Path(temporary)
            base_rows = [self.make_row(f"q{i}", i < 7, reviewed=True) for i in range(10)]
            control_rows = [self.make_row(f"q{i}", None) for i in range(10)]
            self.prepare_model(root, "base", base_rows)
            self.prepare_model(root, "control", control_rows)
            originals = {path: path.read_bytes() for path in (root / "openqa").glob("*/*")}
            base = self.load(root, "base", base_rows)
            control = self.load(root, "control", control_rows)
            self.assertEqual(base[0], [None] * 10)
            self.assertEqual(base[0], control[0])
            self.assertEqual(base[2]["confirmed_correct"], 0)
            self.assertEqual(base[2]["undecided"], 10)
            self.assertEqual(base[2]["primary_scoring"], aggregation.OPENQA_PRIMARY_SCORING)
            sensitivity = base[2]["reviewed_sensitivity"]
            self.assertEqual(sensitivity["confirmed_correct"], 7)
            self.assertEqual(sensitivity["confirmed_incorrect"], 3)
            self.assertEqual(sensitivity["accuracy"], 0.7)
            self.assertEqual(sensitivity["changed_grade_count"], 10)
            self.assertEqual(base[2]["assistant_review_count"], 10)
            self.assertEqual(control[2]["reviewed_sensitivity"]["undecided"], 10)
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_reference_insufficiency_cannot_be_overridden_by_review(self):
        row = self.make_row("q", True, reviewed=True)
        row.pop("judge_reviews")
        row["reference_insufficiency"] = {"problem": "missing event mapping"}
        row["semantic_correct_before_reference_audit"] = False
        with tempfile.TemporaryDirectory(prefix="insufficient-reference-", dir="/tmp") as temporary:
            root = Path(temporary)
            self.prepare_model(root, "base", [row])
            loaded = self.load(root, "base", [row])
            self.assertEqual(loaded[0], [None])
            self.assertEqual(loaded[2]["reviewed_sensitivity"]["undecided"], 1)
            self.assertEqual(loaded[2]["reviewed_sensitivity"]["reference_insufficient_review_overrides_ignored"], 1)

    def test_missing_original_evidence_is_rejected(self):
        row = self.make_row("q", True, reviewed=True)
        row.pop("judge_reviews")
        with self.assertRaisesRegex(ValueError, "lacks an explicit original"):
            aggregation.automatic_openqa_grade(row)
        row["judge_reviews"] = [{"verdict": "YES"}]
        with self.assertRaisesRegex(ValueError, "Invalid original dual judge"):
            aggregation.automatic_openqa_grade(row)

    def test_explicit_original_grade_is_supported_and_conflicts_are_rejected(self):
        row = self.make_row("q", True, reviewed=True)
        row.pop("judge_reviews")
        row["semantic_correct_before_adjudication"] = None
        self.assertIsNone(aggregation.automatic_openqa_grade(row))
        row["judge_reviews"] = [{"verdict": "YES"}, {"verdict": "YES"}]
        with self.assertRaisesRegex(ValueError, "contradicts dual judge"):
            aggregation.automatic_openqa_grade(row)

    def test_stored_summary_is_validated_before_primary_recovery(self):
        row = self.make_row("q", True, reviewed=True)
        with tempfile.TemporaryDirectory(prefix="review-summary-check-", dir="/tmp") as temporary:
            root = Path(temporary)
            self.prepare_model(root, "base", [row], {"correct": 0})
            with self.assertRaisesRegex(ValueError, "Summary correct disagrees"):
                self.load(root, "base", [row])

    def test_prepare_only_or_missing_calibration_is_pending(self):
        row = self.make_row("q", None)
        for overrides in ({"judge_calibration": {"evaluated": False}},
                          {"scoring_pipeline_reliable_on_calibration": False},
                          {"judge_calibration": {}},
                          {"scoring_pipeline_reliable_on_calibration": None}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory(prefix="pending-calibration-", dir="/tmp") as temporary:
                root = Path(temporary)
                self.prepare_model(root, "base", [row], overrides)
                self.assertIsNone(self.load(root, "base", [row]))

    def test_completed_calibrated_score_keeps_real_uncertainty(self):
        row = self.make_row("q", None)
        with tempfile.TemporaryDirectory(prefix="complete-uncertain-", dir="/tmp") as temporary:
            root = Path(temporary)
            self.prepare_model(root, "base", [row], {"judge_reliable_on_calibration": False})
            loaded = self.load(root, "base", [row])
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded[0], [None])
            self.assertEqual(loaded[2]["undecided"], 1)

    def test_legacy_openqa_preserves_reviewed_values_without_new_gate(self):
        row = self.make_row("q", True, reviewed=True)
        row.pop("judge_reviews")
        with tempfile.TemporaryDirectory(prefix="legacy-reviewed-openqa-", dir="/tmp") as temporary:
            root = Path(temporary)
            self.prepare_model(root, "base", [row], {"scoring": "v1", "judge_calibration": {"evaluated": False}})
            loaded = self.load(root, "base", [row], require_v2=False)
            self.assertEqual(loaded[0], [True])
            self.assertEqual(loaded[2]["accuracy"], 1.0)
            self.assertNotIn("reviewed_sensitivity", loaded[2])

    def test_tiers_pairing_bootstrap_and_csv_all_use_primary_grades(self):
        with tempfile.TemporaryDirectory(prefix="primary-report-", dir="/tmp") as temporary:
            root = Path(temporary)
            labels = [{"sample_id": f"q{i}", "video_id": f"v{i}", "tier": "A" if i == 0 else "B",
                       "question_type": "Count"} for i in range(2)]
            write_json(root / "tasks.json", [{"label": "base"}, {"label": "sft_lora_epoch3", "epoch": 3}])
            write_jsonl(root / "data/worldsense.openqa.labels.jsonl", labels)
            write_jsonl(root / "data/worldsense.labels.jsonl", labels)
            write_jsonl(root / "data/worldsense.answer_free.jsonl", [{"case_id": f"q{i}"} for i in range(2)])
            base_rows = [self.make_row("q0", True, reviewed=True), self.make_row("q1", False)]
            control_rows = [self.make_row("q0", True), self.make_row("q1", False)]
            for rows in (base_rows, control_rows):
                for i, row in enumerate(rows):
                    row["video_id"] = f"v{i}"
            self.prepare_model(root, "base", base_rows)
            self.prepare_model(root, "sft_lora_epoch3", control_rows)
            report, table, _ = aggregation.build_report(root, None, True, None, expected_rows=2)
            base = report["modes"]["openqa"]["models"]["base"]
            self.assertEqual(base["by_tier"]["A"]["undecided"], 1)
            self.assertEqual(base["metrics"]["confirmed_correct"], 0)
            self.assertEqual(base["metrics"]["reviewed_sensitivity"]["confirmed_correct"], 1)
            paired = report["modes"]["openqa"]["paired_vs_base"]["sft_lora_epoch3"]
            self.assertEqual(paired["delta_accuracy_lower_bound"], 0.0)
            self.assertEqual(paired["delta_accuracy_upper_bound"], 0.5)
            self.assertIsNone(paired["ci95_video_bootstrap"])
            self.assertEqual(paired["ci95_video_bootstrap_conservative_envelope"], [0.0, 1.0])
            base_csv = next(row for row in table if row["mode"] == "openqa" and row["model"] == "base")
            self.assertEqual(base_csv["primary_scoring"], aggregation.OPENQA_PRIMARY_SCORING)
            self.assertEqual(base_csv["confirmed_correct"], 0)
            self.assertEqual(base_csv["reviewed_sensitivity_confirmed_correct"], 1)
            self.assertEqual(base_csv["reviewed_sensitivity_accuracy_lower_bound"], 0.5)
            self.assertEqual(base_csv["reviewed_sensitivity_changed_grade_count"], 1)
            self.assertEqual(base_csv["assistant_review_count"], 1)


if __name__ == "__main__":
    unittest.main()
