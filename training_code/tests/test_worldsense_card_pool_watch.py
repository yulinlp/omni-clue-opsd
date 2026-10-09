"""Small CPU regressions for per-card health, live appends and legacy handoff."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import watch_worldsense_card_pool as watch


class PoolWatchTests(unittest.TestCase):
    def fixture(self, temporary):
        root = Path(temporary)
        (root / "data").mkdir()
        (root / "logs").mkdir()
        (root / "data/worldsense.labels.jsonl").write_text(
            "".join(json.dumps({"sample_id": str(i)}) + "\n" for i in range(518)))
        (root / "tasks.json").write_text(json.dumps([{"label": "base"}]))
        (root / "queue_state.json").write_text("{}")
        folder = root / "openqa/base/shards/card_0"
        folder.mkdir(parents=True)
        (folder / "input.jsonl").write_text('{"case_id":"one"}\n{"case_id":"two"}\n')
        started = datetime.now().astimezone().isoformat()
        unit = dict(status="running", worker="worker-0", assigned_device="0", kind="generation",
                    task_key="openqa/base", folder=str(folder), expected_rows=2,
                    started_at=started)
        (root / "card_pool_state.json").write_text(json.dumps({"units": {"openqa/base/card_0": unit}}))
        heartbeat = root / "card_pool_workers/worker-0/heartbeat.json"
        heartbeat.parent.mkdir(parents=True)
        heartbeat.write_text(json.dumps(dict(worker="worker-0", heartbeat_at=started, pid=123,
                                            devices={"0": "owned"})))
        return root, folder, unit

    def test_incomplete_append_is_progress_not_failure_and_alerts_clear(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, folder, unit = self.fixture(temporary)
            (folder / "results.jsonl").write_text('{"response":"ok"}\n{"response":')
            observation = watch.snapshot(root)
            self.assertTrue(observation["healthy"])
            self.assertEqual(observation["units"][0]["rows_written"], 1)
            self.assertTrue(observation["units"][0]["incomplete_final_write"])
            self.assertEqual(observation["progress"]["active_cards"], 1)
            watch.persist(root, observation)
            self.assertTrue((root / "card_pool_health.json").exists())
            self.assertFalse((root / "card_pool_alerts.jsonl").exists())
            (folder / "infer.log").write_text("Traceback (most recent call last):\nRuntimeError: synthetic\n")
            failed = watch.snapshot(root, observation)
            self.assertFalse(failed["healthy"])
            watch.persist(root, failed, observation)
            self.assertTrue((root / "card_pool_alerts.jsonl").exists())
            (folder / "infer.log").write_text("Traceback (most recent call last):\nRuntimeError: previous attempt\nCARD_POOL_START {}\nhealthy current attempt\n")
            repaired = watch.snapshot(root, failed)
            watch.persist(root, repaired, failed)
            last = json.loads((root / "card_pool_alerts.jsonl").read_text().splitlines()[-1])
            self.assertTrue(last["cleared_issue_ids"])

    def test_stale_worker_failed_unit_and_duplicate_card_are_reported(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, folder, unit = self.fixture(temporary)
            units = {"running": unit, "duplicate": dict(unit), "failed": dict(unit, status="failed", exit_code=1)}
            (root / "card_pool_state.json").write_text(json.dumps({"units": units}))
            heartbeat = root / "card_pool_workers/worker-0/heartbeat.json"
            heartbeat.write_text(json.dumps(dict(worker="worker-0", heartbeat_at=(
                datetime.now().astimezone() - timedelta(minutes=10)).isoformat(), pid=123)))
            observation = watch.snapshot(root)
            kinds = {item["kind"] for item in observation["issues"]}
            self.assertFalse(observation["healthy"])
            self.assertIn("worker_heartbeat_stale", kinds)
            self.assertIn("unit_failed", kinds)
            self.assertIn("duplicate_running_card_claim", kinds)

    def test_legacy_shard_progress_remains_visible_during_pool_handoff(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, folder, unit = self.fixture(temporary)
            (root / "card_pool_state.json").write_text(json.dumps({"units": {}}))
            queue = {"openqa/base": dict(status="running", worker="legacy-worker",
                                          started_at=datetime.now().astimezone().isoformat())}
            (root / "queue_state.json").write_text(json.dumps(queue))
            (folder / "results.jsonl").write_text('{"response":"one"}\n')
            observation = watch.snapshot(root)
            self.assertTrue(observation["healthy"])
            self.assertEqual(observation["legacy_jobs"][0]["rows_written"], 1)
            self.assertEqual(observation["progress"]["generation_rows_written"], 1)
            self.assertFalse(observation["all_26_tasks_and_summaries_complete"])

    def test_prepare_only_v2_summary_is_not_counted_as_final_grade(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, folder, unit = self.fixture(temporary)
            (root / "card_pool_state.json").write_text(json.dumps({"units": {}}))
            mc = root / "mcq/base"
            mc.mkdir(parents=True)
            (mc / "summary.json").write_text(json.dumps(dict(total=518, scoring="mcq-v2:test")))
            final = root / "openqa/base/summary.json"
            summary = dict(total=518, scoring="v2:test", uncertain=10,
                           scoring_pipeline_reliable_on_calibration=True)
            final.write_text(json.dumps(summary))
            state = {"mcq/base": {"status": "complete"}, "openqa/base": {"status": "generated"},
                     "score/base": {"status": "running"}}
            (root / "queue_state.json").write_text(json.dumps(state))
            prepared = watch.snapshot(root)
            self.assertEqual(prepared["progress"]["summary_files_complete"], {"mcq": 1, "openqa": 0})
            self.assertEqual(prepared["progress"]["unresolved_semantic_grades"], 0)
            self.assertFalse(prepared["all_26_tasks_and_summaries_complete"])
            state["openqa/base"]["status"] = state["score/base"]["status"] = "complete"
            (root / "queue_state.json").write_text(json.dumps(state))
            graded = watch.snapshot(root)
            self.assertEqual(graded["progress"]["summary_files_complete"], {"mcq": 1, "openqa": 1})
            self.assertEqual(graded["progress"]["unresolved_semantic_grades"], 10)
            self.assertTrue(graded["all_26_tasks_and_summaries_complete"])
            self.assertFalse(graded["all_semantic_grades_resolved"])
            summary["scoring_pipeline_reliable_on_calibration"] = False
            final.write_text(json.dumps(summary))
            unreliable = watch.snapshot(root)
            self.assertEqual(unreliable["progress"]["summary_files_complete"]["openqa"], 0)
            self.assertEqual(unreliable["progress"]["unresolved_semantic_grades"], 0)
            self.assertFalse(unreliable["all_26_tasks_and_summaries_complete"])


if __name__ == "__main__":
    unittest.main()
