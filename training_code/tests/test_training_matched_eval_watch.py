"""CPU monitor regressions for prefill, file appends and confirmed failures."""
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/watch_worldsense_training_matched_eval.py"
spec = importlib.util.spec_from_file_location("matched_watch", SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class MonitorTests(unittest.TestCase):
    def fixture(self, temporary):
        root = Path(temporary)
        (root / "logs").mkdir()
        (root / "data").mkdir()
        (root / "data/worldsense.labels.jsonl").write_text("".join(json.dumps({"sample_id": str(i)}) + "\n" for i in range(518)))
        (root / "tasks.json").write_text(json.dumps([{"label": "base"}]))
        started = (datetime.now().astimezone() - timedelta(minutes=5)).isoformat()
        launch = {"worker": "synthetic-worker", "pid": 123456, "launched_at": started,
                  "command": ["python", str(SCRIPT.parent / "queue_worldsense_training_matched_eval.py"),
                              "--root", str(root), "--worker", "synthetic-worker"]}
        (root / "logs/synthetic-worker.launch.json").write_text(json.dumps(launch))
        (root / "logs/synthetic-worker.pid").write_text("123456")
        (root / "logs/synthetic-worker.lease.lock").touch()
        (root / "queue_state.json").write_text(json.dumps({"mcq/base": {"status": "running", "worker": "synthetic-worker",
                                                                      "pid": 123456, "devices": "0", "started_at": started}}))
        shard = root / "mcq/base/shards/card_0"
        shard.mkdir(parents=True)
        (shard / "input.jsonl").write_text("".join(json.dumps({"case_id": str(i)}) + "\n" for i in range(518)))
        return root, shard

    def test_prefill_warning_and_remote_free_lease_are_not_failures(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, shard = self.fixture(temporary)
            (shard / "infer.log").write_text("FutureWarning: librosa is deprecated\nUserWarning: compiler owner differs\n")
            result = watch.snapshot(root)
            self.assertTrue(result["healthy"])
            self.assertEqual(result["issues"], [])
            self.assertEqual(result["jobs"][0]["phase"], "prefill_or_loading")
            self.assertEqual(result["workers"]["synthetic-worker"]["lease_status"], "free")

    def test_incomplete_final_append_is_tolerated(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, shard = self.fixture(temporary)
            (shard / "results.jsonl").write_text('{"response":"A"}\n{"response":')
            result = watch.snapshot(root)
            self.assertEqual(result["progress"]["generation_rows_written"], 1)
            self.assertTrue(result["healthy"])
            self.assertTrue(result["jobs"][0]["cards"][0]["incomplete_final_write"])

    def test_failed_state_and_nonzero_exit_create_repair_alerts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, shard = self.fixture(temporary)
            state = json.loads((root / "queue_state.json").read_text())
            state["mcq/base"].update(status="failed", exit_code=1)
            (root / "queue_state.json").write_text(json.dumps(state))
            (root / "logs/synthetic-worker.exit").write_text("1\n")
            result = watch.snapshot(root)
            self.assertFalse(result["healthy"])
            kinds = {item["kind"] for item in result["issues"]}
            self.assertIn("task_failed", kinds)
            self.assertIn("worker_nonzero_exit", kinds)
            watch.persist(root, result, None)
            self.assertTrue((root / "alerts.jsonl").exists())
            self.assertTrue(json.loads((root / "repair_needed.json").read_text())["items"])

    def test_current_fatal_log_is_detected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root, shard = self.fixture(temporary)
            (shard / "infer.log").write_text("Traceback (most recent call last):\nRuntimeError: synthetic failure\n")
            result = watch.snapshot(root)
            self.assertFalse(result["healthy"])
            self.assertIn("current_shard_fatal_log", {item["kind"] for item in result["issues"]})


if __name__ == "__main__":
    unittest.main()
