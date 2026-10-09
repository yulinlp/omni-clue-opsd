"""CPU checks for duplicate claims, unknown occupancy and failed child exits."""
import json
import contextlib
import io
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from worker_worldsense_card_pool import (
    ClaimBlocked, atomic_save, claim_pending, complete_process, parse_npu_processes,
    read_state, unit_environment, update_owned,
)
import worker_worldsense_card_pool as worker_module


PROCESS_TABLE = """
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 0       0                | 4321          | python                   | 22000                   |
| No running processes found in NPU 1                                                            |
"""


def race_claim(root, worker, ready, begin, output):
    ready.put(worker)
    if not begin.wait(timeout=10):
        raise RuntimeError("test coordination timeout")
    claim = claim_pending(Path(root), worker, "1", "empty", lock_timeout=5)
    output.put(None if claim is None else claim[0])


class CardPoolWorkerTests(unittest.TestCase):
    def seed_root(self, root):
        unit = {"status": "pending", "command": [sys.executable, "-c", "raise SystemExit(7)"],
                "env_overrides": {}, "kind": "generation", "task_key": "mcq/test",
                "folder": str(root / "mcq/test/shards/card_0"), "priority": 0, "expected_rows": 65}
        atomic_save(root / "card_pool_state.json", {"units": {"only_unit": unit}})

    def fail_after_lock_exit(self, mutate=None):
        real_lock = worker_module.DirectoryLock
        class FailingExit:
            def __init__(self, path, **kwargs):
                self.path = path
                self.lock = real_lock(path, **kwargs)
            def __enter__(self):
                return self.lock.__enter__()
            def __exit__(self, *args):
                self.lock.__exit__(*args)
                if mutate:
                    mutate(self.path.parent)
                raise FileNotFoundError('injected owner.json disappearance on lock exit')
        return FailingExit

    def add_second_pending(self, root):
        state = read_state(root / 'card_pool_state.json')
        state['units']['second_unit'] = dict(state['units']['only_unit'], priority=1)
        atomic_save(root / 'card_pool_state.json', state)

    def test_committed_claim_survives_lock_exit_error_with_exact_nonce(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            self.seed_root(root)
            self.add_second_pending(root)
            with patch.object(worker_module, 'DirectoryLock', self.fail_after_lock_exit()), \
                    contextlib.redirect_stderr(io.StringIO()):
                key, unit = claim_pending(root, 'worker', '1', 'empty')
            saved = read_state(root / 'card_pool_state.json')['units'][key]
            self.assertEqual(saved['claim_id'], unit['claim_id'])
            self.assertEqual(saved['attempt'], 1)
            self.assertEqual(saved['allocation_stage'], 'claimed-not-started')
            audit = json.loads((root / 'card_pool_alerts.jsonl').read_text().splitlines()[0])
            self.assertTrue(audit['recovered_committed_transition'])
            self.assertIn('Traceback (most recent call last)', audit['traceback'])
            self.assertIn('owner.json disappearance', audit['traceback'])
            with self.assertRaises(ClaimBlocked):
                claim_pending(root, 'worker', '1', 'empty')
            self.assertEqual(read_state(root / 'card_pool_state.json')['units']['second_unit']['status'], 'pending')
            # Reserving this card does not unnecessarily block a different card.
            self.assertEqual(claim_pending(root, 'worker', '2', 'empty')[0], 'second_unit')

    def test_uncertain_commit_or_changed_owner_blocks_instead_of_adopting(self):
        for changed_field, changed_value in (('claim_id', 'another-nonce'), ('worker', 'another-worker'),
                                           ('assigned_device', '2'), ('pid', 999), ('status', 'pending')):
            with self.subTest(field=changed_field), tempfile.TemporaryDirectory(dir='/tmp') as temporary:
                root = Path(temporary)
                self.seed_root(root)
                self.add_second_pending(root)
                def mutate(directory):
                    state = read_state(directory / 'card_pool_state.json')
                    state['units']['only_unit'][changed_field] = changed_value
                    atomic_save(directory / 'card_pool_state.json', state)
                with patch.object(worker_module, 'DirectoryLock', self.fail_after_lock_exit(mutate)), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(ClaimBlocked):
                    claim_pending(root, 'worker', '1', 'empty')
                self.assertEqual(read_state(root / 'card_pool_state.json')['units']['second_unit']['status'], 'pending')
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            self.seed_root(root)
            with patch.object(worker_module, 'atomic_save', side_effect=OSError('save failed before commit')), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(ClaimBlocked):
                claim_pending(root, 'worker', '1', 'empty')
            self.assertEqual(read_state(root / 'card_pool_state.json')['units']['only_unit']['status'], 'pending')

    def test_existing_ghost_reservation_refuses_a_second_claim(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            self.seed_root(root)
            self.add_second_pending(root)
            state = read_state(root / 'card_pool_state.json')
            state['units']['only_unit'].update(status='running', worker='worker', assigned_device='3',
                                              claim_id='lost-return', worker_pid=999999,
                                              allocation_stage='claimed-not-started')
            atomic_save(root / 'card_pool_state.json', state)
            with self.assertRaises(ClaimBlocked):
                claim_pending(root, 'worker', '3', 'empty')
            self.assertEqual(read_state(root / 'card_pool_state.json')['units']['second_unit']['status'], 'pending')

    def test_main_keeps_an_unsafe_claim_slot_blocked_on_later_polls(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            self.seed_root(root)
            probe = dict(ok=True, observed_at=worker_module.now(),
                         devices={'0': dict(status='empty', processes=[])})
            argv = ['worker', '--root', str(root), '--worker', 'fixture-worker', '--devices', '0']
            with patch.object(sys, 'argv', argv), patch.object(worker_module.signal, 'signal'), \
                    patch.object(worker_module, 'probe_npus', return_value=probe), \
                    patch.object(worker_module, 'claim_pending', side_effect=ClaimBlocked('uncertain commit')) as claim, \
                    patch.object(worker_module.time, 'sleep', side_effect=[None, StopIteration]), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(StopIteration):
                worker_module.main()
            claim.assert_called_once()
            heartbeat = json.loads((root / 'card_pool_workers/fixture-worker/heartbeat.json').read_text())
            self.assertEqual(heartbeat['devices']['0']['status'], 'unknown_blocked')
            self.assertIn('uncertain commit', heartbeat['devices']['0']['reservation_error']['error'])

    def test_completed_transition_is_recoverable_and_retry_is_idempotent(self):
        for returncode in (0, 7):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory(dir='/tmp') as temporary:
                root = Path(temporary)
                self.seed_root(root)
                key, unit = claim_pending(root, 'worker', '3', 'empty')
                with patch.object(worker_module, 'DirectoryLock', self.fail_after_lock_exit()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    first = complete_process(root, key, unit['claim_id'], returncode)
                with patch.object(worker_module, 'atomic_save') as save:
                    second = complete_process(root, key, unit['claim_id'], returncode)
                save.assert_not_called()
                self.assertEqual(first, second)
                self.assertEqual(first['status'], 'complete' if returncode == 0 else 'failed')
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'ownership changed'):
                    complete_process(root, key, 'wrong-nonce', returncode)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'ownership changed'):
                    complete_process(root, key, unit['claim_id'], returncode + 1)
                self.assertEqual(read_state(root / 'card_pool_state.json')['units'][key]['exit_code'], returncode)

    def test_only_explicit_empty_process_table_card_is_eligible(self):
        report = parse_npu_processes(PROCESS_TABLE, ["0", "1", "2"])
        self.assertEqual([report[device]["status"] for device in ("0", "1", "2")], ["busy", "empty", "unknown"])
        self.assertEqual(report["0"]["processes"][0]["pid"], 4321)
        # Telemetry and partial text cannot stand in for the process table.
        no_header = parse_npu_processes("NPU 0 0% 0MB\nNo running processes found in NPU 0", ["0"])
        self.assertEqual(no_header["0"]["status"], "unknown")
        contradiction = parse_npu_processes(PROCESS_TABLE + "| No running processes found in NPU 0 |\n", ["0"])
        self.assertEqual(contradiction["0"]["status"], "unknown")
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self.seed_root(root)
            for unsafe in ("busy", "unknown"):
                self.assertIsNone(claim_pending(root, "worker", "0", unsafe))
            self.assertEqual(read_state(root / "card_pool_state.json")["units"]["only_unit"]["status"], "pending")

    def test_simultaneous_workers_cannot_claim_one_unit_twice(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self.seed_root(root)
            context = multiprocessing.get_context("spawn")
            ready, output = context.Queue(), context.Queue()
            begin = context.Event()
            children = [context.Process(target=race_claim, args=(temporary, f"worker-{i}", ready, begin, output)) for i in range(2)]
            for child in children:
                child.start()
            for _ in children:
                ready.get(timeout=10)
            begin.set()
            for child in children:
                child.join(timeout=10)
                self.assertFalse(child.is_alive(), "Concurrent claim deadlocked")
                self.assertEqual(child.exitcode, 0)
            decisions = [output.get(timeout=2) for _ in children]
            self.assertEqual(decisions.count("only_unit"), 1)
            self.assertEqual(decisions.count(None), 1)
            state = read_state(root / "card_pool_state.json")
            self.assertEqual(state["units"]["only_unit"]["attempt"], 1)
            self.assertEqual(state["units"]["only_unit"]["status"], "running")

    def test_failure_is_persisted_without_retry_and_claim_token_protects_owner(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self.seed_root(root)
            key, unit = claim_pending(root, "worker", "3", "empty")
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                update_owned(root, key, "other-claim", {"status": "complete"})
            import subprocess
            child = subprocess.run(unit["command"], capture_output=True)
            self.assertEqual(child.returncode, 7)
            complete_process(root, key, unit["claim_id"], child.returncode)
            saved = read_state(root / "card_pool_state.json")["units"][key]
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["exit_code"], 7)
            self.assertIsNone(claim_pending(root, "worker", "3", "empty"))
            self.assertEqual(saved["command"], unit["command"])

    def test_single_card_and_distinct_card_port_override_untrusted_allocation_env(self):
        overrides = {"ASCEND_RT_VISIBLE_DEVICES": "0,1", "MASTER_PORT": "12345", "WORLD_SIZE": "8",
                     "RANK_TABLE_FILE": "/tmp/unused.json", "USE_AUDIO_IN_VIDEO": "0", "FROZEN_SEED": "20260904"}
        first, second = unit_environment("3", overrides), unit_environment("4", overrides)
        self.assertEqual(first["ASCEND_RT_VISIBLE_DEVICES"], "3")
        self.assertEqual(first["MASTER_PORT"], "30173")
        self.assertEqual(second["MASTER_PORT"], "30174")
        self.assertNotIn("WORLD_SIZE", first)
        self.assertNotIn("RANK_TABLE_FILE", first)
        self.assertEqual(first["NPROC_PER_NODE"], "1")
        self.assertEqual(first["FROZEN_SEED"], "20260904")
        self.assertEqual(first["USE_AUDIO_IN_VIDEO"], "0")


if __name__ == "__main__":
    unittest.main()
