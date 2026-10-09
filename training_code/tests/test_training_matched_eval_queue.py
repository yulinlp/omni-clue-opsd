"""CPU ownership checks for preserved runners and idle queue behavior."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import queue_worldsense_training_matched_eval as queue


class QueueTests(unittest.TestCase):
    def make_runner(self, root):
        worker = 'test-worker'
        task = dict(label='base', model='/synthetic/model', adapter='-')
        entry = dict(worker=worker, devices='0', pid=999999999, status='running')
        output = root / 'mcq/base'
        output.mkdir(parents=True)
        runner = output / f'runner_mcq_{worker}_{entry["pid"]}.py'
        runner.write_text('import time\ntime.sleep(30)\n')
        command = queue.runner_command(root, 'mcq', task, '0', runner)
        process = subprocess.Popen([sys.executable] + command[1:])
        (output / 'mcq.runner.pid').write_text(str(process.pid))
        deadline = time.monotonic() + 5
        while queue.process_identity(process.pid)['cmdline'][1:] != command[1:]:
            if time.monotonic() >= deadline:
                raise AssertionError('Synthetic runner did not launch')
            time.sleep(0.01)
        return task, entry, process

    def test_adopt_requires_exact_local_uid_worker_devices_and_command(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            task, entry, process = self.make_runner(root)
            try:
                adoption = queue.validate_adoption(root, 'mcq', task, entry, 'test-worker', '0')
                self.assertTrue(adoption['runner_live_verified'])
                self.assertEqual(adoption['runner_pid'], process.pid)
                with self.assertRaisesRegex(RuntimeError, 'worker/devices'):
                    queue.validate_adoption(root, 'mcq', task, entry, 'test-worker', '1')
                wrong = dict(task, model='/unrelated/model')
                with self.assertRaisesRegex(RuntimeError, 'command/UID mismatch'):
                    queue.validate_adoption(root, 'mcq', wrong, entry, 'test-worker', '0')
                self.assertIsNone(process.poll(), 'Adoption must never stop a preserved runner')
            finally:
                process.terminate()
                process.wait(timeout=5)

    def test_zombie_is_treated_as_exited_without_wait(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            identity = dict(pid=123, uid=os.getuid(), state='Z', start_ticks='22', cmdline=[])
            adoption = dict(runner_pid=123, runner_start_ticks='22', runner_live_verified=True, runner_cmdline=[])
            with patch.object(queue, 'process_identity', return_value=identity), patch.object(queue.time, 'sleep') as sleep:
                queue.wait_adopted(Path(temporary), 'mcq/base', adoption)
                sleep.assert_not_called()

    def test_idle_poll_does_not_write_queue_state(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            (root / 'logs').mkdir()
            task = dict(label='base', model='/synthetic/model', adapter='-')
            (root / 'tasks.json').write_text(json.dumps([task]))
            state = {'mcq/base': {'status': 'complete'}, 'openqa/base': {'status': 'generated'}}
            statepath = root / 'queue_state.json'
            statepath.write_text(json.dumps(state))
            original = statepath.read_bytes()
            args = argparse.Namespace(root=root, worker='judge-worker', devices='0', role='judge',
                                      adopt_running=False, lock_timeout=2, first_mode=None)
            with patch.object(queue.time, 'sleep', side_effect=StopIteration), patch.object(queue, 'save') as save:
                with self.assertRaises(StopIteration):
                    queue.run_queue(args)
                save.assert_not_called()
            self.assertEqual(statepath.read_bytes(), original)

    def test_first_openqa_claim_then_normal_mcq_priority(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            (root / 'logs').mkdir()
            task = dict(label='base', model='/synthetic/model', adapter='-')
            (root / 'tasks.json').write_text(json.dumps([task]))
            args = argparse.Namespace(root=root, worker='generate-worker', devices='0', role='generate',
                                      adopt_running=False, lock_timeout=2, first_mode='openqa')
            seen = []
            def finish(a, mode, task, key, claim_id, lock, statepath):
                seen.append(mode)
                with lock:
                    state = queue.read_state(statepath)
                    state[key]['status'] = 'generated' if mode == 'openqa' else 'complete'
                    queue.save(statepath, state)
                return 0 if len(seen) == 1 else 1
            with patch.object(queue, 'execute', side_effect=finish):
                queue.run_queue(args)
            self.assertEqual(seen, ['openqa', 'mcq'])
            state = queue.read_state(root / 'queue_state.json')
            self.assertEqual(state['openqa/base']['claimed_mode'], 'openqa')


if __name__ == '__main__':
    unittest.main()
