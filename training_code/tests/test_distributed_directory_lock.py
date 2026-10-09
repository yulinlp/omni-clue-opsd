"""CPU regressions for cross-process exclusion and conservative lock recovery."""
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from distributed_directory_lock import DirectoryLock, DirectoryLockTimeout


def increment_locked(root, loops, events):
    root = Path(root)
    try:
        for _ in range(loops):
            with DirectoryLock(root / 'lock.d', timeout=10):
                count = int((root / 'counter').read_text())
                # Deliberately widen the lost-update window when exclusion breaks.
                time.sleep(0.003)
                (root / 'counter').write_text(str(count + 1))
        events.put(None)
    except BaseException as error:
        events.put(repr(error))


class DirectoryLockTests(unittest.TestCase):
    def test_four_processes_preserve_every_shared_update(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            (root / 'counter').write_text('0')
            context = multiprocessing.get_context('spawn')
            events = context.Queue()
            children = [context.Process(target=increment_locked, args=(root, 20, events)) for _ in range(4)]
            for child in children:
                child.start()
            for child in children:
                child.join(timeout=20)
                self.assertFalse(child.is_alive(), 'Lock acquisition deadlocked')
                self.assertEqual(child.exitcode, 0)
            self.assertEqual([events.get(timeout=2) for _ in children], [None] * 4)
            self.assertEqual(int((root / 'counter').read_text()), 80)
            self.assertFalse((root / 'lock.d').exists())

    def test_exception_releases_owner_and_next_process_can_acquire(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            lockpath = Path(temporary) / 'lock.d'
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                with DirectoryLock(lockpath):
                    raise ValueError('synthetic critical-section failure')
            with DirectoryLock(lockpath, timeout=0):
                self.assertTrue((lockpath / 'owner.json').is_file())
            self.assertFalse(lockpath.exists())

    def test_dead_recorded_owner_is_never_stolen(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            lockpath = Path(temporary) / 'lock.d'
            lockpath.mkdir()
            recorded = json.dumps({'pid': 999999999, 'hostname': 'other-host', 'token': 'abandoned'})
            (lockpath / 'owner.json').write_text(recorded)
            with self.assertRaisesRegex(DirectoryLockTimeout, 'other-host'):
                DirectoryLock(lockpath, timeout=0.02).acquire()
            self.assertEqual((lockpath / 'owner.json').read_text(), recorded)

    def test_owner_change_prevents_unrelated_release(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            lock = DirectoryLock(Path(temporary) / 'lock.d').acquire()
            (lock.path / 'owner.json').write_text(json.dumps({'token': 'other-owner'}))
            with self.assertRaisesRegex(RuntimeError, 'another owner'):
                lock.release()
            self.assertTrue(lock.path.is_dir())


if __name__ == '__main__':
    unittest.main()
