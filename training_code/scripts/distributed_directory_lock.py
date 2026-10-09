"""Cross-host shared-filesystem exclusion using atomic directory creation.

Locks are never stolen automatically, even when a recorded PID looks dead on this
host: that PID may belong to another host. Recover an abandoned lock only after
an operator has checked its owner. Keep critical sections short.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import time
import uuid


class DirectoryLockTimeout(TimeoutError):
    pass


class DirectoryLock:
    def __init__(self, path, timeout=30.0, poll_interval=0.05):
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.token = None
        self.owner = None

    def acquire(self):
        if self.token is not None:
            raise RuntimeError('DirectoryLock is not reentrant')
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.path.mkdir(mode=0o700)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    try:
                        recorded = (self.path / 'owner.json').read_text()
                    except OSError:
                        recorded = 'owner metadata unavailable (possibly being created)'
                    raise DirectoryLockTimeout(f'Lock occupied: {self.path}; owner={recorded}')
                time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
        token = uuid.uuid4().hex
        owner = dict(token=token, pid=os.getpid(), hostname=socket.gethostname(),
                     uid=os.getuid(), created_at=datetime.now().astimezone().isoformat())
        try:
            with (self.path / 'owner.json').open('x') as stream:
                stream.write(json.dumps(owner, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # This acquisition created the directory; no other claimant can own it.
            try:
                (self.path / 'owner.json').unlink(missing_ok=True)
                self.path.rmdir()
            except OSError:
                pass
            raise
        self.token, self.owner = token, owner
        return self

    def release(self):
        if self.token is None:
            raise RuntimeError('DirectoryLock is not held')
        recorded = json.loads((self.path / 'owner.json').read_text())
        if recorded.get('token') != self.token:
            raise RuntimeError(f'Refusing to release another owner\'s lock: {self.path}')
        (self.path / 'owner.json').unlink()
        self.path.rmdir()
        self.token = self.owner = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback):
        self.release()
