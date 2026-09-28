"""Persistent store for question-conditioned timestamped video captions.

The caption prompt embeds the question and asks the model to describe
question-relevant moments in extra detail, so captions are per question: the
key is ``question_id | prompt version | fps | max_pixels``.  The store still
pays off on resume/retry (a crashed or restarted run never regenerates a
caption it already has) and lets a finished caption set be inspected or reused
by other stages.
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any

PROMPT_VERSION = "caption_v3"


def caption_key(question_id: str, *, fps: float, max_pixels: int, version: str = PROMPT_VERSION) -> str:
    """Captions are question-conditioned, so the key is the question id.

    The caption prompt includes the question and options and asks for extra
    detail around question-relevant moments, therefore two questions about the
    same video need their own caption.
    """

    return f"{question_id}|{version}|fps{fps:g}|px{int(max_pixels)}"


class CaptionStore:
    """Append-only JSONL store with a small in-memory index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._index: dict[str, dict[str, Any]] | None = None
        self._stamp: tuple[int, int] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            with self.path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = str(row.get("key") or "")
                    if key:
                        index[key] = row
        self._stamp = self._file_stamp()
        return index

    def _file_stamp(self) -> tuple[int, int] | None:
        """(mtime_ns, size): size catches appends even on coarse-mtime NFS."""

        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _reload_if_changed(self) -> None:
        """Pick up captions written by other processes (parallel shards).

        A stale in-memory index made concurrent workers regenerate captions
        that another worker had already stored, which doubled caption cost.
        """

        stamp = self._file_stamp()
        if self._index is None or stamp != self._stamp:
            self._index = self._load()

    def get(self, key: str) -> dict[str, Any] | None:
        self._reload_if_changed()
        return self._index.get(key)  # type: ignore[union-attr]

    def put(self, key: str, payload: dict[str, Any]) -> None:
        row = {"key": key, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        if self._index is None:
            self._index = {}
        self._index[key] = row

    def __len__(self) -> int:
        if self._index is None:
            self._index = self._load()
        return len(self._index)
