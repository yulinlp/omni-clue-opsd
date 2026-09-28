"""Batch runner with resume support and per-episode audit traces."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .clients import OmniClient
from .config import AgentConfig
from .episode import run_episode
from .media import MediaIndex
from .schema import QuestionRecord


def _load_done(output_path: Path) -> set[str]:
    """Collect finished question ids and truncate a torn trailing line.

    A native crash can interrupt a write mid-line; resuming must neither
    re-run completed questions nor leave a malformed row in the JSONL.
    """

    done: set[str] = set()
    if not output_path.is_file():
        return done
    text = output_path.read_text(encoding="utf-8", errors="replace")
    good_chars = 0
    for line in text.splitlines(keepends=True):
        if not line.strip():
            good_chars += len(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        question_id = row.get("question_id")
        if question_id:
            done.add(str(question_id))
        good_chars += len(line)
    if good_chars < len(text):
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(text[:good_chars])
    return done


def run_batch(
    records: Iterable[QuestionRecord],
    client: OmniClient | None,
    config: AgentConfig,
    media_index: MediaIndex,
    output_path: str | Path,
    trace_path: str | Path,
    *,
    resume: bool = True,
    client_factory: Callable[[QuestionRecord], OmniClient] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the loop for every record, appending results and traces.

    The runner never overwrites existing rows: with ``resume`` it skips any
    question already present in the output file, so a crashed run can restart.
    ``client_factory`` overrides ``client`` per record (used by mock runs).
    """

    if client is None and client_factory is None:
        raise ValueError("run_batch needs either a client or a client_factory")

    output = Path(output_path)
    trace = Path(trace_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(output) if resume else set()

    by_status: dict[str, int] = {}
    total = 0
    started = time.time()
    with output.open("a", encoding="utf-8") as output_handle, trace.open(
        "a", encoding="utf-8"
    ) as trace_handle:
        for index, record in enumerate(records, 1):
            if record.question_id in done:
                continue
            total += 1
            media_info = media_index.get(record.video_path)
            episode_client = client_factory(record) if client_factory is not None else client
            episode_started = time.time()
            result, messages, events = run_episode(record, episode_client, config, media_info)
            elapsed = time.time() - episode_started
            by_status[result.status] = by_status.get(result.status, 0) + 1

            row = result.as_dict()
            row["elapsed_s"] = round(elapsed, 2)
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()

            trace_row = {
                "question_id": record.question_id,
                "video_id": record.video_id,
                "task_type": record.task_type,
                "status": result.status,
                "elapsed_s": round(elapsed, 2),
                "config": config.as_dict(),
                "media_info": media_info.as_dict(),
                "events": events,
                "messages": messages,
                "result": row,
            }
            trace_handle.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
            trace_handle.flush()

            log(
                f"[{index}] {record.question_id} status={result.status} "
                f"turns={result.turns} inspects={result.inspect_calls} "
                f"intervals={row['clue_intervals']} elapsed={elapsed:.1f}s"
            )

    summary = {
        "processed": total,
        "skipped_resume": len(done),
        "by_status": by_status,
        "elapsed_s": round(time.time() - started, 1),
        "output": str(output),
        "trace": str(trace),
    }
    return summary
