from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

from scripts import materialize_local_training_matrix as materialize


def test_materialize_rewrites_only_structured_video_fields(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "persistent"
    output_root = tmp_path / "local_matrix"
    cache_root = tmp_path / "media"
    source_root.mkdir()
    cache_root.mkdir()
    persistent_video = tmp_path / "persistent_media" / "v1.mp4"
    persistent_video.parent.mkdir()
    persistent_video.write_bytes(b"video")
    (cache_root / "v1.mp4").write_bytes(b"video")
    row = {
        "messages": [{"role": "user", "content": f"literal {persistent_video}"}],
        "videos": [{"video": str(persistent_video), "video_start": 0.0}],
        "teacher_videos": [{"video": str(persistent_video), "video_start": 1.0}],
        "solution": "A",
    }
    expected = deepcopy(row)
    for arm in ("sft", "grpo", "opsd", "clue_opsd"):
        path = source_root / f"omnivideo_100k_train.{arm}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (source_root / "training_matrix_summary.json").write_text(
        json.dumps({"rows": 1}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_local_training_matrix.py",
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--video-cache-root",
            str(cache_root),
        ],
    )
    materialize.main()

    localized = json.loads(
        (output_root / "omnivideo_100k_train.clue_opsd.jsonl").read_text(encoding="utf-8")
    )
    assert localized["videos"][0]["video"] == str(cache_root / "v1.mp4")
    assert localized["teacher_videos"][0]["video"] == str(cache_root / "v1.mp4")
    assert localized["messages"] == expected["messages"]
    assert localized["solution"] == expected["solution"]
    report = json.loads((output_root / "LOCAL_MATRIX_SUCCESS.json").read_text())
    assert report["arms"]["clue_opsd"]["media_references"] == 2


def test_materialize_fails_when_cache_is_incomplete(tmp_path: Path) -> None:
    counters = {"media_references": 0, "unique_sources": set(), "unique_targets": set()}
    try:
        materialize._localize(
            {"video": "/persistent/missing.mp4"}, tmp_path, counters
        )
    except FileNotFoundError as exc:
        assert "absent from required local cache" in str(exc)
    else:
        raise AssertionError("incomplete local media cache was accepted")
