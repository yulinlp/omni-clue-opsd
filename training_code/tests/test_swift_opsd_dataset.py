from omni_opsd.data.swift_opsd import swift_opsd_row, swift_training_matrix_rows, video_split


def _row(sample_id, video_id, answer="B"):
    return {
        "sample_id": sample_id,
        "video_id": video_id,
        "video_path": f"/videos/{video_id}.mp4",
        "question": "What happened?",
        "choices": ["first", "second"],
        "answer": answer,
        "duration": 100.0,
        "evidence_spans": [[10.0, 20.0], [80.0, 90.0]],
    }


def test_video_split_keeps_questions_from_same_video_together():
    rows = [_row("a1", "a"), _row("a2", "a"), _row("b1", "b"), _row("c1", "c")]
    train, val = video_split(rows, val_video_count=1, seed=7)
    assert {row["video_id"] for row in train}.isdisjoint({row["video_id"] for row in val})
    assert len(train) + len(val) == len(rows)


def test_swift_row_separates_student_and_teacher_media_without_answer():
    row = swift_opsd_row(_row("a1", "a"), max_frames=255)
    assert row["videos"][0]["video_start"] == 0.0
    assert row["videos"][0]["video_end"] == 100.0
    assert [item["video_start"] for item in row["teacher_videos"]] == [10.0, 80.0]
    assert sum(item["max_frames"] for item in row["teacher_videos"]) == 255
    assert row["teacher_prompt"].count("<video>") == 2
    assert "answer" not in row
    assert "Answer: B" not in row["messages"][0]["content"]
    assert row["sampling_contract"]["use_audio_in_video"] is False


def test_swift_row_requires_explicit_opt_in_for_long_audio():
    row = swift_opsd_row(_row("a1", "a"), max_frames=64, use_audio_in_video=True)
    assert row["sampling_contract"]["max_frames_per_view"] == 64
    assert row["sampling_contract"]["use_audio_in_video"] is True


def test_swift_row_expands_point_timestamp_to_a_teacher_clip():
    source = _row("point", "p")
    source["evidence_spans"] = []
    source["metadata"] = {"time_reference": "00:00:54-00:00:54"}
    row = swift_opsd_row(source)
    assert row["clue_intervals"] == [[53.0, 55.0]]


def test_training_matrix_changes_only_the_supervision_channel():
    canonical = _row("a1", "a", answer="B")
    materialized = swift_opsd_row(canonical)
    materialized["videos"] = ["/cache/full.mp4"]
    materialized["teacher_videos"] = ["/cache/clue-1.mp4", "/cache/clue-2.mp4"]
    rows = swift_training_matrix_rows(canonical, materialized)

    assert set(rows) == {"sft", "grpo", "opsd", "clue_opsd"}
    assert {tuple(row["videos"]) for row in rows.values()} == {("/cache/full.mp4",)}
    assert rows["sft"]["messages"][-1] == {"role": "assistant", "content": "B"}
    assert rows["grpo"]["solution"] == "B"
    assert "solution" not in rows["opsd"]
    assert "correct option is B" in rows["opsd"]["teacher_prompt"]
    assert rows["opsd"]["teacher_videos"] == rows["opsd"]["videos"]
    assert rows["clue_opsd"]["teacher_videos"] == [
        "/cache/clue-1.mp4",
        "/cache/clue-2.mp4",
    ]
    assert "answer" not in rows["clue_opsd"]
    assert "solution" not in rows["clue_opsd"]


def test_training_matrix_rejects_misaligned_case_ids():
    canonical = _row("a1", "a", answer="B")
    materialized = swift_opsd_row(canonical)
    materialized["case_id"] = "other"
    try:
        swift_training_matrix_rows(canonical, materialized)
    except ValueError as exc:
        assert "case mismatch" in str(exc)
    else:
        raise AssertionError("mismatched cases should be rejected")
