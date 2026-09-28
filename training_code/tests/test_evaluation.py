import pytest

from omni_opsd.evaluation import compare_mcq_runs, summarize_gap_recovery, summarize_mcq_results


def test_summarize_mcq_results_is_strict_and_counts_unparsed_wrong():
    labels = [
        {"sample_id": "a", "answer": "A"},
        {"sample_id": "b", "answer": "B"},
        {"sample_id": "c", "answer": "C"},
    ]
    results = [
        {"case_id": "c", "response": "not an option"},
        {"case_id": "a", "response": "Option A"},
        {"case_id": "b", "response": "C"},
    ]

    summary = summarize_mcq_results(results, labels)

    assert summary["total"] == 3
    assert summary["parsed"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["prediction_counts"] == {"A": 1, "C": 1, "UNPARSED": 1}


def test_summarize_mcq_results_rejects_missing_or_duplicate_ids():
    labels = [{"sample_id": "a", "answer": "A"}]
    with pytest.raises(ValueError, match="mismatch"):
        summarize_mcq_results([], labels)
    with pytest.raises(ValueError, match="duplicate result"):
        summarize_mcq_results(
            [{"case_id": "a", "response": "A"}, {"case_id": "a", "response": "A"}],
            labels,
        )


def test_compare_mcq_runs_reports_paired_delta_against_base():
    labels = [
        {"sample_id": "a", "answer": "A"},
        {"sample_id": "b", "answer": "B"},
        {"sample_id": "c", "answer": "C"},
        {"sample_id": "d", "answer": "D"},
    ]
    report = compare_mcq_runs(
        {
            "base": [
                {"case_id": "a", "response": "A"},
                {"case_id": "b", "response": "A"},
                {"case_id": "c", "response": "A"},
                {"case_id": "d", "response": "A"},
            ],
            "sft": [
                {"case_id": "a", "response": "A"},
                {"case_id": "b", "response": "B"},
                {"case_id": "c", "response": "C"},
                {"case_id": "d", "response": "A"},
            ],
        },
        labels,
        resamples=100,
    )

    assert report["arms"]["base"]["accuracy"] == pytest.approx(0.25)
    assert report["arms"]["sft"]["accuracy"] == pytest.approx(0.75)
    assert report["paired_vs_reference"]["sft"]["delta_accuracy"] == pytest.approx(0.5)
    assert report["paired_vs_reference"]["sft"]["n_pairs"] == 4


def test_gap_recovery_uses_same_ids_and_returns_unavailable_for_nonpositive_gap():
    labels = [
        {"sample_id": "a", "answer": "A"},
        {"sample_id": "b", "answer": "B"},
        {"sample_id": "c", "answer": "C"},
        {"sample_id": "d", "answer": "D"},
    ]
    rows = lambda answers: [
        {"case_id": sample_id, "response": answer}
        for sample_id, answer in zip(("a", "b", "c", "d"), answers)
    ]
    report = summarize_gap_recovery(
        {
            "initial_student": rows(("A", "A", "A", "A")),
            "current_student": rows(("A", "B", "C", "A")),
            "teacher": rows(("A", "B", "C", "D")),
        },
        labels,
    )
    assert report["initial_student_accuracy"] == pytest.approx(0.25)
    assert report["current_student_accuracy"] == pytest.approx(0.75)
    assert report["teacher_accuracy"] == pytest.approx(1.0)
    assert report["gap_recovery_rate"] == pytest.approx(2 / 3)
    assert report["available"] is True

    unavailable = summarize_gap_recovery(
        {
            "initial_student": rows(("A", "A", "A", "A")),
            "current_student": rows(("A", "B", "C", "A")),
            "teacher": rows(("A", "A", "A", "A")),
        },
        labels,
    )
    assert unavailable["gap_recovery_rate"] is None
    assert unavailable["available"] is False


def test_summarize_maps_swift_results_back_to_source_inputs():
    sources = [
        {
            "case_id": "a",
            "messages": [{"role": "user", "content": "Question A"}],
            "videos": ["/cache/video.mp4"],
        },
        {
            "case_id": "b",
            "messages": [{"role": "user", "content": "Question B"}],
            "videos": ["/cache/video.mp4"],
        },
    ]
    results = [
        {
            "messages": [
                {"role": "user", "content": "Question B", "loss": None, "loss_scale": None},
                {"role": "assistant", "content": "B"},
            ],
            "videos": ["/cache/video.mp4"],
            "response": "B",
        },
        {
            "messages": [
                {"role": "user", "content": "Question A", "loss": None, "loss_scale": None},
                {"role": "assistant", "content": "A"},
            ],
            "videos": ["/cache/video.mp4"],
            "response": "A",
        },
    ]

    summary = summarize_mcq_results(
        results,
        [{"sample_id": "a", "answer": "A"}, {"sample_id": "b", "answer": "B"}],
        sources,
    )

    assert summary["total"] == 2
    assert summary["correct"] == 2
    assert summary["accuracy"] == 1.0


def test_source_signature_distinguishes_bounded_audio_views():
    labels = [{"sample_id": "a", "answer": "A"}, {"sample_id": "b", "answer": "B"}]
    sources = [
        {
            "case_id": sample_id,
            "messages": [{"role": "user", "content": "same"}],
            "videos": ["/cache/video.mp4"],
            "audios": [{"audio": "/cache/video.mp4", "audio_start": start, "audio_end": end}],
        }
        for sample_id, start, end in (("a", 0, 1), ("b", 1, 2))
    ]
    results = [
        {
            "messages": [
                {"role": "user", "content": "same"},
                {"role": "assistant", "content": answer},
            ],
            "videos": ["/cache/video.mp4"],
            "audios": source["audios"],
            "response": answer,
        }
        for source, answer in zip(sources, ("A", "B"))
    ]

    summary = summarize_mcq_results(results, labels, sources)
    assert summary["correct"] == 2
