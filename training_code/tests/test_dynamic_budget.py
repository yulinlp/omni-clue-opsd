from omni_opsd.data.dynamic_budget import dynamic_budget_for, dynamic_video_spec


def _row(duration=129.0, resolution="854x480"):
    return {
        "sample_id": "case-0",
        "video_id": "video-0",
        "video_path": "/videos/video-0.mp4",
        "duration": duration,
        "metadata": {"resolution": resolution},
    }


def test_full_video_budget_matches_documented_16_9_example():
    budget = dynamic_budget_for(_row())
    assert budget["nframes"] == 240
    assert budget["resized_height"] == 280
    assert budget["resized_width"] == 560
    assert budget["visual_budget_tokens"] == 24_000
    assert 100 <= budget["visual_tokens_per_sampled_frame"] <= 128
    assert budget["max_checked_tokens"] <= 32_768

    spec = dynamic_video_spec(_row(), budget)
    assert spec["nframes"] == 240
    assert "fps" not in spec
    assert spec["max_pixels"] == 280 * 560


def test_shorter_video_uses_fewer_uniform_frames_and_keeps_context_headroom():
    budget = dynamic_budget_for(_row(duration=60.0, resolution="720x1280"))
    assert budget["nframes"] == 120
    assert budget["visual_budget_tokens"] <= 24_000
    assert budget["visual_tokens_per_sampled_frame"] <= 128
    assert budget["max_checked_tokens"] <= 32_768
