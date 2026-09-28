from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_opsd_frame_lists.py"


def _module():
    spec = spec_from_file_location("materialize_opsd_frame_lists", SCRIPT)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frame_count_can_preserve_allocated_teacher_cap():
    frame_count = _module()._frame_count
    assert frame_count({"max_frames": 17}, configured=64, respect_spec_cap=True) == 17
    assert frame_count({"max_frames": 96}, configured=64, respect_spec_cap=True) == 64


def test_frame_count_fixed_mode_is_backwards_compatible():
    assert _module()._frame_count({}, configured=4, respect_spec_cap=False) == 4


def test_frame_count_rejects_missing_cap_in_contract_mode():
    with pytest.raises(ValueError, match="invalid max_frames"):
        _module()._frame_count({}, configured=64, respect_spec_cap=True)


def test_audio_budget_is_shared_across_teacher_intervals():
    allocate = _module()._allocate_audio_seconds
    specs = [
        {"video_start": 0.0, "video_end": 100.0},
        {"video_start": 200.0, "video_end": 400.0},
    ]

    assert allocate(specs, 150.0) == [50.0, 100.0]
    assert allocate(specs, 600.0) == [100.0, 200.0]


def test_av_timeline_encoder_keeps_duration_and_audio_mapping(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    frames = []
    for index in range(2):
        frame = tmp_path / f"{index}.jpg"
        frame.write_bytes(b"frame")
        frames.append(str(frame))

    commands = []

    def fake_run(command, check):
        assert check is True
        commands.append(command)
        Path(command[-1]).write_bytes(b"proxy")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = module._encode_av_timeline(
        frames,
        spec={"video": str(source), "video_start": 10.0, "video_end": 74.0},
        cache_dir=tmp_path / "cache",
        ffmpeg="ffmpeg",
        audio_bitrate="64k",
    )

    assert len(commands) == 3
    video_command, audio_command, mux_command = commands
    assert Path(output).is_file()
    assert video_command[video_command.index("-framerate") + 1] == "0.031250000000"
    assert video_command[video_command.index("-frames:v") + 1] == "2"
    assert "-an" in video_command
    assert audio_command[audio_command.index("-t") + 1] == "64.000000"
    assert audio_command[audio_command.index("-ar") + 1] == "16000"
    assert mux_command[mux_command.index("-map", mux_command.index("-map") + 1) + 1] == "1:a:0"
    assert mux_command[mux_command.index("-c") + 1] == "copy"


def test_uniform_audio_encoder_covers_full_interval_with_bounded_windows(
    tmp_path, monkeypatch
):
    module = _module()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    frames = []
    for index in range(2):
        frame = tmp_path / f"{index}.jpg"
        frame.write_bytes(b"frame")
        frames.append(str(frame))

    commands = []

    def fake_run(command, check):
        assert check is True
        commands.append(command)
        Path(command[-1]).write_bytes(b"proxy")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = module._encode_av_uniform_windows(
        frames,
        spec={"video": str(source), "video_start": 10.0, "video_end": 650.0},
        cache_dir=tmp_path / "cache",
        ffmpeg="ffmpeg",
        audio_bitrate="64k",
        audio_seconds=64.0,
    )

    assert len(commands) == 3
    video_command, audio_command, mux_command = commands
    assert Path(output).is_file()
    assert video_command[video_command.index("-framerate") + 1] == "0.031250000000"
    filter_graph = audio_command[audio_command.index("-filter_complex") + 1]
    assert "asplit=2[a0][a1]" in filter_graph
    assert "atrim=start=144.000000:duration=32.000000" in filter_graph
    assert "atrim=start=464.000000:duration=32.000000" in filter_graph
    assert "[s0][s1]concat=n=2:v=0:a=1[outa]" in filter_graph
    assert mux_command[mux_command.index("-c") + 1] == "copy"
