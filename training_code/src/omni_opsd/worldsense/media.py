"""Media probing and the per-video metadata index."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    fps: float
    n_frames: int
    has_audio: bool
    width: int = 0
    height: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _ffprobe_path() -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    for candidate in (
        "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffprobe",
        "/usr/bin/ffprobe",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _probe_audio(path: Path) -> bool:
    probe = _ffprobe_path()
    if probe is None:
        return True
    try:
        out = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return True
    return "audio" in out.stdout


def probe_video(path: str | Path) -> MediaInfo:
    """Probe duration/frame metadata with decord, falling back to ffprobe."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"video not found: {resolved}")
    import decord

    reader = decord.VideoReader(str(resolved), num_threads=1)
    n_frames = len(reader)
    fps = float(reader.get_avg_fps()) or 30.0
    duration = n_frames / fps if fps > 0 else 0.0
    width = height = 0
    try:
        frame = reader[0].asnumpy()
        height, width = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        pass
    return MediaInfo(
        duration_s=float(duration),
        fps=float(fps),
        n_frames=int(n_frames),
        has_audio=_probe_audio(resolved),
        width=width,
        height=height,
    )


class MediaIndex:
    """Probe cache keyed by resolved video path."""

    def __init__(self, entries: dict[str, MediaInfo] | None = None):
        self.entries: dict[str, MediaInfo] = dict(entries or {})

    def get(self, video_path: str | Path) -> MediaInfo:
        key = str(Path(video_path).resolve())
        if key not in self.entries:
            self.entries[key] = probe_video(key)
        return self.entries[key]

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value.as_dict() for key, value in self.entries.items()}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MediaIndex":
        source = Path(path)
        if not source.is_file():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls({key: MediaInfo(**value) for key, value in payload.items()})
