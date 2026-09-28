#!/usr/bin/env python3
"""Optional legacy frame-list materializer; NOT the full-video F1 training path.

Requires ffmpeg/ffprobe. Output is an engineering pilot and must not replace
full-video inputs in the four-arm comparison without redefining its protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def _frame_count(spec, configured, respect_spec_cap=False):
    if configured < 2:
        raise ValueError('configured frame count must be >= 2')
    if not respect_spec_cap:
        return configured
    cap = spec.get('max_frames')
    if not isinstance(cap, int) or cap < 2:
        raise ValueError('invalid max_frames')
    return min(configured, cap)


def _duration(spec):
    duration = float(spec['video_end']) - float(spec.get('video_start', 0))
    if duration <= 0:
        raise ValueError('invalid video interval')
    return duration


def _allocate_audio_seconds(specs, total):
    if total <= 0:
        raise ValueError('audio budget must be positive')
    durations = [_duration(s) for s in specs]
    factor = min(1.0, total / sum(durations)) if durations else 0
    return [d * factor for d in durations]


def _encode(frames, spec, cache_dir, ffmpeg, audio_bitrate, audio_seconds=None):
    duration = _duration(spec)
    if not frames:
        raise ValueError('empty frame list')
    start = float(spec.get('video_start', 0))
    seconds = duration if audio_seconds is None else min(float(audio_seconds), duration)
    if seconds <= 0:
        raise ValueError('audio budget must be positive')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([frames, spec, audio_bitrate, seconds], sort_keys=True).encode()).hexdigest()
    output = cache_dir / f'{key}.mp4'
    if output.is_file() and output.stat().st_size:
        return str(output)
    with tempfile.TemporaryDirectory(dir=cache_dir) as temp:
        temp = Path(temp)
        for i, frame in enumerate(frames):
            shutil.copyfile(frame, temp / f'{i:06d}.jpg')
        video, audio, mux = [temp / name for name in ['video.mp4', 'audio.m4a', 'mux.mp4']]
        subprocess.run([ffmpeg, '-y', '-v', 'error', '-framerate', f'{len(frames)/seconds:.12f}', '-i', str(temp/'%06d.jpg'), '-frames:v', str(len(frames)), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
        command = [ffmpeg, '-y', '-v', 'error']
        if audio_seconds is None or seconds == duration:
            command += ['-ss', f'{start:.6f}', '-i', str(spec['video']), '-t', f'{seconds:.6f}']
        else:
            count = len(frames)
            window = seconds/count
            graph = f'[0:a]asplit={count}' + ''.join(f'[a{i}]' for i in range(count)) + ';'
            for i in range(count):
                offset = (i+0.5)*duration/count - window/2
                graph += f'[a{i}]atrim=start={offset:.6f}:duration={window:.6f},asetpts=PTS-STARTPTS[s{i}];'
            graph += ''.join(f'[s{i}]' for i in range(count)) + f'concat=n={count}:v=0:a=1[outa]'
            command += ['-ss', f'{start:.6f}', '-i', str(spec['video']), '-filter_complex', graph, '-map', '[outa]']
        subprocess.run(command + ['-vn', '-ar', '16000', '-ac', '1', '-c:a', 'aac', '-b:a', audio_bitrate, str(audio)], check=True)
        subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(video), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', str(mux)], check=True)
        mux.replace(output)
    return str(output)


def _encode_av_timeline(frames, *, spec, cache_dir, ffmpeg, audio_bitrate):
    return _encode(frames, spec, cache_dir, ffmpeg, audio_bitrate)


def _encode_av_uniform_windows(frames, *, spec, cache_dir, ffmpeg, audio_bitrate, audio_seconds):
    return _encode(frames, spec, cache_dir, ffmpeg, audio_bitrate, audio_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--respect-spec-cap', action='store_true')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if any(r.get('sampling_contract', {}).get('use_audio_in_video') for r in rows):
        raise ValueError('frame-list CLI is visual-only; use original full-video matrix for A/V training')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for field in ['videos', 'teacher_videos']:
            converted = []
            for spec in row.get(field, []):
                count = _frame_count(spec, args.frames, args.respect_spec_cap)
                duration = _duration(spec)
                key = hashlib.sha256(json.dumps([spec, count], sort_keys=True).encode()).hexdigest()
                frames = []
                for i in range(count):
                    path = args.cache_dir / f'{key}.{i:06d}.jpg'
                    timestamp = float(spec.get('video_start', 0)) + (i+0.5)*duration/count
                    if not path.is_file():
                        subprocess.run([args.ffmpeg, '-y', '-v', 'error', '-ss', str(timestamp), '-i', spec['video'], '-frames:v', '1', str(path)], check=True)
                    if not path.is_file() or not path.stat().st_size:
                        raise ValueError(f'frame extraction failed: {path}')
                    frames.append(str(path.resolve()))
                converted.append(frames)
            if field in row:
                row[field] = converted
        row.setdefault('sampling_contract', {}).update(frames_per_video_input=args.frames, engineering_pilot=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


if __name__ == '__main__':
    main()
