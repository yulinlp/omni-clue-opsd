#!/usr/bin/env python3
"""CPU smoke: verify actual decoded frames and nonempty audio, without a model."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    os.environ['FORCE_QWENVL_VIDEO_READER'] = 'pyav_seek'
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from qwen_omni_utils import fetch_video, process_audio_info
    rows = [json.loads(x) for x in (a.root/'media_smoke_inputs.jsonl').open()]
    outputs = []
    for row in rows:
        v = row['videos'][0]
        frames, fps = fetch_video(v, return_video_sample_fps=True)
        shape = list(frames.shape)
        expected = [v['nframes'], 3, v['resized_height'], v['resized_width']]
        if shape != expected:
            raise RuntimeError(f"Frame shape mismatch: {row['case_id']}: {shape} != {expected}")
        audio = process_audio_info([{'role':'user', 'content':[{'type':'video', **v}]}],
                                   use_audio_in_video=True)
        if audio is None or len(audio) != 1 or len(audio[0]) == 0:
            raise RuntimeError(f"Missing audio: {row['case_id']}")
        waveform = audio[0]
        decoded_seconds = len(waveform)/16000
        expected_seconds = v['video_end']-v['video_start']
        if abs(decoded_seconds-expected_seconds) > 0.2:
            raise RuntimeError(f"Audio duration mismatch: {row['case_id']}")
        rms = float((waveform.astype('float64')**2).mean()**0.5)
        visual_tokens = shape[0]//2*(shape[2]//28)*(shape[3]//28)
        if visual_tokens != row['dynamic_student_budget']['visual_budget_tokens']:
            raise RuntimeError(f"Visual token mismatch: {row['case_id']}")
        record = dict(case_id=row['case_id'], actual_frame_shape=shape, actual_sample_fps=fps,
                      actual_audio_samples=len(waveform), actual_audio_seconds=decoded_seconds,
                      audio_rms=rms, estimated_visual_tokens=visual_tokens, passed=True)
        outputs.append(record)
        print(json.dumps(record), flush=True)
        del frames, audio, waveform
    (a.root/'actual_media_smoke.json').write_text(json.dumps(outputs, indent=2)+'\n')


if __name__ == '__main__':
    main()
