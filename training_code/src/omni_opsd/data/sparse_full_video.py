"""Memory-bounded Full-video RGB conversion with conservative fallback.

Decode every source frame but convert only Qwen's selected indices to RGB.
Interval reads, unreliable frame counts, and unusual timestamps retain the
original reader. No resampling, resizing, or timestamp repair is performed.
"""
from __future__ import annotations

import logging
import math


def sparse_full_read(ele, smart_nframes):
    import av
    import torch

    if float(ele.get('video_start', 0) or 0) != 0:
        raise ValueError('interval read requires original reader')
    path = str(ele['video']).removeprefix('file://')
    with av.open(path) as container:
        stream = container.streams.video[0]
        total = stream.frames
        fps = float(stream.average_rate or 0)
        if total < 2 or not math.isfinite(fps) or fps <= 0 or stream.duration is None:
            raise ValueError('unreliable stream metadata')
        stream_end = float((stream.start_time or 0) + stream.duration) * float(stream.time_base)
        end = ele.get('video_end')
        if end is not None and float(end) < stream_end:
            raise ValueError('requested end may trim source frames')
        count = smart_nframes(ele, total_frames=total, video_fps=fps)
        indices = torch.linspace(0, total - 1, count).round().long()
        wanted = set(indices.tolist())
        selected = {}
        observed = 0
        previous_time = -math.inf
        for index, frame in enumerate(container.decode(video=0)):
            timestamp = frame.time
            if timestamp is None or timestamp < 0 or timestamp <= previous_time:
                raise ValueError('unsupported source timestamp sequence')
            if end is not None and timestamp > float(end):
                raise ValueError('frame extends beyond requested interval')
            previous_time = timestamp
            if index in wanted:
                selected[index] = torch.from_numpy(frame.to_ndarray(format='rgb24'))
            observed = index + 1
        if observed != total or len(selected) != len(wanted):
            raise ValueError('decoded count differs from stream frame count')
        video = torch.stack([selected[i] for i in indices.tolist()]).permute(0, 3, 1, 2)
        return video, {'fps': fps, 'frames_indices': indices,
                       'total_num_frames': observed, 'video_backend': 'pyav_sparse_full',
                       'last_source_timestamp': previous_time}, count / total * fps


def install_sparse_full_reader():
    from qwen_omni_utils.v2_5 import vision_process

    original = vision_process.VIDEO_READER_BACKENDS['torchvision']
    if getattr(original, '_omni_sparse_full', False):
        return

    def reader(ele):
        if float(ele.get('video_start', 0) or 0) != 0:
            return original(ele)
        try:
            return sparse_full_read(ele, vision_process.smart_nframes)
        except Exception as exc:
            logging.getLogger(__name__).info('sparse Full fallback: %s', type(exc).__name__)
            return original(ele)

    reader._omni_sparse_full = True
    vision_process.VIDEO_READER_BACKENDS['torchvision'] = reader
