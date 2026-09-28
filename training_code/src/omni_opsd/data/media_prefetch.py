"""Bounded, CPU-only media lookahead using the unchanged Qwen media reader.

Workers are spawned, never forked from an initialized NPU process. Tensor
payloads travel as NumPy byte arrays, not torch shared-memory handles. This
module does not load model weights or alter sampling, resizing, or audio.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
import time


@dataclass
class TensorPayload:
    dtype: str
    shape: tuple
    data: object


def pack_tensors(value, torch):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('media worker attempted to return a non-CPU tensor')
        return TensorPayload(str(value.dtype).split('.')[-1], tuple(value.shape),
                             value.contiguous().reshape(-1).view(torch.uint8).numpy())
    if isinstance(value, dict):
        return {k: pack_tensors(v, torch) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(pack_tensors(v, torch) for v in value)
    if isinstance(value, list):
        return [pack_tensors(v, torch) for v in value]
    return value


def unpack_tensors(value, torch):
    if isinstance(value, TensorPayload):
        return torch.from_numpy(value.data).view(getattr(torch, value.dtype)).reshape(value.shape)
    if isinstance(value, dict):
        return {k: unpack_tensors(v, torch) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(unpack_tensors(v, torch) for v in value)
    if isinstance(value, list):
        return [unpack_tensors(v, torch) for v in value]
    return value


def request_key(conversation, use_audio):
    raw = json.dumps([conversation, bool(use_audio)], sort_keys=True,
                     ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def cpu_groups(cpus, workers):
    cpus = sorted(set(cpus))
    if workers < 1 or len(cpus) < workers:
        raise ValueError('each media worker needs at least one reserved CPU')
    return [cpus[i::workers] for i in range(workers)]


def _initialize(groups, counter, sparse):
    with counter.get_lock():
        slot = counter.value % len(groups)
        counter.value += 1
    os.sched_setaffinity(0, groups[slot])
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sparse:
        from omni_opsd.data.sparse_full_video import install_sparse_full_reader
        install_sparse_full_reader()


def _prepare(request):
    conversation, use_audio = request
    import torch
    from qwen_omni_utils import process_mm_info
    started = time.monotonic()
    result = process_mm_info(conversation, use_audio_in_video=use_audio,
                             return_video_kwargs=True, return_video_metadata=True)
    initialized = bool(torch.npu.is_initialized()) if hasattr(torch, 'npu') else False
    if initialized:
        raise RuntimeError('CPU media worker unexpectedly initialized an NPU')
    payload = pack_tensors(result, torch)
    return payload, {'worker_pid': os.getpid(), 'cpus': sorted(os.sched_getaffinity(0)),
                     'media_seconds': time.monotonic() - started, 'npu_initialized': initialized}


class MediaPrefetch:
    def __init__(self, requests, *, workers=4, depth=8, sparse=True, timeout=600, cpus=None):
        if not workers <= depth <= 2 * workers:
            raise ValueError('lookahead depth must be between workers and twice workers')
        groups = cpu_groups(os.sched_getaffinity(0) if cpus is None else cpus, workers)
        context = mp.get_context('spawn')
        self.pool = context.Pool(workers, initializer=_initialize,
                                 initargs=(groups, context.Value('i', 0), sparse))
        self.requests = iter(requests)
        self.pending = deque()
        self.depth = depth
        self.timeout = timeout
        self.closed = False
        self.last_trace = None
        try:
            self._fill()
        except BaseException:
            self.close()
            raise

    def _fill(self):
        while len(self.pending) < self.depth:
            try:
                request = next(self.requests)
            except StopIteration:
                break
            key = request_key(*request)
            self.pending.append((key, self.pool.apply_async(_prepare, (request,))))

    def __call__(self, conversation, *, use_audio_in_video, return_video_kwargs,
                 return_video_metadata):
        if self.closed or return_video_kwargs is not True or return_video_metadata is not True:
            raise ValueError('prefetch requires an open reader and the formal metadata mode')
        key = request_key(conversation, use_audio_in_video)
        position = next((i for i, (candidate, _) in enumerate(self.pending) if candidate == key), None)
        if position is None:
            raise ValueError('requested media does not match bounded lookahead plan')
        # A template error may skip a row before calling the media reader.
        # Skip only explicitly earlier planned requests; never return another QA's media.
        for _ in range(position):
            self.pending.popleft()
        _, future = self.pending.popleft()
        self._fill()
        started = time.monotonic()
        payload, trace = future.get(timeout=self.timeout)
        trace['parent_wait_seconds'] = time.monotonic() - started
        self.last_trace = trace
        import torch
        return unpack_tensors(payload, torch)

    def close(self):
        if not self.closed:
            self.closed = True
            self.pool.terminate()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
