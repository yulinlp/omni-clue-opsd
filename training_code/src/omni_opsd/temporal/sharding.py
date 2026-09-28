"""Deterministic sample sharding shared by inference backends."""

from __future__ import annotations

import os

from .dataset import Sample


def shard_samples(
    samples: list[Sample],
    *,
    shard_index: int,
    num_shards: int,
    strategy: str = "strided",
) -> list[Sample]:
    """Assign samples to workers, optionally keeping each video's QAs together."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    strategy = str(strategy).strip().lower()
    if strategy == "strided":
        return [sample for index, sample in enumerate(samples) if index % num_shards == shard_index]
    if strategy != "video_grouped":
        raise ValueError(f"unsupported sharding strategy: {strategy}")

    groups: dict[str, list[tuple[int, Sample]]] = {}
    for index, sample in enumerate(samples):
        key = os.path.realpath(os.path.abspath(sample.video_path))
        groups.setdefault(key, []).append((index, sample))
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[1][0][0], item[0]))
    loads = [0] * num_shards
    assigned: list[list[tuple[int, list[tuple[int, Sample]]]]] = [[] for _ in range(num_shards)]
    for _, group in ordered:
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        assigned[target].append((group[0][0], group))
        loads[target] += len(group)
    selected: list[Sample] = []
    for _, group in sorted(assigned[shard_index], key=lambda item: item[0]):
        selected.extend(sample for _, sample in group)
    return selected
