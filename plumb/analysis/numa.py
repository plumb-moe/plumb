from __future__ import annotations

from dataclasses import dataclass

from ..topology import Topology


@dataclass
class NumaStats:
    layer_id: int
    cross_numa_rate: float
    total_dispatches: int
    cross_numa_dispatches: int


def compute_cross_numa(
    expert_placement: dict[tuple[int, int], int | list[int]],  # (layer, expert) -> gpu or [gpu, ...]
    expert_loads: dict[tuple[int, int], int],                   # (layer, expert) -> token_count
    topology: Topology,
    src_gpu: int = 0,
) -> list[NumaStats]:
    """Cross-NUMA dispatch rate per layer given a placement map."""
    by_layer: dict[int, tuple[int, int]] = {}  # layer -> (total, cross)

    for (layer_id, expert_id), count in expert_loads.items():
        dst_raw = expert_placement.get((layer_id, expert_id), 0)
        dst_gpu = dst_raw[0] if isinstance(dst_raw, list) else dst_raw
        total, cross = by_layer.get(layer_id, (0, 0))
        total += count
        if not topology.same_numa(src_gpu, dst_gpu):
            cross += count
        by_layer[layer_id] = (total, cross)

    return [
        NumaStats(
            layer_id=layer_id,
            cross_numa_rate=round(cross / total, 4) if total else 0.0,
            total_dispatches=total,
            cross_numa_dispatches=cross,
        )
        for layer_id, (total, cross) in sorted(by_layer.items())
        if total > 0
    ]
