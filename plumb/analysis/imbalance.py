from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..counter import ActivationCounter


@dataclass
class LayerImbalance:
    layer_id: int
    imbalance_ratio: float   # max(load) / mean(load)
    max_expert_id: int
    min_expert_id: int
    expert_loads: dict[int, int]  # expert_id -> token_count


def compute_imbalance(counter: ActivationCounter) -> list[LayerImbalance]:
    snapshot = counter.snapshot()
    if not snapshot:
        return []

    by_layer: dict[int, dict[int, int]] = {}
    for (layer_id, expert_id), count in snapshot.items():
        by_layer.setdefault(layer_id, {})[expert_id] = count

    results = []
    for layer_id in sorted(by_layer):
        loads = by_layer[layer_id]
        values = np.array(list(loads.values()), dtype=float)
        mean = float(values.mean())
        if mean == 0.0:
            continue
        results.append(LayerImbalance(
            layer_id=layer_id,
            imbalance_ratio=round(float(values.max() / mean), 4),
            max_expert_id=max(loads, key=loads.__getitem__),
            min_expert_id=min(loads, key=loads.__getitem__),
            expert_loads=loads,
        ))
    return results
