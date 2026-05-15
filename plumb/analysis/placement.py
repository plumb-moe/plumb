from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..counter import ActivationCounter
from ..topology import Topology

logger = logging.getLogger(__name__)

# Improvement bounds from HarMoEny paper (arXiv:2506.12417)
_IMPROVEMENT_MIN = 37.0
_IMPROVEMENT_MAX = 70.0


@dataclass
class PlacementRecommendation:
    expert_placement: dict[tuple[int, int], int]  # (layer_id, expert_id) -> gpu_id
    method: str                                    # "eplb" | "greedy" | "none"
    estimated_improvement_pct_min: float
    estimated_improvement_pct_max: float
    estimated_improvement_pct: float               # point estimate from mean imbalance ratio
    warning: str = ""                              # non-empty when method == "none"

# Imbalance below this threshold → placement not recommended
_LOW_IMBALANCE_THRESHOLD = 3.0


def recommend_placement(
    counter: ActivationCounter,
    topology: Topology,
    num_gpus: int | None = None,
) -> PlacementRecommendation | None:
    snapshot = counter.snapshot()
    if not snapshot:
        return None

    n_gpus = num_gpus or max(len(topology.gpu_to_numa), 1)
    layers = sorted({k[0] for k in snapshot})
    experts = sorted({k[1] for k in snapshot})

    if not experts:
        return None

    # Build (num_layers, num_experts) load matrix
    load = np.zeros((len(layers), len(experts)), dtype=np.float32)
    for (lid, eid), count in snapshot.items():
        load[layers.index(lid), experts.index(eid)] = count

    # Compute peak imbalance ratio: max-expert / mean-expert per layer
    row_means = np.where(load.mean(axis=1) > 0, load.mean(axis=1), 1.0)
    peak_imbalance = float(np.max(load.max(axis=1) / row_means))

    if peak_imbalance < _LOW_IMBALANCE_THRESHOLD:
        warning_msg = (
            f"Expert load imbalance is low (peak ratio {peak_imbalance:.2f}× < "
            f"{_LOW_IMBALANCE_THRESHOLD}×) — placement rebalancing not recommended."
        )
        logger.info(warning_msg)
        return PlacementRecommendation(
            expert_placement={},
            method="none",
            estimated_improvement_pct_min=_IMPROVEMENT_MIN,
            estimated_improvement_pct_max=_IMPROVEMENT_MAX,
            estimated_improvement_pct=0.0,
            warning=warning_msg,
        )

    placement, method = _try_eplb(load, n_gpus, layers, experts)
    if placement is None:
        placement = _greedy(load, n_gpus, layers, experts)
        method = "greedy"

    placement = _numa_finetune(placement, topology, load, layers, experts)

    mean_ratio = float(np.mean(load.max(axis=1) / np.where(load.mean(axis=1) > 0, load.mean(axis=1), 1.0)))
    point_est = float(np.clip((1.0 - 1.0 / max(mean_ratio, 1.01)) * 70.0, _IMPROVEMENT_MIN, _IMPROVEMENT_MAX))

    return PlacementRecommendation(
        expert_placement=placement,
        method=method,
        estimated_improvement_pct_min=_IMPROVEMENT_MIN,
        estimated_improvement_pct_max=_IMPROVEMENT_MAX,
        estimated_improvement_pct=round(point_est, 1),
    )


def _try_eplb(
    load: np.ndarray,
    n_gpus: int,
    layers: list[int],
    experts: list[int],
) -> tuple[dict[tuple[int, int], int] | None, str]:
    try:
        import torch
        from eplb import rebalance  # type: ignore[import]

        n_layers, n_experts = load.shape
        weight = torch.tensor(load)
        # rebalance(weight, num_physical, num_groups, num_nodes, num_gpus_per_group)
        physical, _ = rebalance(weight, n_gpus * n_experts, n_gpus, 1, n_gpus)
        placement: dict[tuple[int, int], int] = {}
        for li, lid in enumerate(layers):
            for ei, eid in enumerate(experts):
                placement[(lid, eid)] = int(physical[li, ei].item()) % n_gpus
        logger.info("EPLB placement computed")
        return placement, "eplb"
    except ImportError:
        logger.debug("EPLB not available, using greedy")
    except Exception as e:
        logger.warning("EPLB failed (%s), falling back to greedy", e)
    return None, ""


def _greedy(
    load: np.ndarray,
    n_gpus: int,
    layers: list[int],
    experts: list[int],
) -> dict[tuple[int, int], int]:
    """Spread hottest experts across GPUs round-robin per layer."""
    placement: dict[tuple[int, int], int] = {}
    for li, lid in enumerate(layers):
        order = np.argsort(-load[li])  # hottest first
        for rank, ei in enumerate(order):
            placement[(lid, experts[ei])] = rank % n_gpus
    return placement


def _numa_finetune(
    placement: dict[tuple[int, int], int],
    topology: Topology,
    load: np.ndarray,
    layers: list[int],
    experts: list[int],
) -> dict[tuple[int, int], int]:
    """Pin the hottest experts in each layer to NUMA-0 GPUs."""
    if len(topology.numa_nodes()) <= 1:
        return placement

    result = dict(placement)
    numa0 = topology.gpus_in_numa(0)
    if not numa0:
        return result

    for li, lid in enumerate(layers):
        hot_experts = sorted(range(len(experts)), key=lambda ei: -load[li, ei])
        for rank, ei in enumerate(hot_experts[: len(numa0)]):
            result[(lid, experts[ei])] = numa0[rank % len(numa0)]

    return result
