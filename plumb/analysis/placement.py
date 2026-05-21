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
    expert_placement: dict[tuple[int, int], list[int]]  # (layer_id, expert_id) -> [gpu_id, ...]
    method: str                                          # "eplb" | "greedy" | "none"
    estimated_improvement_pct_min: float
    estimated_improvement_pct_max: float
    estimated_improvement_pct: float                     # point estimate from mean imbalance ratio
    warning: str = ""                                    # non-empty when method == "none"

# Imbalance below this threshold → placement not recommended
_LOW_IMBALANCE_THRESHOLD = 3.0


def recommend_placement(
    counter: ActivationCounter,
    topology: Topology,
    num_gpus: int | None = None,
    num_redundant_experts: int = 0,
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

    placement, method = _try_eplb(load, n_gpus, layers, experts, num_redundant_experts)
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
    num_redundant_experts: int = 0,
) -> tuple[dict[tuple[int, int], list[int]] | None, str]:
    try:
        import torch
        from eplb import rebalance_experts  # type: ignore[import]

        n_layers, n_experts = load.shape
        # num_physical > n_logical enables replication of hot experts
        n_physical = n_gpus * n_experts + num_redundant_experts
        weight = torch.tensor(load)
        # rebalance_experts returns (phy2log, log2phy, logcnt)
        # log2phy[li, ei, replica] = physical slot; GPU = slot // n_experts
        # logcnt[li, ei] = number of valid replicas for expert ei in layer li
        _, log2phy, logcnt = rebalance_experts(weight, n_physical, n_gpus, 1, n_gpus)
        placement: dict[tuple[int, int], list[int]] = {}
        for li, lid in enumerate(layers):
            for ei, eid in enumerate(experts):
                cnt = int(logcnt[li, ei].item())
                gpus: list[int] = []
                seen: set[int] = set()
                for ri in range(max(cnt, 1)):
                    slot = int(log2phy[li, ei, ri].item())
                    g = slot // n_experts
                    if g not in seen:
                        gpus.append(g)
                        seen.add(g)
                placement[(lid, eid)] = gpus
        logger.info(
            "EPLB placement computed (redundant_experts=%d)", num_redundant_experts
        )
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
) -> dict[tuple[int, int], list[int]]:
    """Spread hottest experts across GPUs round-robin per layer."""
    placement: dict[tuple[int, int], list[int]] = {}
    for li, lid in enumerate(layers):
        order = np.argsort(-load[li])  # hottest first
        for rank, ei in enumerate(order):
            placement[(lid, experts[ei])] = [rank % n_gpus]
    return placement


def worst_case_placement(
    counter: ActivationCounter,
    topology: Topology,
    num_gpus: int | None = None,
) -> dict[tuple[int, int], list[int]]:
    """Adversarial placement: concentrate hot experts on GPU 0 to maximise imbalance.

    Inverse of _greedy: instead of spreading hottest experts round-robin across GPUs,
    assigns them in contiguous rank-sorted blocks so GPU 0 owns all the busiest experts.
    """
    snapshot = counter.snapshot()
    if not snapshot:
        return {}

    n_gpus = num_gpus or max(len(topology.gpu_to_numa), 1)
    layers = sorted({k[0] for k in snapshot})
    experts = sorted({k[1] for k in snapshot})

    load = np.zeros((len(layers), len(experts)), dtype=np.float32)
    for (lid, eid), count in snapshot.items():
        load[layers.index(lid), experts.index(eid)] = count

    experts_per_gpu = max(1, len(experts) // n_gpus)
    placement: dict[tuple[int, int], list[int]] = {}
    for li, lid in enumerate(layers):
        order = np.argsort(-load[li])  # hottest first
        for rank, ei in enumerate(order):
            gpu = min(rank // experts_per_gpu, n_gpus - 1)
            placement[(lid, experts[ei])] = [gpu]
    return placement


def _numa_finetune(
    placement: dict[tuple[int, int], list[int]],
    topology: Topology,
    load: np.ndarray,
    layers: list[int],
    experts: list[int],
) -> dict[tuple[int, int], list[int]]:
    """Pin the primary replica of the hottest experts in each layer to NUMA-0 GPUs."""
    if len(topology.numa_nodes()) <= 1:
        return placement

    result = dict(placement)
    numa0 = topology.gpus_in_numa(0)
    if not numa0:
        return result

    for li, lid in enumerate(layers):
        hot_experts = sorted(range(len(experts)), key=lambda ei: -load[li, ei])
        for rank, ei in enumerate(hot_experts[: len(numa0)]):
            key = (lid, experts[ei])
            primary = numa0[rank % len(numa0)]
            existing = result.get(key, [])
            # Swap in the NUMA-0 GPU as primary; keep EPLB-assigned replicas
            result[key] = [primary] + [g for g in existing[1:] if g != primary]

    return result
