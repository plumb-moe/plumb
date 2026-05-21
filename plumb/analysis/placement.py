from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..counter import ActivationCounter
from ..topology import Topology

if TYPE_CHECKING:
    from numa_topology.pcie import PCIeTopology

    from .comms import CommunicationConstants

logger = logging.getLogger(__name__)

# Improvement bounds from HarMoEny paper (arXiv:2506.12417)
_IMPROVEMENT_MIN = 37.0
_IMPROVEMENT_MAX = 70.0


@dataclass
class PlacementRecommendation:
    expert_placement: dict[tuple[int, int], list[int]]  # (layer_id, expert_id) -> [gpu_id, ...]
    method: str                                          # "eplb" | "comm_aware_greedy" | "greedy" | "none"
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
    pcie_topology: PCIeTopology | None = None,
    comm_constants: CommunicationConstants | None = None,
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
        if len(topology.numa_nodes()) > 1:
            # Non-trivial topology: use comm-aware greedy to penalise cross-NUMA routing
            placement = _comm_aware_greedy(
                load, n_gpus, layers, experts, topology, comm_constants
            )
            method = "comm_aware_greedy"
        else:
            placement = _greedy(load, n_gpus, layers, experts)
            method = "greedy"

    placement = _numa_finetune(
        placement, topology, load, layers, experts, pcie_topology=pcie_topology
    )

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


def _comm_aware_greedy(
    load: np.ndarray,
    n_gpus: int,
    layers: list[int],
    experts: list[int],
    topology: Topology,
    constants: CommunicationConstants | None = None,
    src_gpu: int = 0,
) -> dict[tuple[int, int], list[int]]:
    """Greedy placement that jointly minimises load imbalance and communication cost.

    For each expert (in descending load order), picks the GPU that minimises
    a normalised sum of:
      - relative load (current_tokens_on_gpu / total_layer_tokens)
      - relative communication cost from src_gpu

    When all GPUs share a NUMA node the comm_cost vector is zero, so this
    degenerates to the same result as _greedy.
    """
    from .comms import CommunicationConstants as CC

    if constants is None:
        constants = CC()

    # Build per-GPU communication cost from src_gpu (µs, unnormalised).
    # Same-NUMA GPUs all get 0 — the meaningful penalty is cross-NUMA only.
    # This ensures that on a flat (single-NUMA) system the comm vector is
    # all-zeros and the function degenerates to pure load-balance (like _greedy).
    comm_us = np.array([
        0.0 if topology.same_numa(src_gpu, g)
        else constants.cross_numa_pcie_us
        for g in range(n_gpus)
    ], dtype=np.float64)

    # Normalise to [0, 1] so it is scale-compatible with load fractions
    max_comm = comm_us.max()
    comm_norm = comm_us / max_comm if max_comm > 0 else comm_us

    placement: dict[tuple[int, int], list[int]] = {}
    for li, lid in enumerate(layers):
        order = np.argsort(-load[li])  # hottest first
        gpu_tokens = np.zeros(n_gpus, dtype=np.float64)
        for ei in order:
            total = gpu_tokens.sum() or 1.0
            load_frac = gpu_tokens / total          # fraction of load already on each GPU
            scores = load_frac + comm_norm
            best_gpu = int(np.argmin(scores))
            placement[(lid, experts[ei])] = [best_gpu]
            gpu_tokens[best_gpu] += float(load[li, ei])

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


def coactivation_partition(
    counter: ActivationCounter,
    topology: Topology,
    num_gpus: int | None = None,
) -> dict[tuple[int, int], list[int]]:
    """Place experts to co-locate frequently co-activated pairs on the same GPU.

    Builds a co-activation weighted graph per layer and partitions it into
    num_gpus communities (pymetis → networkx louvain → greedy fallback).
    Applies a load-balance correction if any partition holds >2× average load.
    """
    from .coactivation import build_coactivation_matrix

    snapshot = counter.snapshot()
    if not snapshot:
        return {}

    n_gpus = num_gpus or max(len(topology.gpu_to_numa), 1)
    layers = sorted({k[0] for k in snapshot})
    coact = build_coactivation_matrix(snapshot)

    placement: dict[tuple[int, int], list[int]] = {}
    for lid in layers:
        layer_experts = sorted({eid for (l, eid) in snapshot if l == lid})
        if not layer_experts:
            continue

        pairs = coact.get(lid, {})
        loads = {eid: snapshot.get((lid, eid), 0) for eid in layer_experts}

        partition = _partition_experts(layer_experts, pairs, n_gpus)
        partition = _balance_partition(partition, loads, n_gpus)

        for eid, gpu in partition.items():
            placement[(lid, eid)] = [gpu]

    return placement


def _partition_experts(
    experts: list[int],
    coact_pairs: dict[tuple[int, int], int],
    n_gpus: int,
) -> dict[int, int]:
    """expert_id → gpu_id via graph partitioning. Three fallback levels."""
    if len(experts) <= n_gpus:
        return {eid: i % n_gpus for i, eid in enumerate(experts)}

    try:
        return _partition_pymetis(experts, coact_pairs, n_gpus)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("pymetis partition failed (%s), trying networkx", exc)

    try:
        return _partition_networkx(experts, coact_pairs, n_gpus)
    except Exception as exc:
        logger.debug("networkx partition failed (%s), using greedy co-activation", exc)

    return _partition_greedy_coact(experts, coact_pairs, n_gpus)


def _partition_pymetis(
    experts: list[int],
    coact_pairs: dict[tuple[int, int], int],
    n_gpus: int,
) -> dict[int, int]:
    import pymetis  # type: ignore[import]

    idx = {e: i for i, e in enumerate(experts)}
    n = len(experts)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    eweights: list[list[int]] = [[] for _ in range(n)]
    for (ea, eb), w in coact_pairs.items():
        if ea not in idx or eb not in idx:
            continue
        ia, ib = idx[ea], idx[eb]
        adjacency[ia].append(ib)
        adjacency[ib].append(ia)
        eweights[ia].append(w)
        eweights[ib].append(w)

    _, parts = pymetis.part_graph(n_gpus, adjacency=adjacency, eweights=eweights)
    return {experts[i]: parts[i] for i in range(n)}


def _partition_networkx(
    experts: list[int],
    coact_pairs: dict[tuple[int, int], int],
    n_gpus: int,
) -> dict[int, int]:
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(experts)
    for (ea, eb), w in coact_pairs.items():
        if ea in G and eb in G:
            G.add_edge(ea, eb, weight=w)

    communities = nx.community.louvain_communities(G, seed=0)
    # Map communities → GPUs: merge smallest communities until we have n_gpus
    while len(communities) > n_gpus:
        communities.sort(key=len)
        merged = communities[0] | communities[1]
        communities = [merged] + communities[2:]

    result: dict[int, int] = {}
    for gpu, community in enumerate(communities[:n_gpus]):
        for eid in community:
            result[eid] = gpu
    # Any experts not assigned (empty communities edge case) → round-robin
    for i, eid in enumerate(e for e in experts if e not in result):
        result[eid] = i % n_gpus
    return result


def _partition_greedy_coact(
    experts: list[int],
    coact_pairs: dict[tuple[int, int], int],
    n_gpus: int,
) -> dict[int, int]:
    """Greedy: assign each expert to the GPU holding the most of its co-activated partners."""
    result: dict[int, int] = {}
    gpu_members: dict[int, set[int]] = {g: set() for g in range(n_gpus)}
    gpu_load: dict[int, int] = {g: 0 for g in range(n_gpus)}

    # Sort experts by total co-activation weight descending (busiest first)
    expert_weight = {e: 0 for e in experts}
    for (ea, eb), w in coact_pairs.items():
        expert_weight[ea] = expert_weight.get(ea, 0) + w
        expert_weight[eb] = expert_weight.get(eb, 0) + w
    order = sorted(experts, key=lambda e: -expert_weight.get(e, 0))

    for eid in order:
        # Score each GPU: co-activation weight already on GPU minus load penalty
        scores: dict[int, float] = {}
        for g in range(n_gpus):
            coact_score = sum(
                coact_pairs.get((min(eid, m), max(eid, m)), 0)
                for m in gpu_members[g]
            )
            scores[g] = coact_score - gpu_load[g]
        best = max(scores, key=scores.__getitem__)
        result[eid] = best
        gpu_members[best].add(eid)
        gpu_load[best] += expert_weight.get(eid, 0)

    return result


def _balance_partition(
    partition: dict[int, int],
    loads: dict[int, int],
    n_gpus: int,
) -> dict[int, int]:
    """Move experts from overloaded GPUs to underloaded ones until no GPU has >2× average."""
    gpu_load = {g: 0 for g in range(n_gpus)}
    for eid, gpu in partition.items():
        gpu_load[gpu] = gpu_load.get(gpu, 0) + loads.get(eid, 0)

    total = sum(gpu_load.values()) or 1
    avg = total / n_gpus

    result = dict(partition)
    for _ in range(len(partition)):  # bounded iterations
        overloaded = max(gpu_load, key=gpu_load.__getitem__)
        underloaded = min(gpu_load, key=gpu_load.__getitem__)
        if gpu_load[overloaded] <= 2 * avg:
            break
        # Move the coldest expert off the overloaded GPU
        candidates = sorted(
            [e for e, g in result.items() if g == overloaded],
            key=lambda e: loads.get(e, 0),
        )
        if not candidates:
            break
        mover = candidates[0]
        gpu_load[overloaded] -= loads.get(mover, 0)
        gpu_load[underloaded] += loads.get(mover, 0)
        result[mover] = underloaded

    return result


def _numa_finetune(
    placement: dict[tuple[int, int], list[int]],
    topology: Topology,
    load: np.ndarray,
    layers: list[int],
    experts: list[int],
    pcie_topology: PCIeTopology | None = None,
) -> dict[tuple[int, int], list[int]]:
    """Pin the hottest experts to the highest-quality GPU connections.

    With pcie_topology: ranks all GPUs by theoretical_bw_gbs and pins hot
    experts to highest-bandwidth GPUs, works even on single-NUMA systems.

    Without pcie_topology: original NUMA-0 pinning behaviour (no-op on
    single-NUMA systems).
    """
    result = dict(placement)

    if pcie_topology is not None:
        gpu_bw = {g.gpu_idx: g.theoretical_bw_gbs for g in pcie_topology.gpus}
        if len(gpu_bw) <= 1:
            return result
        preferred = sorted(gpu_bw.keys(), key=lambda g: -gpu_bw[g])
    elif len(topology.numa_nodes()) <= 1:
        return placement
    else:
        preferred = topology.gpus_in_numa(0)

    if not preferred:
        return result

    for li, lid in enumerate(layers):
        hot_experts = sorted(range(len(experts)), key=lambda ei: -load[li, ei])
        for rank, ei in enumerate(hot_experts[: len(preferred)]):
            key = (lid, experts[ei])
            primary = preferred[rank % len(preferred)]
            existing = result.get(key, [])
            result[key] = [primary] + [g for g in existing[1:] if g != primary]

    return result
