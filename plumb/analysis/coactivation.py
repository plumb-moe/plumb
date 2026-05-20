from __future__ import annotations

from dataclasses import dataclass, field

from ..topology import Topology


@dataclass
class CoactivationPair:
    expert_a: int
    expert_b: int
    coactivation_count: int   # proxy: load_a * load_b
    cross_gpu: bool


@dataclass
class CrossGPUCoactivationResult:
    layer_id: int
    cross_gpu_coactivation_rate: float
    estimated_extra_hops_per_pass: float  # cross_gpu_rate * layer_token_count
    total_coactivation_count: int
    top_misplaced_pairs: list[CoactivationPair] = field(default_factory=list)


def build_coactivation_matrix(
    snapshot: dict[tuple[int, int], int],
) -> dict[int, dict[tuple[int, int], int]]:
    """Build a per-layer estimated co-activation matrix from expert load counts.

    Uses load_a * load_b as a proxy for the frequency with which expert pair (a, b)
    is selected for the same token under top-k routing.  Keys are (a, b) with a < b.
    The proxy preserves relative ordering: hot pairs have large values, cold pairs small.
    """
    by_layer: dict[int, dict[int, int]] = {}
    for (lid, eid), count in snapshot.items():
        by_layer.setdefault(lid, {})[eid] = count

    result: dict[int, dict[tuple[int, int], int]] = {}
    for lid, expert_counts in by_layer.items():
        experts = sorted(expert_counts)
        pairs: dict[tuple[int, int], int] = {}
        for i, ea in enumerate(experts):
            for eb in experts[i + 1:]:
                pairs[(ea, eb)] = expert_counts[ea] * expert_counts[eb]
        result[lid] = pairs

    return result


def compute_cross_gpu_coactivation(
    matrix: dict[int, dict[tuple[int, int], int]],
    placement: dict[tuple[int, int], list[int]],
    loads: dict[tuple[int, int], int],
    topology: Topology,
) -> list[CrossGPUCoactivationResult]:
    """For each layer: classify co-activation pairs as same-GPU vs cross-GPU.

    Args:
        matrix:    Output of build_coactivation_matrix.
        placement: (layer_id, expert_id) → [gpu, ...]; primary GPU is index 0.
        loads:     (layer_id, expert_id) → token_count (for normalising extra-hops estimate).
        topology:  NUMA topology (unused here but kept for API symmetry with comms).
    """
    layer_tokens: dict[int, int] = {}
    for (lid, _), count in loads.items():
        layer_tokens[lid] = layer_tokens.get(lid, 0) + count

    results: list[CrossGPUCoactivationResult] = []
    for layer_id, pairs in sorted(matrix.items()):
        total = 0
        cross_count = 0
        misplaced: list[CoactivationPair] = []

        for (ea, eb), count in pairs.items():
            gpu_a = (placement.get((layer_id, ea)) or [0])[0]
            gpu_b = (placement.get((layer_id, eb)) or [0])[0]
            is_cross = gpu_a != gpu_b
            total += count
            if is_cross:
                cross_count += count
                misplaced.append(CoactivationPair(
                    expert_a=ea, expert_b=eb,
                    coactivation_count=count, cross_gpu=True,
                ))

        if total == 0:
            continue

        misplaced.sort(key=lambda p: -p.coactivation_count)
        rate = round(cross_count / total, 4)
        tokens = layer_tokens.get(layer_id, 0)
        extra_hops = round(rate * tokens, 1) if tokens else 0.0

        results.append(CrossGPUCoactivationResult(
            layer_id=layer_id,
            cross_gpu_coactivation_rate=rate,
            estimated_extra_hops_per_pass=extra_hops,
            total_coactivation_count=total,
            top_misplaced_pairs=misplaced[:10],
        ))

    return results
