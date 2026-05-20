from __future__ import annotations

from dataclasses import dataclass, field

from numa_topology.pcie import PCIeTopology

from ..topology import Topology


@dataclass
class CommunicationConstants:
    """Latency constants (µs) for each GPU-to-GPU link type. Override for hardware-specific tuning."""
    same_numa_us: float = 2.0         # same NUMA node, cross-GPU, PCIe intra-socket
    cross_numa_pcie_us: float = 15.0  # cross-NUMA PCIe (inter-socket)
    nvlink_us: float = 3.0            # NVLink (takes priority over NUMA check)


@dataclass
class LayerCommunicationStats:
    layer_id: int
    cross_numa_rate: float        # fraction of dispatches going to a different NUMA node
    cross_pcie_rate: float        # fraction of dispatches over PCIe (non-NVLink, cross-GPU)
    estimated_overhead_us: float  # total weighted dispatch latency across all token loads
    total_dispatches: int


@dataclass
class CommunicationCostResult:
    current_overhead_us: float
    recommended_overhead_us: float
    delta_us: float                    # recommended - current; negative = improvement
    per_layer_current: list[LayerCommunicationStats] = field(default_factory=list)
    per_layer_recommended: list[LayerCommunicationStats] = field(default_factory=list)


def compute_communication_cost(
    current_placement: dict[tuple[int, int], list[int]],
    recommended_placement: dict[tuple[int, int], list[int]],
    expert_loads: dict[tuple[int, int], int],
    topology: Topology,
    pcie_topology: PCIeTopology,
    constants: CommunicationConstants | None = None,
    src_gpu: int = 0,
) -> CommunicationCostResult:
    """Estimate weighted dispatch latency for current and recommended expert placements.

    For each token dispatch (layer, expert) → primary GPU, multiplies the token count
    by the link latency and sums across all layers. Negative delta_us means the
    recommended placement reduces communication overhead.

    Args:
        current_placement:     (layer_id, expert_id) → [gpu, ...], existing placement.
        recommended_placement: (layer_id, expert_id) → [gpu, ...], proposed placement.
        expert_loads:          (layer_id, expert_id) → token_count from profiling.
        topology:              NUMA affinity map.
        pcie_topology:         PCIe link info (identifies NVLink-capable GPUs).
        constants:             Latency constants in µs; uses defaults if None.
        src_gpu:               Dispatching GPU index (default 0).
    """
    if constants is None:
        constants = CommunicationConstants()

    nvlink_gpus: frozenset[int] = frozenset(g.gpu_idx for g in pcie_topology.gpus if g.nvlink)

    def _compute_layers(
        placement: dict[tuple[int, int], list[int]],
    ) -> tuple[float, list[LayerCommunicationStats]]:
        # layer_id → [total, cross_numa, cross_pcie, overhead_us]
        by_layer: dict[int, list] = {}

        for (layer_id, _expert_id), count in expert_loads.items():
            gpus = placement.get((layer_id, _expert_id), [src_gpu])
            dst_gpu = gpus[0] if gpus else src_gpu

            entry = by_layer.setdefault(layer_id, [0, 0, 0, 0.0])
            entry[0] += count

            if dst_gpu == src_gpu:
                continue

            cross_numa = not topology.same_numa(src_gpu, dst_gpu)
            use_nvlink = dst_gpu in nvlink_gpus

            if use_nvlink:
                latency = constants.nvlink_us
            elif cross_numa:
                latency = constants.cross_numa_pcie_us
            else:
                latency = constants.same_numa_us

            if cross_numa:
                entry[1] += count
            if not use_nvlink:
                entry[2] += count
            entry[3] += count * latency

        stats: list[LayerCommunicationStats] = []
        total_overhead = 0.0
        for layer_id, (total, cross_numa, cross_pcie, overhead) in sorted(by_layer.items()):
            if total == 0:
                continue
            stats.append(LayerCommunicationStats(
                layer_id=layer_id,
                cross_numa_rate=round(cross_numa / total, 4),
                cross_pcie_rate=round(cross_pcie / total, 4),
                estimated_overhead_us=round(overhead, 3),
                total_dispatches=total,
            ))
            total_overhead += overhead

        return round(total_overhead, 3), stats

    current_overhead, per_layer_current = _compute_layers(current_placement)
    recommended_overhead, per_layer_recommended = _compute_layers(recommended_placement)

    return CommunicationCostResult(
        current_overhead_us=current_overhead,
        recommended_overhead_us=recommended_overhead,
        delta_us=round(recommended_overhead - current_overhead, 3),
        per_layer_current=per_layer_current,
        per_layer_recommended=per_layer_recommended,
    )
