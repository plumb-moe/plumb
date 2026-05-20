from __future__ import annotations

import platform
from datetime import datetime, timezone

from numa_topology.pcie import PCIeTopology

from ..analysis.coactivation import build_coactivation_matrix, compute_cross_gpu_coactivation
from ..analysis.comms import compute_communication_cost
from ..analysis.heterogeneous import hottest_experts_on_fastest_gpu
from ..analysis.imbalance import compute_imbalance
from ..analysis.numa import compute_cross_numa
from ..analysis.placement import recommend_placement
from ..counter import ActivationCounter
from ..topology import Topology
from .schema import (
    CoactivationLayerReport,
    CoactivationPairReport,
    CoactivationReport,
    CommunicationCostReport,
    ExpertLoad,
    GpuCapabilityReport,
    GpuStatsReport,
    HeterogeneousPlacementReport,
    HeterogeneousTopologyReport,
    LayerReport,
    PlacementReport,
    PlacementViolationReport,
    ProfileReport,
)


def generate_report(
    counter: ActivationCounter,
    topology: Topology,
    model_name: str,
    duration_seconds: float,
    num_gpus: int | None = None,
    gpu_stats: GpuStatsReport | None = None,
    pcie_topology: PCIeTopology | None = None,
    hetero_topology=None,
) -> ProfileReport:
    imbalances = compute_imbalance(counter)
    snapshot = counter.snapshot()

    rec = recommend_placement(counter, topology, num_gpus)
    expert_placement = rec.expert_placement if rec else {}

    numa_by_layer = {
        s.layer_id: s
        for s in compute_cross_numa(expert_placement, snapshot, topology)
    }

    layers = []
    for imb in imbalances:
        total = sum(imb.expert_loads.values()) or 1
        experts = [
            ExpertLoad(
                expert_id=eid,
                token_count=count,
                activation_fraction=round(count / total, 4),
            )
            for eid, count in sorted(imb.expert_loads.items())
        ]
        numa = numa_by_layer.get(imb.layer_id)
        layers.append(LayerReport(
            layer_id=imb.layer_id,
            imbalance_ratio=imb.imbalance_ratio,
            max_expert_id=imb.max_expert_id,
            min_expert_id=imb.min_expert_id,
            cross_numa_rate=numa.cross_numa_rate if numa else None,
            experts=experts,
        ))

    placement_report = None
    if rec:
        placement_report = PlacementReport(
            method=rec.method,
            expert_placement={
                f"{lid}:{eid}": gpus
                for (lid, eid), gpus in rec.expert_placement.items()
            },
            estimated_improvement_pct_min=rec.estimated_improvement_pct_min,
            estimated_improvement_pct_max=rec.estimated_improvement_pct_max,
            estimated_improvement_pct=rec.estimated_improvement_pct,
        )

    gpu_to_numa = dict(topology.gpu_to_numa) if topology.gpu_to_numa else None
    if gpu_to_numa and len(set(gpu_to_numa.values())) <= 1:
        gpu_to_numa = None

    communication_cost_report = _compute_comm_cost(
        rec, snapshot, topology, pcie_topology,
        num_gpus or max(len(topology.gpu_to_numa), 1),
    )

    coactivation_report = _compute_coactivation(rec, snapshot, expert_placement, topology)

    hetero_topology_report, hetero_placement_report = _build_hetero_reports(
        hetero_topology, rec, snapshot,
    )

    return ProfileReport(
        model_name=model_name,
        hardware_config=_hw_desc(),
        profiling_duration_seconds=round(duration_seconds, 2),
        total_forward_passes=counter.pass_count,
        layers=layers,
        placement=placement_report,
        gpu_to_numa=gpu_to_numa,
        gpu_stats=gpu_stats,
        communication_cost=communication_cost_report,
        coactivation=coactivation_report,
        heterogeneous_topology=hetero_topology_report,
        heterogeneous_placement=hetero_placement_report,
        generated_at=datetime.now(tz=timezone.utc),
    )


def generate_report_from_snapshot(snapshot: dict) -> ProfileReport:
    """Reconstruct a ProfileReport from a live session snapshot dict.

    The snapshot must contain 'expert_counts' (written by autoattach since 0.1.0).
    """
    expert_counts_raw: dict[str, int] = snapshot.get("expert_counts", {})
    if not expert_counts_raw:
        raise ValueError("snapshot missing 'expert_counts' — requires plumb >=0.1.0 auto-attach")

    counter = ActivationCounter(window_size=len(expert_counts_raw) + 1)
    for key, count in expert_counts_raw.items():
        lid_str, eid_str = key.split(":", 1)
        counter.record(int(lid_str), int(eid_str), count)
    for _ in range(snapshot.get("pass_count", 0)):
        counter.increment_pass()

    gpu_to_numa = {int(k): int(v) for k, v in snapshot.get("gpu_to_numa", {}).items()}
    topology = Topology(gpu_to_numa) if gpu_to_numa else Topology.flat(1)

    started = snapshot.get("started_at", snapshot.get("updated_at", 0))
    updated = snapshot.get("updated_at", started)
    duration = max(0.0, updated - started)

    from numa_topology.gpu_capabilities import discover_gpu_capabilities
    hetero_topology = discover_gpu_capabilities()

    return generate_report(
        counter=counter,
        topology=topology,
        model_name=snapshot.get("model_name", "unknown"),
        duration_seconds=duration,
        num_gpus=len(gpu_to_numa) or None,
        hetero_topology=hetero_topology,
    )


def _compute_coactivation(
    rec: object,
    snapshot: dict[tuple[int, int], int],
    expert_placement: dict[tuple[int, int], list[int]],
    topology: Topology,
) -> CoactivationReport | None:
    if not rec or getattr(rec, "method", "none") == "none":
        return None
    if not expert_placement or not snapshot:
        return None

    matrix = build_coactivation_matrix(snapshot)
    results = compute_cross_gpu_coactivation(matrix, expert_placement, snapshot, topology)
    if not results:
        return None

    total_weighted = sum(r.cross_gpu_coactivation_rate * r.total_coactivation_count for r in results)
    total_count = sum(r.total_coactivation_count for r in results)
    total_rate = round(total_weighted / total_count, 4) if total_count else 0.0

    layers = [
        CoactivationLayerReport(
            layer_id=r.layer_id,
            cross_gpu_coactivation_rate=r.cross_gpu_coactivation_rate,
            estimated_extra_hops_per_pass=r.estimated_extra_hops_per_pass,
            total_coactivation_count=r.total_coactivation_count,
            top_misplaced_pairs=[
                CoactivationPairReport(
                    expert_a=p.expert_a,
                    expert_b=p.expert_b,
                    coactivation_count=p.coactivation_count,
                    cross_gpu=p.cross_gpu,
                )
                for p in r.top_misplaced_pairs
            ],
        )
        for r in results
    ]
    return CoactivationReport(layers=layers, total_cross_gpu_coactivation_rate=total_rate)


def _build_hetero_reports(
    hetero_topology,
    rec: object,
    snapshot: dict[tuple[int, int], int],
) -> tuple[HeterogeneousTopologyReport | None, HeterogeneousPlacementReport | None]:
    if hetero_topology is None:
        return None, None

    topo_report = HeterogeneousTopologyReport(
        gpus=[
            GpuCapabilityReport(
                index=g.index,
                name=g.name,
                memory_total_mib=g.memory_total_mib,
                memory_free_mib=g.memory_free_mib,
                compute_cap=g.compute_cap,
                max_sm_clock_mhz=g.max_sm_clock_mhz,
                max_mem_clock_mhz=g.max_mem_clock_mhz,
                relative_compute_score=g.relative_compute_score,
            )
            for g in hetero_topology.gpus
        ],
        is_homogeneous=hetero_topology.is_homogeneous,
        mixed_vendor=hetero_topology.mixed_vendor,
        compute_score_range=hetero_topology.compute_score_range,
    )

    placement_report = None
    if rec and getattr(rec, "method", "none") != "none":
        expert_placement = getattr(rec, "expert_placement", {})
        gpu_counts: dict[str, int] = {}
        for gpus in expert_placement.values():
            if gpus:
                key = str(gpus[0])
                gpu_counts[key] = gpu_counts.get(key, 0) + 1

        violations = hottest_experts_on_fastest_gpu(expert_placement, hetero_topology, snapshot)
        placement_report = HeterogeneousPlacementReport(
            gpu_expert_counts=gpu_counts,
            violations=[
                PlacementViolationReport(
                    layer_id=v.layer_id,
                    expert_id=v.expert_id,
                    assigned_gpu=v.assigned_gpu,
                    fastest_gpu=v.fastest_gpu,
                )
                for v in violations
            ],
        )

    return topo_report, placement_report


def _compute_comm_cost(
    rec: object,
    snapshot: dict[tuple[int, int], int],
    topology: Topology,
    pcie_topology: PCIeTopology | None,
    n_gpus: int,
) -> CommunicationCostReport | None:
    if not rec or getattr(rec, "method", "none") == "none":
        return None
    expert_placement = getattr(rec, "expert_placement", {})
    if not expert_placement or not snapshot:
        return None

    pcie = pcie_topology or PCIeTopology._flat_fallback(n_gpus)

    layers_seen = sorted({lid for (lid, _) in snapshot})
    experts_seen = sorted({eid for (_, eid) in snapshot})
    baseline: dict[tuple[int, int], list[int]] = {
        (lid, eid): [eid % n_gpus]
        for lid in layers_seen
        for eid in experts_seen
    }

    caveat_parts = ["current placement estimated as uniform round-robin (expert e -> GPU e % n_gpus)"]
    if pcie_topology is None:
        caveat_parts.append("PCIe topology not provided -- NVLink detection unavailable")

    result = compute_communication_cost(baseline, expert_placement, snapshot, topology, pcie)
    return CommunicationCostReport(
        current_overhead_us=result.current_overhead_us,
        recommended_overhead_us=result.recommended_overhead_us,
        delta_us=result.delta_us,
        caveat="; ".join(caveat_parts),
    )


def _hw_desc() -> str:
    try:
        import torch
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return f"{platform.node()} -- {', '.join(gpus) or 'CPU only'}"
    except Exception:
        return platform.node()
