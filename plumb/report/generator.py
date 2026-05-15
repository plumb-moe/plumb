from __future__ import annotations

import platform
from datetime import datetime, timezone

from ..analysis.imbalance import compute_imbalance
from ..analysis.numa import compute_cross_numa
from ..analysis.placement import recommend_placement
from ..counter import ActivationCounter
from ..topology import Topology
from .schema import ExpertLoad, LayerReport, PlacementReport, ProfileReport


def generate_report(
    counter: ActivationCounter,
    topology: Topology,
    model_name: str,
    duration_seconds: float,
    num_gpus: int | None = None,
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
                f"{lid}:{eid}": gpu
                for (lid, eid), gpu in rec.expert_placement.items()
            },
            estimated_improvement_pct_min=rec.estimated_improvement_pct_min,
            estimated_improvement_pct_max=rec.estimated_improvement_pct_max,
            estimated_improvement_pct=rec.estimated_improvement_pct,
        )

    gpu_to_numa = dict(topology.gpu_to_numa) if topology.gpu_to_numa else None
    # Treat flat topologies (all same NUMA node) as None — bar chart uses single colour
    if gpu_to_numa and len(set(gpu_to_numa.values())) <= 1:
        gpu_to_numa = None

    return ProfileReport(
        model_name=model_name,
        hardware_config=_hw_desc(),
        profiling_duration_seconds=round(duration_seconds, 2),
        total_forward_passes=counter.pass_count,
        layers=layers,
        placement=placement_report,
        gpu_to_numa=gpu_to_numa,
        generated_at=datetime.now(tz=timezone.utc),
    )


def generate_report_from_snapshot(snapshot: dict) -> ProfileReport:
    """Reconstruct a ProfileReport from a live session snapshot dict.

    The snapshot must contain 'expert_counts' (written by autoattach since 0.1.0).
    """
    expert_counts_raw: dict[str, int] = snapshot.get("expert_counts", {})
    if not expert_counts_raw:
        raise ValueError("snapshot missing 'expert_counts' — requires plumb ≥0.1.0 auto-attach")

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

    return generate_report(
        counter=counter,
        topology=topology,
        model_name=snapshot.get("model_name", "unknown"),
        duration_seconds=duration,
        num_gpus=len(gpu_to_numa) or None,
    )


def _hw_desc() -> str:
    try:
        import torch
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return f"{platform.node()} — {', '.join(gpus) or 'CPU only'}"
    except Exception:
        return platform.node()
