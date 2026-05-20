"""Expert allocation across heterogeneous GPU topologies.

compute_expert_allocation() partitions experts by compute score with VRAM caps.
hottest_experts_on_fastest_gpu() checks placement compliance.
detect_vram_pressure() identifies GPUs below a free-VRAM threshold.
plan_vram_replan() migrates cold experts off pressured GPUs to GPUs with headroom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from numa_topology.gpu_capabilities import GPUCapability, HeterogeneousTopology

_MIB_PER_GB = 1024.0


@dataclass
class PlacementViolation:
    layer_id: int
    expert_id: int
    assigned_gpu: int
    fastest_gpu: int


def compute_expert_allocation(
    num_experts: int,
    gpus: list[GPUCapability],
    expert_size_gb: float,
) -> dict[int, int]:
    """Allocate experts across GPUs weighted by relative_compute_score.

    Algorithm:
      1. Proportional target = num_experts * score_i / sum(scores)
      2. Floor + largest-remainder rounding so allocations sum exactly to num_experts
      3. Cap each GPU at floor(memory_free_mib / expert_size_mib)
      4. Iteratively redistribute overflow to unconstrained GPUs (same rounding)

    Args:
        num_experts:    Total number of experts to place.
        gpus:           GPUCapability list from HeterogeneousTopology.
        expert_size_gb: Memory footprint of one expert replica in GB.

    Returns:
        Mapping of GPU index → expert count. Guaranteed to sum to num_experts.

    Raises:
        ValueError: total available VRAM cannot fit num_experts.
    """
    if not gpus:
        raise ValueError("gpus list is empty")

    expert_size_mib = expert_size_gb * _MIB_PER_GB
    caps: dict[int, int] = {
        g.index: int(g.memory_free_mib / expert_size_mib)
        for g in gpus
    }

    total_cap = sum(caps.values())
    if total_cap < num_experts:
        needed_gb = num_experts * expert_size_gb
        avail_gb  = total_cap * expert_size_gb
        raise ValueError(
            f"Insufficient VRAM: need {needed_gb:.1f} GB for {num_experts} experts "
            f"({expert_size_gb:.1f} GB each) but only {avail_gb:.1f} GB available"
        )

    scores: dict[int, float] = {g.index: g.relative_compute_score for g in gpus}
    alloc = _proportional_round(num_experts, scores)
    alloc = _apply_caps(alloc, caps, scores)
    return alloc


def hottest_experts_on_fastest_gpu(
    expert_placement: dict[tuple[int, int], list[int]],
    topology: HeterogeneousTopology,
    expert_loads: dict[tuple[int, int], int],
) -> list[PlacementViolation]:
    """Return violations where a layer's hottest expert isn't on the fastest GPU.

    Args:
        expert_placement: (layer_id, expert_id) → [gpu_id, ...] (primary = [0])
        topology:         HeterogeneousTopology from discover_gpu_capabilities()
        expert_loads:     (layer_id, expert_id) → token_count snapshot

    Returns:
        List of PlacementViolation, one per layer where the hottest expert's
        primary GPU is not the highest-score GPU in the topology.
        Empty list when topology is homogeneous or placement is compliant.
    """
    if not topology.gpus:
        return []

    score_by_gpu: dict[int, float] = {g.index: g.relative_compute_score for g in topology.gpus}
    fastest_gpu = max(score_by_gpu, key=score_by_gpu.__getitem__)

    # Nothing to violate in a homogeneous cluster (all scores equal)
    if topology.is_homogeneous:
        return []

    # Group loads by layer
    layers: dict[int, dict[int, int]] = {}
    for (lid, eid), count in expert_loads.items():
        layers.setdefault(lid, {})[eid] = count

    violations: list[PlacementViolation] = []
    for lid, layer_loads in sorted(layers.items()):
        if not layer_loads:
            continue
        hottest_eid = max(layer_loads, key=layer_loads.__getitem__)
        gpus = expert_placement.get((lid, hottest_eid))
        if not gpus:
            continue
        primary = gpus[0]
        if primary != fastest_gpu:
            violations.append(PlacementViolation(
                layer_id=lid,
                expert_id=hottest_eid,
                assigned_gpu=primary,
                fastest_gpu=fastest_gpu,
            ))

    return violations


# ---------------------------------------------------------------------------
# VRAM-pressure replanning
# ---------------------------------------------------------------------------

@dataclass
class VramPressureEvent:
    gpu_index: int
    free_mib: int
    total_mib: int
    free_pct: float   # 0.0–100.0


@dataclass
class ExpertMigration:
    layer_id: int
    expert_id: int
    from_gpu: int
    to_gpu: int
    activation_count: int   # lower = colder


@dataclass
class VramReplanResult:
    pressured_gpus: list[VramPressureEvent]
    migrations: list[ExpertMigration] = field(default_factory=list)
    updated_placement: dict[tuple[int, int], list[int]] = field(default_factory=dict)


def detect_vram_pressure(
    gpus: list[GPUCapability],
    free_pct_threshold: float = 10.0,
) -> list[VramPressureEvent]:
    """Return GPUs where free VRAM % is below *free_pct_threshold*.

    Args:
        gpus:               GPU capability list with memory_free_mib / memory_total_mib set.
        free_pct_threshold: Trigger level in %; default 10.0 means <10 % free = pressure.

    Returns:
        One VramPressureEvent per pressured GPU.  Empty when all GPUs have
        sufficient headroom or VRAM data is unavailable (total_mib == 0).
    """
    events: list[VramPressureEvent] = []
    for gpu in gpus:
        if gpu.memory_total_mib <= 0:
            continue
        free_pct = 100.0 * gpu.memory_free_mib / gpu.memory_total_mib
        if free_pct < free_pct_threshold:
            events.append(VramPressureEvent(
                gpu_index=gpu.index,
                free_mib=gpu.memory_free_mib,
                total_mib=gpu.memory_total_mib,
                free_pct=round(free_pct, 2),
            ))
    return events


def plan_vram_replan(
    pressured_gpus: list[VramPressureEvent],
    expert_placement: dict[tuple[int, int], list[int]],
    expert_loads: dict[tuple[int, int], int],
    gpus: list[GPUCapability],
    expert_size_gb: float,
    free_pct_threshold: float = 10.0,
) -> VramReplanResult:
    """Plan cold-expert migrations away from VRAM-pressured GPUs.

    For each pressured GPU, identifies the coldest experts (lowest activation
    count) and migrates them to the GPU with the most remaining headroom,
    provided it can hold the expert without OOM.  Each planned migration
    reduces the target GPU's tracked free VRAM by *expert_size_gb* so
    subsequent decisions in the same plan respect the updated capacity.

    Args:
        pressured_gpus:    Output of detect_vram_pressure().
        expert_placement:  (layer_id, expert_id) → [gpu_id, ...]; primary = index 0.
        expert_loads:      (layer_id, expert_id) → token_count snapshot.
        gpus:              Full GPU list including non-pressured GPUs.
        expert_size_gb:    Estimated VRAM footprint per expert in GB.
        free_pct_threshold: Unused in planning logic; kept for API symmetry.

    Returns:
        VramReplanResult with the migration list and an updated_placement copy
        reflecting the new primary GPU for each migrated expert.
    """
    if not pressured_gpus:
        return VramReplanResult(
            pressured_gpus=[],
            migrations=[],
            updated_placement={k: list(v) for k, v in expert_placement.items()},
        )

    expert_size_mib = expert_size_gb * _MIB_PER_GB
    pressured_set = {e.gpu_index for e in pressured_gpus}

    # Track remaining free VRAM per GPU for planning; decremented as migrations are assigned.
    remaining_mib: dict[int, float] = {g.index: float(g.memory_free_mib) for g in gpus}

    # Candidates: experts whose primary GPU is under pressure, sorted coldest first.
    candidates: list[tuple[int, int, int]] = []  # (activation_count, layer_id, expert_id)
    for (lid, eid), gpu_list in expert_placement.items():
        if gpu_list and gpu_list[0] in pressured_set:
            candidates.append((expert_loads.get((lid, eid), 0), lid, eid))
    candidates.sort()  # ascending by activation_count → coldest first

    updated = {k: list(v) for k, v in expert_placement.items()}
    migrations: list[ExpertMigration] = []

    for load, lid, eid in candidates:
        from_gpu = updated[(lid, eid)][0]
        target = _best_migration_target(remaining_mib, pressured_set, expert_size_mib)
        if target is None:
            continue  # no GPU has enough headroom — skip rather than risk OOM
        updated[(lid, eid)] = [target] + updated[(lid, eid)][1:]
        remaining_mib[target] -= expert_size_mib
        migrations.append(ExpertMigration(
            layer_id=lid,
            expert_id=eid,
            from_gpu=from_gpu,
            to_gpu=target,
            activation_count=load,
        ))

    return VramReplanResult(
        pressured_gpus=pressured_gpus,
        migrations=migrations,
        updated_placement=updated,
    )


def _best_migration_target(
    remaining_mib: dict[int, float],
    pressured_set: set[int],
    expert_size_mib: float,
) -> int | None:
    """Return the non-pressured GPU with the most free VRAM that fits one expert.

    Returns None when no safe target exists (all candidates exhausted or full).
    """
    candidates = [
        (rem, idx)
        for idx, rem in remaining_mib.items()
        if idx not in pressured_set and rem >= expert_size_mib
    ]
    return max(candidates)[1] if candidates else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proportional_round(total: int, scores: dict[int, float]) -> dict[int, int]:
    """Largest-remainder allocation: floor + give remainders to highest fractions."""
    total_score = sum(scores.values()) or 1.0
    raw = {idx: total * s / total_score for idx, s in scores.items()}
    alloc = {idx: int(v) for idx, v in raw.items()}
    remainder = total - sum(alloc.values())
    fracs = sorted(((raw[idx] - alloc[idx], idx) for idx in alloc), reverse=True)
    for i in range(remainder):
        alloc[fracs[i][1]] += 1
    return alloc


def _apply_caps(
    alloc: dict[int, int],
    caps: dict[int, int],
    scores: dict[int, float],
) -> dict[int, int]:
    """Iteratively clamp allocations to VRAM caps, redistributing overflow."""
    alloc = dict(alloc)
    constrained: set[int] = set()

    while True:
        overflow = 0
        for idx in list(alloc):
            if alloc[idx] > caps[idx]:
                overflow += alloc[idx] - caps[idx]
                alloc[idx] = caps[idx]
                constrained.add(idx)

        if overflow == 0:
            break

        free = {idx: s for idx, s in scores.items()
                if idx not in constrained and alloc[idx] < caps[idx]}
        if not free:
            raise ValueError("Cannot redistribute overflow: all GPUs at VRAM capacity")

        extra = _proportional_round(overflow, free)
        for idx, add in extra.items():
            alloc[idx] += add

    return alloc
