"""Tests for analysis/heterogeneous.py — no GPU hardware required."""
from __future__ import annotations

import pytest

from plumb.analysis.heterogeneous import (
    ExpertMigration,
    PlacementViolation,
    VramPressureEvent,
    VramReplanResult,
    compute_expert_allocation,
    detect_vram_pressure,
    hottest_experts_on_fastest_gpu,
    plan_vram_replan,
)
from numa_topology.gpu_capabilities import (
    GPUCapability,
    HeterogeneousTopology,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu(index: int, score: float, mem_free_mib: int = 40_000) -> GPUCapability:
    return GPUCapability(
        index=index, name=f"GPU-{index}",
        memory_total_mib=mem_free_mib, memory_free_mib=mem_free_mib,
        compute_cap="8.0", max_sm_clock_mhz=1000, max_mem_clock_mhz=1000,
        relative_compute_score=score,
    )


def _topo(gpus: list[GPUCapability], homogeneous: bool = False) -> HeterogeneousTopology:
    scores = [g.relative_compute_score for g in gpus]
    return HeterogeneousTopology(
        gpus=gpus,
        is_homogeneous=homogeneous,
        mixed_vendor=False,
        compute_score_range=(min(scores), max(scores)),
    )


# ---------------------------------------------------------------------------
# compute_expert_allocation — partition correctness
# ---------------------------------------------------------------------------

def test_partition_sums_to_num_experts():
    # 80 GB free per GPU → cap = 80 experts at 1 GB each; 4×80 = 320 ≥ 256
    gpus = [_gpu(i, 1.0, mem_free_mib=80 * 1024) for i in range(4)]
    alloc = compute_expert_allocation(256, gpus, expert_size_gb=1.0)
    assert sum(alloc.values()) == 256


def test_partition_sums_odd_total():
    gpus = [_gpu(0, 1.0), _gpu(1, 1.0), _gpu(2, 1.0)]
    alloc = compute_expert_allocation(10, gpus, expert_size_gb=1.0)
    assert sum(alloc.values()) == 10


def test_proportional_by_score():
    # GPU 0 has 2× score → should get ~2× experts
    gpus = [_gpu(0, 1.0), _gpu(1, 0.5)]
    alloc = compute_expert_allocation(9, gpus, expert_size_gb=1.0)
    assert sum(alloc.values()) == 9
    assert alloc[0] > alloc[1]


def test_homogeneous_even_split():
    gpus = [_gpu(i, 1.0) for i in range(4)]
    alloc = compute_expert_allocation(8, gpus, expert_size_gb=1.0)
    assert all(v == 2 for v in alloc.values())


def test_three_gpu_heterogeneous_weighted():
    # GPU 0: score 1.0 (fastest), GPU 1: score 0.5, GPU 2: score 0.25
    # proportional targets for 14 experts: 8, 4, 2 (sum = 14)
    gpus = [_gpu(0, 1.0), _gpu(1, 0.5), _gpu(2, 0.25)]
    alloc = compute_expert_allocation(14, gpus, expert_size_gb=1.0)
    assert sum(alloc.values()) == 14
    # Fastest GPU gets the most; slowest gets the least
    assert alloc[0] > alloc[1] > alloc[2]
    assert alloc[0] == 8
    assert alloc[1] == 4
    assert alloc[2] == 2


# ---------------------------------------------------------------------------
# compute_expert_allocation — VRAM cap
# ---------------------------------------------------------------------------

def test_vram_cap_binds():
    # expert_size_gb=10 → cap = floor(40000 / (10*1024)) = floor(40000/10240) = 3
    gpus = [_gpu(0, 1.0, mem_free_mib=40_000), _gpu(1, 1.0, mem_free_mib=40_000)]
    alloc = compute_expert_allocation(4, gpus, expert_size_gb=10.0)
    cap = 40_000 // (10 * 1024)  # = 3
    assert alloc[0] <= cap
    assert alloc[1] <= cap
    assert sum(alloc.values()) == 4


def test_overflow_redistributes_to_unconstrained():
    # GPU 0: tiny VRAM → cap 1; GPU 1: large VRAM → absorbs overflow
    gpus = [
        _gpu(0, 1.0, mem_free_mib=1_500),   # cap = floor(1500/1024) = 1
        _gpu(1, 1.0, mem_free_mib=40_000),  # cap = 39
    ]
    alloc = compute_expert_allocation(4, gpus, expert_size_gb=1.0)
    assert alloc[0] <= 1
    assert alloc[1] >= 3
    assert sum(alloc.values()) == 4


def test_insufficient_vram_raises():
    # 1 GB free, expert 2 GB → cap = 0
    gpus = [_gpu(0, 1.0, mem_free_mib=1_000)]
    with pytest.raises(ValueError, match="Insufficient VRAM"):
        compute_expert_allocation(1, gpus, expert_size_gb=2.0)


def test_insufficient_vram_message_informative():
    gpus = [_gpu(0, 1.0, mem_free_mib=2_000), _gpu(1, 1.0, mem_free_mib=2_000)]
    with pytest.raises(ValueError) as exc_info:
        compute_expert_allocation(10, gpus, expert_size_gb=5.0)
    msg = str(exc_info.value)
    assert "GB" in msg
    assert "10" in msg


def test_empty_gpu_list_raises():
    with pytest.raises(ValueError):
        compute_expert_allocation(8, [], expert_size_gb=1.0)


# ---------------------------------------------------------------------------
# hottest_experts_on_fastest_gpu
# ---------------------------------------------------------------------------

def _hetero_topo() -> HeterogeneousTopology:
    return _topo([_gpu(0, 1.0), _gpu(1, 0.5)], homogeneous=False)


def test_no_violations_when_hottest_on_fastest():
    topo = _hetero_topo()  # GPU 0 is fastest
    placement = {(0, 0): [0], (0, 1): [1]}
    loads = {(0, 0): 1000, (0, 1): 100}
    assert hottest_experts_on_fastest_gpu(placement, topo, loads) == []


def test_violation_when_hottest_on_slow_gpu():
    topo = _hetero_topo()  # GPU 0 fastest
    placement = {(0, 0): [1], (0, 1): [0]}  # hottest expert 0 on slow GPU 1
    loads = {(0, 0): 1000, (0, 1): 100}
    violations = hottest_experts_on_fastest_gpu(placement, topo, loads)
    assert len(violations) == 1
    v = violations[0]
    assert v.layer_id == 0
    assert v.expert_id == 0
    assert v.assigned_gpu == 1
    assert v.fastest_gpu == 0


def test_no_violations_on_homogeneous_topology():
    topo = _topo([_gpu(0, 1.0), _gpu(1, 1.0)], homogeneous=True)
    placement = {(0, 0): [1]}
    loads = {(0, 0): 500}
    assert hottest_experts_on_fastest_gpu(placement, topo, loads) == []


def test_violations_across_multiple_layers():
    topo = _hetero_topo()
    placement = {
        (0, 0): [1], (0, 1): [0],   # layer 0: hottest (0) on slow GPU → violation
        (1, 0): [0], (1, 1): [1],   # layer 1: hottest (0) on fast GPU → ok
    }
    loads = {(0, 0): 1000, (0, 1): 100, (1, 0): 900, (1, 1): 200}
    violations = hottest_experts_on_fastest_gpu(placement, topo, loads)
    assert len(violations) == 1
    assert violations[0].layer_id == 0


def test_empty_placement_no_violations():
    topo = _hetero_topo()
    assert hottest_experts_on_fastest_gpu({}, topo, {}) == []


# ---------------------------------------------------------------------------
# detect_vram_pressure
# ---------------------------------------------------------------------------

def _gpu_with_vram(index: int, free_mib: int, total_mib: int) -> GPUCapability:
    return GPUCapability(
        index=index, name=f"GPU-{index}",
        memory_total_mib=total_mib, memory_free_mib=free_mib,
        compute_cap="8.0", max_sm_clock_mhz=1000, max_mem_clock_mhz=1000,
        relative_compute_score=1.0,
    )


def test_pressure_fires_below_threshold():
    # 900 / 10000 = 9% free → below default threshold of 10%
    gpus = [_gpu_with_vram(0, free_mib=900, total_mib=10_000)]
    events = detect_vram_pressure(gpus)
    assert len(events) == 1
    assert events[0].gpu_index == 0
    assert events[0].free_pct == pytest.approx(9.0, abs=0.1)


def test_pressure_no_event_above_threshold():
    # 1100 / 10000 = 11% free → above threshold
    gpus = [_gpu_with_vram(0, free_mib=1_100, total_mib=10_000)]
    assert detect_vram_pressure(gpus) == []


def test_pressure_boundary_not_triggered():
    # Exactly 10% free → NOT triggered (condition is strict <)
    gpus = [_gpu_with_vram(0, free_mib=1_000, total_mib=10_000)]
    assert detect_vram_pressure(gpus, free_pct_threshold=10.0) == []


def test_pressure_custom_threshold():
    # 15% free; fires at threshold=20, not at threshold=10
    gpus = [_gpu_with_vram(0, free_mib=1_500, total_mib=10_000)]
    assert detect_vram_pressure(gpus, free_pct_threshold=10.0) == []
    assert len(detect_vram_pressure(gpus, free_pct_threshold=20.0)) == 1


def test_pressure_skips_zero_total_mib():
    # GPUs with total_mib=0 carry no VRAM data — should not raise or trigger
    gpus = [_gpu_with_vram(0, free_mib=0, total_mib=0)]
    assert detect_vram_pressure(gpus) == []


def test_pressure_multiple_gpus_only_pressured_returned():
    gpus = [
        _gpu_with_vram(0, free_mib=500,   total_mib=10_000),  # 5% → pressured
        _gpu_with_vram(1, free_mib=5_000, total_mib=10_000),  # 50% → ok
        _gpu_with_vram(2, free_mib=800,   total_mib=10_000),  # 8% → pressured
    ]
    events = detect_vram_pressure(gpus)
    assert len(events) == 2
    assert {e.gpu_index for e in events} == {0, 2}


# ---------------------------------------------------------------------------
# plan_vram_replan
# ---------------------------------------------------------------------------

def _placement(*items: tuple[tuple[int, int], int]) -> dict[tuple[int, int], list[int]]:
    """Build a placement dict: ((layer, expert), gpu) → {(layer, expert): [gpu]}."""
    return {k: [v] for k, v in items}


def test_replan_empty_pressured_returns_unchanged_placement():
    placement = _placement(((0, 0), 0), ((0, 1), 1))
    loads = {(0, 0): 100, (0, 1): 200}
    gpus = [_gpu(0, 1.0, 40_000), _gpu(1, 1.0, 40_000)]
    result = plan_vram_replan([], placement, loads, gpus, expert_size_gb=1.0)
    assert result.migrations == []
    assert result.updated_placement == {(0, 0): [0], (0, 1): [1]}


def test_replan_cold_expert_migrated_off_pressured_gpu():
    # GPU 0: pressured. GPU 1: healthy with headroom.
    # Two experts on GPU 0; only one on GPU 1.
    pressured = [VramPressureEvent(gpu_index=0, free_mib=500, total_mib=10_000, free_pct=5.0)]
    placement = _placement(((0, 0), 0), ((0, 1), 0))
    loads = {(0, 0): 10, (0, 1): 1000}   # expert 0 = cold, expert 1 = hot
    gpus = [
        _gpu_with_vram(0, free_mib=500,    total_mib=10_000),
        _gpu_with_vram(1, free_mib=30_000, total_mib=40_000),
    ]
    result = plan_vram_replan(pressured, placement, loads, gpus, expert_size_gb=1.0)
    # At least the cold expert should be migrated
    migrated_experts = {(m.layer_id, m.expert_id) for m in result.migrations}
    assert (0, 0) in migrated_experts


def test_replan_cold_before_hot():
    # One target slot available — coldest expert should get it.
    pressured = [VramPressureEvent(gpu_index=0, free_mib=500, total_mib=10_000, free_pct=5.0)]
    placement = _placement(((0, 0), 0), ((0, 1), 0))
    loads = {(0, 0): 10, (0, 1): 5000}   # expert 0 cold, expert 1 hot
    gpus = [
        _gpu_with_vram(0, free_mib=500,   total_mib=10_000),
        # Target fits exactly one expert (1024 MiB for 1 GB expert)
        _gpu_with_vram(1, free_mib=1_024, total_mib=40_000),
    ]
    result = plan_vram_replan(pressured, placement, loads, gpus, expert_size_gb=1.0)
    assert len(result.migrations) == 1
    m = result.migrations[0]
    assert m.expert_id == 0       # coldest migrated
    assert m.activation_count == 10


def test_replan_no_oom_target_must_fit_expert():
    # Target GPU has only 512 MiB free; expert needs 1 GB (1024 MiB) → no migration.
    pressured = [VramPressureEvent(gpu_index=0, free_mib=200, total_mib=10_000, free_pct=2.0)]
    placement = _placement(((0, 0), 0))
    loads = {(0, 0): 5}
    gpus = [
        _gpu_with_vram(0, free_mib=200, total_mib=10_000),
        _gpu_with_vram(1, free_mib=512, total_mib=40_000),  # insufficient
    ]
    result = plan_vram_replan(pressured, placement, loads, gpus, expert_size_gb=1.0)
    assert result.migrations == []


def test_replan_no_safe_target_all_full():
    pressured = [VramPressureEvent(gpu_index=0, free_mib=100, total_mib=10_000, free_pct=1.0)]
    placement = _placement(((0, 0), 0), ((0, 1), 0))
    loads = {(0, 0): 1, (0, 1): 2}
    gpus = [
        _gpu_with_vram(0, free_mib=100,   total_mib=10_000),
        _gpu_with_vram(1, free_mib=100,   total_mib=40_000),  # also too small
    ]
    result = plan_vram_replan(pressured, placement, loads, gpus, expert_size_gb=1.0)
    assert result.migrations == []


def test_replan_updated_placement_reflects_migration():
    pressured = [VramPressureEvent(gpu_index=0, free_mib=500, total_mib=10_000, free_pct=5.0)]
    placement = _placement(((0, 0), 0))
    loads = {(0, 0): 5}
    gpus = [
        _gpu_with_vram(0, free_mib=500,    total_mib=10_000),
        _gpu_with_vram(1, free_mib=30_000, total_mib=40_000),
    ]
    result = plan_vram_replan(pressured, placement, loads, gpus, expert_size_gb=1.0)
    assert len(result.migrations) == 1
    assert result.updated_placement[(0, 0)][0] == 1   # moved to GPU 1


def test_replan_target_vram_decremented_across_migrations():
    # Target GPU has room for exactly 2 experts (2 × 1024 MiB ≤ 2048 MiB).
    # Three experts to migrate → only 2 should succeed.
    pressured = [VramPressureEvent(gpu_index=0, free_mib=200, total_mib=10_000, free_pct=2.0)]
    placement = _placement(((0, 0), 0), ((0, 1), 0), ((0, 2), 0))
    loads = {(0, 0): 1, (0, 1): 2, (0, 2): 3}
    gpus = [
        _gpu_with_vram(0, free_mib=200,   total_mib=10_000),
        _gpu_with_vram(1, free_mib=2_048, total_mib=40_000),  # fits exactly 2
    ]
    result = plan_vram_replan(pressured, placement, loads, gpus, expert_size_gb=1.0)
    assert len(result.migrations) == 2
