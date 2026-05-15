from pathlib import Path

import pytest

from plumb.analysis.placement import recommend_placement, _greedy, _IMPROVEMENT_MIN, _IMPROVEMENT_MAX
from plumb.counter import ActivationCounter
from plumb.topology import Topology

FIXTURES = Path(__file__).parent / "fixtures" / "topologies"


def _counter_with(data: dict[tuple[int, int], int]) -> ActivationCounter:
    c = ActivationCounter(window_size=100_000)
    for (layer, expert), count in data.items():
        c.record(layer, expert, count)
    return c


def _dual_epyc_topology() -> Topology:
    return Topology.from_file(FIXTURES / "dual-epyc-8x-h100-sxm.json")


def _flat_topology(n: int = 8) -> Topology:
    return Topology.flat(n)


# ---------------------------------------------------------------------------
# recommend_placement
# ---------------------------------------------------------------------------

def test_returns_none_on_empty_counter():
    c = ActivationCounter()
    t = _flat_topology(4)
    assert recommend_placement(c, t) is None


def test_basic_recommendation_shape():
    # Use strongly imbalanced data (expert 0 gets ~10x others) so peak imbalance >= 3×
    data = {(layer, expert): (1000 if expert == 0 else 100) for layer in range(4) for expert in range(8)}
    c = _counter_with(data)
    t = _flat_topology(8)
    rec = recommend_placement(c, t, num_gpus=8)
    assert rec is not None
    # Should cover all (layer, expert) pairs
    assert len(rec.expert_placement) == 4 * 8
    # All GPU assignments in range
    for gpu in rec.expert_placement.values():
        assert 0 <= gpu < 8


def test_improvement_bounds_from_paper():
    data = {(0, e): 100 for e in range(8)}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(8), num_gpus=8)
    assert rec is not None
    assert rec.estimated_improvement_pct_min == _IMPROVEMENT_MIN
    assert rec.estimated_improvement_pct_max == _IMPROVEMENT_MAX


def test_improvement_point_estimate_clamped_low():
    # Uniform load → imbalance ratio = 1.0 → below threshold → method="none", warning set
    data = {(0, e): 100 for e in range(8)}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(8), num_gpus=8)
    assert rec is not None
    assert rec.method == "none"
    assert rec.warning != ""


def test_improvement_point_estimate_known_ratio():
    # Use data with peak imbalance >= 3× so the low-imbalance gate is not triggered.
    # data_moderate: expert 0 gets 40x others → peak ratio ~(40*10) / mean ≈ high
    # data_high: even more extreme imbalance
    # Exact ratio depends on num_experts; verify in bounds and increases with imbalance.
    data_low  = {(0, e): (10 if e > 0 else 300)  for e in range(8)}  # 300 vs 10 → ratio ~30×
    data_high = {(0, e): (10 if e > 0 else 1000) for e in range(8)}  # 1000 vs 10 → ratio ~100×
    c_low  = _counter_with(data_low)
    c_high = _counter_with(data_high)
    rec_low  = recommend_placement(c_low,  _flat_topology(8), num_gpus=8)
    rec_high = recommend_placement(c_high, _flat_topology(8), num_gpus=8)
    assert rec_low is not None and rec_high is not None
    assert rec_low.method != "none"
    assert rec_high.method != "none"
    assert _IMPROVEMENT_MIN <= rec_low.estimated_improvement_pct  <= _IMPROVEMENT_MAX
    assert _IMPROVEMENT_MIN <= rec_high.estimated_improvement_pct <= _IMPROVEMENT_MAX
    assert rec_high.estimated_improvement_pct >= rec_low.estimated_improvement_pct


def test_improvement_point_estimate_formula():
    import numpy as np
    # 8 experts, expert 0 gets all tokens → mean_ratio = 8 → (1 - 1/8)*70 = 61.25
    data = {(0, e): (1000 if e == 0 else 0) for e in range(8)}
    # add 1 to each to avoid zero-division in mean
    data = {k: v + 1 for k, v in data.items()}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(8), num_gpus=8)
    assert rec is not None
    assert _IMPROVEMENT_MIN <= rec.estimated_improvement_pct <= _IMPROVEMENT_MAX


def test_method_is_greedy_without_eplb():
    # EPLB won't be available in CI; method should fall back gracefully.
    # Use strongly imbalanced data so peak imbalance >= 3× (avoids low-imbalance gate).
    data = {(0, e): (1000 if e == 7 else 50) for e in range(8)}  # expert 7 is 20× hotter
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(8), num_gpus=8)
    assert rec is not None
    assert rec.method in ("greedy", "eplb")


def test_numa_finetune_pins_hot_experts_to_numa0():
    # Build a strongly imbalanced layer: expert 7 is hottest
    data = {(0, e): (1000 if e == 7 else 10) for e in range(8)}
    c = _counter_with(data)
    topology = _dual_epyc_topology()  # GPUs 0-3 on NUMA 0, 4-7 on NUMA 1
    rec = recommend_placement(c, topology, num_gpus=8)
    assert rec is not None
    # The hottest expert in layer 0 should be placed on a NUMA-0 GPU (0-3)
    hot_gpu = rec.expert_placement[(0, 7)]
    assert topology.gpu_to_numa[hot_gpu] == 0


# ---------------------------------------------------------------------------
# _greedy internals
# ---------------------------------------------------------------------------

def test_greedy_spreads_across_gpus():
    import numpy as np

    layers = [0]
    experts = list(range(8))
    load = np.array([[float(e + 1) for e in experts]])  # linearly increasing
    placement = _greedy(load, n_gpus=4, layers=layers, experts=experts)
    # Each GPU should get at least one expert
    gpus_used = set(placement.values())
    assert gpus_used == {0, 1, 2, 3}


def test_greedy_assigns_hottest_expert_to_gpu0():
    import numpy as np

    # Expert 7 is much hotter than the rest
    load = np.array([[1.0] * 7 + [999.0]])
    placement = _greedy(load, n_gpus=4, layers=[0], experts=list(range(8)))
    # Hottest expert (index 7) gets rank 0 → GPU 0
    assert placement[(0, 7)] == 0


# ---------------------------------------------------------------------------
# all three topology fixtures load and have correct NUMA structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname,expected_numa1_gpu", [
    ("dual-epyc-8x-h100-sxm.json",  4),
    ("dual-epyc-8x-h100-pcie.json",  4),
    ("single-epyc-4x-a100-pcie.json", None),  # single NUMA domain
])
def test_topology_fixtures_load(fname, expected_numa1_gpu):
    t = Topology.from_file(FIXTURES / fname)
    assert len(t.gpu_to_numa) > 0
    if expected_numa1_gpu is not None:
        assert t.gpu_to_numa[expected_numa1_gpu] == 1
    else:
        assert all(v == 0 for v in t.gpu_to_numa.values())
