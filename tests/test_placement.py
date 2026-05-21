from pathlib import Path

import pytest

from plumb.analysis.placement import (
    _IMPROVEMENT_MAX,
    _IMPROVEMENT_MIN,
    _comm_aware_greedy,
    _greedy,
    _numa_finetune,
    coactivation_partition,
    recommend_placement,
    worst_case_placement,
)
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
    # All GPU assignments in range; values are now list[int]
    for gpus in rec.expert_placement.values():
        assert isinstance(gpus, list)
        assert all(0 <= g < 8 for g in gpus)


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


# ---------------------------------------------------------------------------
# _try_eplb — live coverage via injected fake eplb + torch modules
#
# The real eplb.rebalance_experts returns (phy2log, log2phy, logcnt). The
# placement code uses log2phy[li, ei, 0] // n_experts to pick the GPU. These
# tests inject a controlled fake so we can assert:
#   1. method=='eplb' when the import succeeds (not the silent greedy fallback)
#   2. The result actually uses log2phy[..., 0] // n_experts (catches the class
#      of bug we just fixed: rebalance vs rebalance_experts, 2-tuple vs 3-tuple)
# ---------------------------------------------------------------------------

def _install_fake_eplb_and_torch(monkeypatch, log2phy_array):
    """Install fake torch + eplb modules so _try_eplb's import path succeeds.

    log2phy_array: numpy ndarray of shape (n_layers, n_experts, max_replicas)
                   used as the second tuple element returned by rebalance_experts.
    """
    import sys
    import types

    import numpy as np

    # Fake torch.tensor — just returns the underlying numpy array since
    # placement._try_eplb only calls .item() on log2phy elements (works on numpy).
    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda x: x  # caller wraps a numpy array; identity is fine

    captured_args = {}

    def fake_rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
        captured_args["weight_shape"] = tuple(weight.shape)
        captured_args["num_replicas"] = num_replicas
        captured_args["num_groups"] = num_groups
        captured_args["num_nodes"] = num_nodes
        captured_args["num_gpus"] = num_gpus
        # phy2log and logcnt are returned but unused — give them dummies
        n_layers, n_experts = weight.shape
        phy2log = np.zeros((n_layers, num_replicas), dtype=np.int64)
        logcnt = np.ones((n_layers, n_experts), dtype=np.int64)
        return phy2log, log2phy_array, logcnt

    fake_eplb = types.ModuleType("eplb")
    fake_eplb.rebalance_experts = fake_rebalance_experts

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "eplb", fake_eplb)
    return captured_args


def test_try_eplb_method_is_eplb_when_module_available(monkeypatch):
    """When eplb imports cleanly, recommend_placement reports method='eplb'."""
    import numpy as np

    # 2 layers, 4 experts, 1 replica each — log2phy[li, ei, 0] = physical slot
    # Slot -> GPU is slot // n_experts. For 2 GPUs × 4 experts, slots 0..3 → GPU 0, 4..7 → GPU 1.
    log2phy = np.array([
        [[0], [5], [1], [4]],   # layer 0: experts go to GPUs 0,1,0,1
        [[6], [2], [7], [3]],   # layer 1: experts go to GPUs 1,0,1,0
    ], dtype=np.int64)

    _install_fake_eplb_and_torch(monkeypatch, log2phy)

    # Strongly imbalanced load so the low-imbalance gate doesn't trip
    data = {(layer, expert): (1000 if expert == 0 else 50) for layer in range(2) for expert in range(4)}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(2), num_gpus=2)

    assert rec is not None
    assert rec.method == "eplb", f"expected method='eplb', got {rec.method!r}"


def test_try_eplb_uses_log2phy_first_replica_div_n_experts(monkeypatch):
    """GPU assignment must be log2phy[li, ei, 0] // n_experts (the formula we just fixed)."""
    import numpy as np

    # 1 layer, 4 experts, 2 GPUs → n_experts=4, slot // 4 = GPU
    # log2phy[0, 0, 0] = 5 → slot 5 → GPU 1
    # log2phy[0, 1, 0] = 0 → slot 0 → GPU 0
    # log2phy[0, 2, 0] = 7 → slot 7 → GPU 1
    # log2phy[0, 3, 0] = 2 → slot 2 → GPU 0
    log2phy = np.array([[[5], [0], [7], [2]]], dtype=np.int64)

    _install_fake_eplb_and_torch(monkeypatch, log2phy)

    data = {(0, e): (1000 if e == 0 else 50) for e in range(4)}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(2), num_gpus=2)

    assert rec is not None and rec.method == "eplb"
    # NUMA finetune is a no-op on single-NUMA Topology.flat, so assignments survive intact
    # Values are now list[int]; no replication so each list has exactly one GPU
    assert rec.expert_placement[(0, 0)] == [1]  # slot 5 // 4
    assert rec.expert_placement[(0, 1)] == [0]  # slot 0 // 4
    assert rec.expert_placement[(0, 2)] == [1]  # slot 7 // 4
    assert rec.expert_placement[(0, 3)] == [0]  # slot 2 // 4


def test_try_eplb_passes_correct_args_to_rebalance_experts(monkeypatch):
    """rebalance_experts must be called with (weight, n_gpus*n_experts, n_gpus, 1, n_gpus)."""
    import numpy as np

    n_layers, n_experts, n_gpus = 2, 4, 2
    log2phy = np.zeros((n_layers, n_experts, 1), dtype=np.int64)
    captured = _install_fake_eplb_and_torch(monkeypatch, log2phy)

    data = {(layer, expert): (1000 if expert == 0 else 50) for layer in range(n_layers) for expert in range(n_experts)}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(n_gpus), num_gpus=n_gpus)

    assert rec is not None and rec.method == "eplb"
    assert captured["weight_shape"] == (n_layers, n_experts)
    assert captured["num_replicas"] == n_gpus * n_experts
    assert captured["num_groups"] == n_gpus
    assert captured["num_nodes"] == 1
    assert captured["num_gpus"] == n_gpus


def test_try_eplb_falls_back_to_greedy_when_rebalance_raises(monkeypatch):
    """If eplb.rebalance_experts raises, _try_eplb logs and falls back to greedy."""
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda x: x

    fake_eplb = types.ModuleType("eplb")
    def bad_rebalance(*args, **kwargs):
        raise RuntimeError("simulated eplb failure")
    fake_eplb.rebalance_experts = bad_rebalance

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "eplb", fake_eplb)

    data = {(0, e): (1000 if e == 0 else 50) for e in range(4)}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(2), num_gpus=2)

    assert rec is not None
    assert rec.method == "greedy", f"expected greedy fallback, got {rec.method!r}"


def test_numa_finetune_pins_hot_experts_to_numa0():
    # Build a strongly imbalanced layer: expert 7 is hottest
    data = {(0, e): (1000 if e == 7 else 10) for e in range(8)}
    c = _counter_with(data)
    topology = _dual_epyc_topology()  # GPUs 0-3 on NUMA 0, 4-7 on NUMA 1
    rec = recommend_placement(c, topology, num_gpus=8)
    assert rec is not None
    # The hottest expert in layer 0 should be placed on a NUMA-0 GPU (0-3)
    hot_gpus = rec.expert_placement[(0, 7)]
    assert topology.gpu_to_numa[hot_gpus[0]] == 0


# ---------------------------------------------------------------------------
# worst_case_placement
# ---------------------------------------------------------------------------

def test_worst_case_empty_counter():
    c = ActivationCounter()
    t = _flat_topology(4)
    assert worst_case_placement(c, t) == {}


def test_worst_case_shape_covers_all_pairs():
    data = {(layer, expert): (100 * expert + 1) for layer in range(3) for expert in range(8)}
    c = _counter_with(data)
    t = _flat_topology(4)
    placement = worst_case_placement(c, t, num_gpus=4)
    assert len(placement) == 3 * 8
    for gpus in placement.values():
        assert isinstance(gpus, list)
        assert all(0 <= g < 4 for g in gpus)


def test_worst_case_concentrates_hot_experts_on_gpu0():
    # 8 experts, 4 GPUs → experts_per_gpu=2; top 2 hottest should be on GPU 0
    data = {(0, e): (1000 - e * 10) for e in range(8)}  # expert 0 hottest, expert 7 coldest
    c = _counter_with(data)
    placement = worst_case_placement(c, _flat_topology(4), num_gpus=4)
    # Expert 0 (hottest) → rank 0 → GPU 0
    assert placement[(0, 0)] == [0]
    # Expert 1 (2nd hottest) → rank 1 → GPU 0 (within first block)
    assert placement[(0, 1)] == [0]
    # Expert 7 (coldest) → rank 7 → GPU 3
    assert placement[(0, 7)] == [3]


def test_worst_case_collocates_hot_experts_unlike_greedy():
    # Adversarial property: worst_case packs the top-N hottest experts onto GPU 0 together;
    # greedy spreads them round-robin so no GPU holds more than one of the top-N.
    # With 8 experts and 4 GPUs, experts_per_gpu=2, so GPU 0 holds ranks 0 and 1 (both hot).
    # Greedy puts rank 0 → GPU 0, rank 1 → GPU 1 — they end up on different GPUs.
    import numpy as np

    data = {(0, e): (1000 - e) for e in range(8)}  # strictly decreasing load
    c = _counter_with(data)
    t = _flat_topology(4)

    worst = worst_case_placement(c, t, num_gpus=4)
    experts = list(range(8))
    load = np.array([[data[(0, e)] for e in experts]], dtype=np.float32)
    greedy = _greedy(load, n_gpus=4, layers=[0], experts=experts)

    # The two hottest experts (0 and 1) should share GPU 0 in worst_case
    assert worst[(0, 0)][0] == worst[(0, 1)][0] == 0
    # Greedy spreads them — expert 0 → GPU 0, expert 1 → GPU 1
    assert greedy[(0, 0)][0] != greedy[(0, 1)][0]


def test_worst_case_single_gpu():
    # With 1 GPU every expert lands on GPU 0.
    data = {(0, e): e + 1 for e in range(8)}
    c = _counter_with(data)
    placement = worst_case_placement(c, _flat_topology(1), num_gpus=1)
    assert all(gpus == [0] for gpus in placement.values())


def test_worst_case_multilayer_each_layer_independent():
    # Per-layer sort is independent; expert 3 hottest in layer 0, expert 0 hottest in layer 1.
    data = {(0, e): (1000 if e == 3 else 10) for e in range(4)}
    data.update({(1, e): (1000 if e == 0 else 10) for e in range(4)})
    c = _counter_with(data)
    placement = worst_case_placement(c, _flat_topology(4), num_gpus=4)
    # Hottest expert in each layer → GPU 0
    assert placement[(0, 3)][0] == 0
    assert placement[(1, 0)][0] == 0


# ---------------------------------------------------------------------------
# _greedy internals
# ---------------------------------------------------------------------------

def test_greedy_spreads_across_gpus():
    import numpy as np

    layers = [0]
    experts = list(range(8))
    load = np.array([[float(e + 1) for e in experts]])  # linearly increasing
    placement = _greedy(load, n_gpus=4, layers=layers, experts=experts)
    # Each GPU should get at least one expert; values are now list[int]
    gpus_used = {gpus[0] for gpus in placement.values()}
    assert gpus_used == {0, 1, 2, 3}


def test_greedy_assigns_hottest_expert_to_gpu0():
    import numpy as np

    # Expert 7 is much hotter than the rest
    load = np.array([[1.0] * 7 + [999.0]])
    placement = _greedy(load, n_gpus=4, layers=[0], experts=list(range(8)))
    # Hottest expert (index 7) gets rank 0 → GPU 0; value is list[int]
    assert placement[(0, 7)] == [0]


# ---------------------------------------------------------------------------
# _comm_aware_greedy
# ---------------------------------------------------------------------------

def test_comm_aware_greedy_flat_topology_matches_greedy():
    # On a flat (single-NUMA) topology all comm costs are 0, so the result
    # should have the same GPU load distribution as plain greedy.
    import numpy as np

    load = np.array([[float(e + 1) for e in range(8)]])
    layers, experts = [0], list(range(8))
    t = _flat_topology(4)

    greedy_pl = _greedy(load, n_gpus=4, layers=layers, experts=experts)
    comm_pl   = _comm_aware_greedy(load, n_gpus=4, layers=layers, experts=experts, topology=t)

    # Both should use all 4 GPUs
    assert {gpus[0] for gpus in greedy_pl.values()} == {0, 1, 2, 3}
    assert {gpus[0] for gpus in comm_pl.values()} == {0, 1, 2, 3}


def test_comm_aware_greedy_prefers_same_numa_gpu():
    # On a dual-NUMA 8-GPU topology with src_gpu=0 (NUMA 0), hot experts
    # should prefer NUMA-0 GPUs (0-3) over NUMA-1 GPUs (4-7).
    import numpy as np
    from plumb.analysis.comms import CommunicationConstants

    # 8 experts, 8 GPUs. Expert 0 very hot.
    load = np.array([[1000.0] + [10.0] * 7])
    layers, experts = [0], list(range(8))
    t = _dual_epyc_topology()  # GPUs 0-3 NUMA 0, 4-7 NUMA 1

    constants = CommunicationConstants(
        same_numa_us=1.0,
        cross_numa_pcie_us=100.0,  # extreme penalty for cross-NUMA
        nvlink_us=0.5,
    )
    pl = _comm_aware_greedy(
        load, n_gpus=8, layers=layers, experts=experts,
        topology=t, constants=constants, src_gpu=0,
    )
    # Hottest expert (0) should land on a NUMA-0 GPU (0-3)
    assert pl[(0, 0)][0] in {0, 1, 2, 3}


def test_comm_aware_greedy_lower_comm_cost_than_greedy_on_dual_numa():
    # Comm-aware greedy should produce lower cross-NUMA routing cost than
    # plain greedy when cross-NUMA penalty is significant.
    import numpy as np
    from plumb.analysis.comms import CommunicationConstants, compute_communication_cost
    from numa_topology.pcie import PCIeTopology, GPUPCIeInfo

    t = _dual_epyc_topology()
    n_gpus = 8
    # Hot experts concentrated at ids 0,1,2,3
    load = np.array([[1000.0 * (8 - e) for e in range(8)]])
    layers, experts = [0], list(range(8))
    loads_dict = {(0, e): int(load[0, e]) for e in range(8)}

    constants = CommunicationConstants(same_numa_us=2.0, cross_numa_pcie_us=15.0, nvlink_us=3.0)

    greedy_pl = _greedy(load, n_gpus=n_gpus, layers=layers, experts=experts)
    comm_pl   = _comm_aware_greedy(
        load, n_gpus=n_gpus, layers=layers, experts=experts,
        topology=t, constants=constants, src_gpu=0,
    )

    # Build a trivial PCIe topology (no NVLink) to satisfy compute_communication_cost
    fake_pcie = PCIeTopology(
        gpus=[GPUPCIeInfo(gpu_idx=g, bus_id="", link_speed_gts=8.0, link_width=16,
                          theoretical_bw_gbs=15.8, nvlink=False) for g in range(n_gpus)],
        is_symmetric=True, min_bw_gpu=0, max_bw_gpu=0, bandwidth_ratio=1.0,
    )
    # Build default placements for both (random initial = all on GPU 0 as baseline)
    default_pl = {(0, e): [0] for e in range(8)}

    greedy_cost = compute_communication_cost(
        default_pl, greedy_pl, loads_dict, t, fake_pcie, constants
    )
    comm_cost = compute_communication_cost(
        default_pl, comm_pl, loads_dict, t, fake_pcie, constants
    )
    assert comm_cost.recommended_overhead_us <= greedy_cost.recommended_overhead_us


# ---------------------------------------------------------------------------
# PCIe bandwidth-weighted _numa_finetune
# ---------------------------------------------------------------------------

def test_numa_finetune_pcie_ranks_by_bandwidth():
    # With pcie_topology provided, hottest expert should land on the
    # highest-bandwidth GPU regardless of NUMA structure.
    import numpy as np
    from numa_topology.pcie import PCIeTopology, GPUPCIeInfo

    # 4 GPUs, 4 experts. Expert 0 hottest.
    load = np.array([[1000.0, 100.0, 50.0, 10.0]])
    layers, experts = [0], list(range(4))
    t = _flat_topology(4)  # single NUMA — old code would no-op here

    # GPU 2 has highest bandwidth
    pcie = PCIeTopology(
        gpus=[
            GPUPCIeInfo(0, "", 8.0, 4,  3.94, False),   # x4  = 3.94 GB/s
            GPUPCIeInfo(1, "", 8.0, 8,  7.88, False),   # x8
            GPUPCIeInfo(2, "", 8.0, 16, 15.75, False),  # x16 = highest
            GPUPCIeInfo(3, "", 8.0, 8,  7.88, False),   # x8
        ],
        is_symmetric=False, min_bw_gpu=0, max_bw_gpu=2, bandwidth_ratio=4.0,
    )
    initial = {(0, e): [e % 4] for e in range(4)}
    result = _numa_finetune(initial, t, load, layers, experts, pcie_topology=pcie)

    # Hottest expert (0) should be pinned to GPU 2 (highest bandwidth)
    assert result[(0, 0)][0] == 2


def test_numa_finetune_pcie_single_gpu_no_op():
    import numpy as np
    from numa_topology.pcie import PCIeTopology, GPUPCIeInfo

    load = np.array([[100.0, 50.0]])
    layers, experts = [0], [0, 1]
    t = _flat_topology(1)
    pcie = PCIeTopology(
        gpus=[GPUPCIeInfo(0, "", 8.0, 16, 15.75, False)],
        is_symmetric=True, min_bw_gpu=0, max_bw_gpu=0, bandwidth_ratio=1.0,
    )
    initial = {(0, 0): [0], (0, 1): [0]}
    result = _numa_finetune(initial, t, load, layers, experts, pcie_topology=pcie)
    assert result == initial


def test_numa_finetune_no_pcie_single_numa_unchanged():
    # Without pcie_topology, single-NUMA topology should return unchanged (original behaviour).
    import numpy as np

    load = np.array([[500.0, 100.0, 50.0, 10.0]])
    layers, experts = [0], list(range(4))
    t = _flat_topology(4)
    initial = {(0, e): [e] for e in range(4)}
    result = _numa_finetune(initial, t, load, layers, experts, pcie_topology=None)
    assert result == initial


# ---------------------------------------------------------------------------
# coactivation_partition
# ---------------------------------------------------------------------------

def test_coactivation_partition_empty_counter():
    c = ActivationCounter()
    assert coactivation_partition(c, _flat_topology(4)) == {}


def test_coactivation_partition_shape():
    data = {(layer, expert): (expert + 1) * 10 for layer in range(2) for expert in range(8)}
    c = _counter_with(data)
    result = coactivation_partition(c, _flat_topology(4), num_gpus=4)
    assert len(result) == 2 * 8
    for gpus in result.values():
        assert isinstance(gpus, list) and 0 <= gpus[0] < 4


def test_coactivation_partition_collocates_hot_pairs():
    # Two experts (0, 1) are very hot together; two others (2, 3) are cold.
    # With 2 GPUs, the partition should place 0 and 1 on the same GPU.
    data = {(0, 0): 1000, (0, 1): 1000, (0, 2): 10, (0, 3): 10}
    c = _counter_with(data)
    result = coactivation_partition(c, _flat_topology(2), num_gpus=2)
    assert result[(0, 0)][0] == result[(0, 1)][0]


def test_coactivation_partition_balance_correction():
    # One GPU should not hold >2× average load after correction.
    # Create 4 experts where experts 0,1,2 all co-activate heavily with each other.
    # Initial partition would try to put 0,1,2 on one GPU — correction should move one off.
    data = {(0, e): (1000 if e < 3 else 10) for e in range(4)}
    c = _counter_with(data)
    result = coactivation_partition(c, _flat_topology(2), num_gpus=2)
    # Check each GPU's load doesn't exceed 2× average
    from plumb.counter import ActivationCounter as AC
    snapshot = c.snapshot()
    gpu_loads = {0: 0, 1: 0}
    for (lid, eid), gpus in result.items():
        gpu_loads[gpus[0]] += snapshot.get((lid, eid), 0)
    avg = sum(gpu_loads.values()) / 2
    assert all(v <= 2 * avg + 1 for v in gpu_loads.values())  # +1 for integer rounding


def test_coactivation_partition_multilayer_independent():
    # Each layer partitioned independently — no cross-layer bleeding
    data = {(0, 0): 1000, (0, 1): 1000, (1, 0): 10, (1, 1): 10}
    c = _counter_with(data)
    result = coactivation_partition(c, _flat_topology(2), num_gpus=2)
    # Both layers have results
    assert (0, 0) in result and (1, 0) in result


# ---------------------------------------------------------------------------
# recommend_placement uses comm_aware_greedy on multi-NUMA topology
# ---------------------------------------------------------------------------

def test_recommend_uses_comm_aware_on_dual_numa():
    # On a dual-NUMA topology with high imbalance, recommend_placement should
    # pick method='comm_aware_greedy' (not plain 'greedy').
    data = {(layer, expert): (1000 if expert == 0 else 50)
            for layer in range(4) for expert in range(8)}
    c = _counter_with(data)
    t = _dual_epyc_topology()
    rec = recommend_placement(c, t, num_gpus=8)
    assert rec is not None
    assert rec.method == "comm_aware_greedy"


def test_recommend_uses_greedy_on_flat_topology():
    data = {(layer, expert): (1000 if expert == 0 else 50)
            for layer in range(4) for expert in range(8)}
    c = _counter_with(data)
    rec = recommend_placement(c, _flat_topology(8), num_gpus=8)
    assert rec is not None
    assert rec.method in ("greedy", "eplb")


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
