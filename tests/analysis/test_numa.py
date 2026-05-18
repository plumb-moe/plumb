from pathlib import Path

import pytest

from plumb.analysis.numa import compute_cross_numa, NumaStats
from plumb.topology import Topology

FIXTURES = Path(__file__).parent.parent / "fixtures" / "topologies"


def _dual_epyc() -> Topology:
    return Topology.from_file(FIXTURES / "dual-epyc-8x-h100-sxm.json")


def _pcie() -> Topology:
    return Topology.from_file(FIXTURES / "dual-epyc-8x-h100-pcie.json")


# ---------------------------------------------------------------------------
# rate correctness
# ---------------------------------------------------------------------------

def test_all_intra_numa_rate_is_zero():
    topo = _dual_epyc()
    placement = {(0, e): 0 for e in range(8)}   # GPU 0 = NUMA 0
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_rate == 0.0


def test_all_cross_numa_rate_is_one():
    topo = _dual_epyc()
    placement = {(0, e): 4 for e in range(8)}   # GPU 4 = NUMA 1
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_rate == 1.0


def test_half_cross_numa_rate():
    topo = _dual_epyc()
    # Experts 0-3 → GPU 0 (NUMA 0), experts 4-7 → GPU 4 (NUMA 1), equal loads
    placement = {(0, e): (0 if e < 4 else 4) for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_rate == pytest.approx(0.5, abs=0.01)


def test_flat_topology_always_zero_rate():
    topo = Topology.flat(4)
    placement = {(0, e): e % 4 for e in range(8)}
    loads = {(0, e): 50 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_rate == 0.0


# ---------------------------------------------------------------------------
# NumaStats field values
# ---------------------------------------------------------------------------

def test_total_dispatches_equals_sum_of_all_loads():
    topo = _dual_epyc()
    loads = {(0, 0): 300, (0, 1): 200, (0, 2): 100}
    placement = {(0, e): 0 for e in range(3)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].total_dispatches == 600


def test_cross_numa_dispatches_counts_only_remote_loads():
    topo = _dual_epyc()
    # Experts 0,1 → GPU 0 (NUMA 0, intra). Expert 2 → GPU 4 (NUMA 1, cross).
    loads = {(0, 0): 100, (0, 1): 100, (0, 2): 200}
    placement = {(0, 0): 0, (0, 1): 0, (0, 2): 4}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_dispatches == 200
    assert stats[0].total_dispatches == 400
    assert stats[0].cross_numa_rate == pytest.approx(0.5, abs=0.01)


def test_rate_is_load_weighted_not_expert_count_weighted():
    """A single high-load expert going cross-NUMA dominates the rate."""
    topo = _dual_epyc()
    # 7 experts on NUMA 0 with load 1 each; 1 expert on NUMA 1 with load 1000
    loads = {(0, e): (1000 if e == 7 else 1) for e in range(8)}
    placement = {(0, e): (4 if e == 7 else 0) for e in range(8)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    # Rate ≈ 1000 / (7 + 1000) ≈ 0.993 — not 1/8 = 0.125 (naïve expert count)
    assert stats[0].cross_numa_rate > 0.99


# ---------------------------------------------------------------------------
# multi-layer
# ---------------------------------------------------------------------------

def test_multi_layer_produces_one_stat_per_layer():
    topo = _dual_epyc()
    loads = {(layer, e): 100 for layer in range(4) for e in range(8)}
    placement = {(layer, e): 0 for layer in range(4) for e in range(8)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert len(stats) == 4


def test_multi_layer_results_sorted_by_layer_id():
    topo = _dual_epyc()
    loads = {(layer, e): 50 for layer in [3, 1, 0, 2] for e in range(4)}
    placement = {(layer, e): 0 for layer in [3, 1, 0, 2] for e in range(4)}
    stats = compute_cross_numa(placement, loads, topo)
    assert [s.layer_id for s in stats] == [0, 1, 2, 3]


def test_per_layer_dispatches_are_independent():
    topo = _dual_epyc()
    # Layer 0: all intra. Layer 1: all cross.
    loads = {(0, e): 100 for e in range(4)}
    loads.update({(1, e): 100 for e in range(4)})
    placement = {(0, e): 0 for e in range(4)}
    placement.update({(1, e): 4 for e in range(4)})
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    by_layer = {s.layer_id: s for s in stats}
    assert by_layer[0].cross_numa_rate == 0.0
    assert by_layer[1].cross_numa_rate == 1.0


# ---------------------------------------------------------------------------
# list[int] placement (expert replication)
# ---------------------------------------------------------------------------

def test_replicated_placement_uses_primary_gpu_for_numa_check():
    """Primary GPU (index 0 of list) determines cross-NUMA, not replicas."""
    topo = _dual_epyc()
    # Primary on GPU 0 (NUMA 0) with a replica on GPU 4 (NUMA 1) — should be intra
    placement: dict = {(0, e): [0, 4] for e in range(4)}
    loads = {(0, e): 100 for e in range(4)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_rate == 0.0


def test_replicated_placement_cross_socket_when_primary_is_remote():
    topo = _dual_epyc()
    placement: dict = {(0, e): [4, 0] for e in range(4)}   # primary on GPU 4 = NUMA 1
    loads = {(0, e): 100 for e in range(4)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_rate == 1.0


# ---------------------------------------------------------------------------
# PCIe fixture (same 4+4 NUMA split, different form factor)
# ---------------------------------------------------------------------------

def test_pcie_fixture_numa_split():
    t = _pcie()
    assert all(t.gpu_to_numa[g] == 0 for g in range(4))
    assert all(t.gpu_to_numa[g] == 1 for g in range(4, 8))


def test_pcie_all_cross_numa():
    placement = {(0, e): 5 for e in range(8)}   # GPU 5 = NUMA 1
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, _pcie(), src_gpu=1)
    assert stats[0].cross_numa_rate == 1.0


def test_pcie_intra_socket_rate_zero():
    topo = _pcie()
    placement = {(0, e): 2 for e in range(8)}   # GPU 2 = NUMA 0
    loads = {(0, e): 50 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topo, src_gpu=0)
    assert stats[0].cross_numa_rate == 0.0
