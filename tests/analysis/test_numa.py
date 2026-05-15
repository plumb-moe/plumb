from pathlib import Path

from plumb.analysis.numa import compute_cross_numa
from plumb.topology import Topology

FIXTURES = Path(__file__).parent.parent / "fixtures" / "topologies"


def _dual_epyc_topology() -> Topology:
    return Topology.from_file(FIXTURES / "dual-epyc-8x-h100-sxm.json")


def _dual_epyc_pcie_topology() -> Topology:
    return Topology.from_file(FIXTURES / "dual-epyc-8x-h100-pcie.json")


def test_no_cross_numa_when_all_same_domain():
    topology = _dual_epyc_topology()
    # Place all experts on GPU 0 (NUMA 0)
    placement = {(0, e): 0 for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topology, src_gpu=0)
    assert len(stats) == 1
    assert stats[0].cross_numa_rate == 0.0


def test_all_cross_numa():
    topology = _dual_epyc_topology()
    # src_gpu=0 (NUMA 0), all experts on GPU 4 (NUMA 1)
    placement = {(0, e): 4 for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topology, src_gpu=0)
    assert stats[0].cross_numa_rate == 1.0


def test_half_cross_numa():
    topology = _dual_epyc_topology()
    # Experts 0-3 on GPU 0 (NUMA 0), experts 4-7 on GPU 4 (NUMA 1), equal load
    placement = {(0, e): (0 if e < 4 else 4) for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topology, src_gpu=0)
    assert abs(stats[0].cross_numa_rate - 0.5) < 0.01


def test_topology_from_file():
    t = _dual_epyc_topology()
    assert t.gpu_to_numa[0] == 0
    assert t.gpu_to_numa[4] == 1
    assert t.same_numa(0, 1)
    assert not t.same_numa(0, 4)


def test_flat_topology_no_cross_numa():
    topology = Topology.flat(4)
    placement = {(0, e): e % 4 for e in range(8)}
    loads = {(0, e): 50 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topology, src_gpu=0)
    assert stats[0].cross_numa_rate == 0.0


# ---------------------------------------------------------------------------
# dual-EPYC H100 PCIe fixture — same 4+4 NUMA split as SXM
# ---------------------------------------------------------------------------

def test_pcie_fixture_loads_and_maps_correctly():
    t = _dual_epyc_pcie_topology()
    # GPUs 0-3 on NUMA 0 (CPU0 socket), GPUs 4-7 on NUMA 1 (CPU1 socket)
    for gpu in range(4):
        assert t.gpu_to_numa[gpu] == 0
    for gpu in range(4, 8):
        assert t.gpu_to_numa[gpu] == 1
    assert t.same_numa(0, 3)
    assert t.same_numa(4, 7)
    assert not t.same_numa(3, 4)


def test_pcie_all_cross_numa():
    topology = _dual_epyc_pcie_topology()
    # src_gpu=1 (NUMA 0), all experts on GPU 5 (NUMA 1)
    placement = {(0, e): 5 for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topology, src_gpu=1)
    assert stats[0].cross_numa_rate == 1.0


def test_pcie_half_cross_numa():
    topology = _dual_epyc_pcie_topology()
    # Experts split evenly: half on GPU 0 (NUMA 0), half on GPU 4 (NUMA 1)
    placement = {(0, e): (0 if e < 4 else 4) for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topology, src_gpu=0)
    assert abs(stats[0].cross_numa_rate - 0.5) < 0.01


def test_pcie_no_cross_numa_intra_socket():
    topology = _dual_epyc_pcie_topology()
    # All on GPU 2 (NUMA 0), src_gpu=0 (NUMA 0) — zero cross-NUMA
    placement = {(0, e): 2 for e in range(8)}
    loads = {(0, e): 50 for e in range(8)}
    stats = compute_cross_numa(placement, loads, topology, src_gpu=0)
    assert stats[0].cross_numa_rate == 0.0
