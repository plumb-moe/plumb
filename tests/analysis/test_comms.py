from numa_topology import Topology
from numa_topology.pcie import GPUPCIeInfo, PCIeTopology
from plumb.analysis.comms import (
    CommunicationConstants,
    compute_communication_cost,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _dual_numa_topology() -> Topology:
    """GPUs 0-3 on NUMA 0, GPUs 4-7 on NUMA 1."""
    return Topology({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1})


def _flat_pcie_topology(num_gpus: int = 8, nvlink: bool = False) -> PCIeTopology:
    """Symmetric PCIe topology; optionally mark all GPUs as NVLink-capable."""
    gpus = [
        GPUPCIeInfo(gpu_idx=i, bus_id=f"0000:0{i}:00.0",
                    link_speed_gts=16.0, link_width=16,
                    theoretical_bw_gbs=31.508, nvlink=nvlink)
        for i in range(num_gpus)
    ]
    return PCIeTopology(gpus=gpus, is_symmetric=True, min_bw_gpu=0,
                        max_bw_gpu=0, bandwidth_ratio=1.0)


def _nvlink_topology(num_gpus: int = 8) -> PCIeTopology:
    return _flat_pcie_topology(num_gpus, nvlink=True)


# ---------------------------------------------------------------------------
# delta == 0 when placements are identical (AC requirement)
# ---------------------------------------------------------------------------

def test_delta_zero_when_placements_identical():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    placement = {(0, e): [e % 4] for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    result = compute_communication_cost(placement, placement, loads, topo, pcie)
    assert result.delta_us == 0.0
    assert result.current_overhead_us == result.recommended_overhead_us


# ---------------------------------------------------------------------------
# All dispatches same GPU → zero overhead
# ---------------------------------------------------------------------------

def test_same_gpu_zero_overhead():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    placement = {(0, e): [0] for e in range(8)}
    loads = {(0, e): 100 for e in range(8)}
    result = compute_communication_cost(placement, placement, loads, topo, pcie, src_gpu=0)
    assert result.current_overhead_us == 0.0
    assert result.per_layer_current[0].cross_numa_rate == 0.0
    assert result.per_layer_current[0].cross_pcie_rate == 0.0


# ---------------------------------------------------------------------------
# All cross-NUMA PCIe dispatches use cross_numa_pcie_us constant
# ---------------------------------------------------------------------------

def test_all_cross_numa_uses_correct_latency():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    constants = CommunicationConstants(cross_numa_pcie_us=15.0)
    # src=0 (NUMA 0), all experts on GPU 4 (NUMA 1)
    placement = {(0, e): [4] for e in range(4)}
    loads = {(0, e): 100 for e in range(4)}
    result = compute_communication_cost(placement, placement, loads, topo, pcie,
                                        constants=constants, src_gpu=0)
    layer = result.per_layer_current[0]
    # 4 experts × 100 tokens each = 400 total dispatches, all cross-NUMA
    assert layer.cross_numa_rate == 1.0
    assert layer.cross_pcie_rate == 1.0
    assert layer.total_dispatches == 400
    # 400 tokens × 15 us = 6000 us
    assert abs(layer.estimated_overhead_us - 6000.0) < 0.01


# ---------------------------------------------------------------------------
# Same-NUMA, cross-GPU dispatches use same_numa_us constant
# ---------------------------------------------------------------------------

def test_same_numa_cross_gpu_latency():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    constants = CommunicationConstants(same_numa_us=2.0)
    # src=0, experts on GPU 1 (same NUMA 0)
    placement = {(0, e): [1] for e in range(4)}
    loads = {(0, e): 50 for e in range(4)}
    result = compute_communication_cost(placement, placement, loads, topo, pcie,
                                        constants=constants, src_gpu=0)
    layer = result.per_layer_current[0]
    assert layer.cross_numa_rate == 0.0
    assert layer.cross_pcie_rate == 1.0   # PCIe, just same-NUMA
    # 200 tokens × 2 us = 400 us
    assert abs(layer.estimated_overhead_us - 400.0) < 0.01


# ---------------------------------------------------------------------------
# NVLink dispatches use nvlink_us and are NOT counted as cross_pcie
# ---------------------------------------------------------------------------

def test_nvlink_not_counted_as_cross_pcie():
    topo = _dual_numa_topology()
    pcie = _nvlink_topology()
    constants = CommunicationConstants(nvlink_us=3.0)
    # src=0, experts on GPU 4 (NUMA 1, but NVLink)
    placement = {(0, e): [4] for e in range(4)}
    loads = {(0, e): 100 for e in range(4)}
    result = compute_communication_cost(placement, placement, loads, topo, pcie,
                                        constants=constants, src_gpu=0)
    layer = result.per_layer_current[0]
    # cross-NUMA (NUMA 0 → NUMA 1), but NOT cross-PCIe (NVLink)
    assert layer.cross_numa_rate == 1.0
    assert layer.cross_pcie_rate == 0.0
    # 400 tokens × 3 us = 1200 us
    assert abs(layer.estimated_overhead_us - 1200.0) < 0.01


# ---------------------------------------------------------------------------
# constants are overridable
# ---------------------------------------------------------------------------

def test_constants_overridable():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    placement = {(0, e): [4] for e in range(4)}
    loads = {(0, e): 10 for e in range(4)}
    custom = CommunicationConstants(same_numa_us=1.0, cross_numa_pcie_us=100.0, nvlink_us=0.5)
    result = compute_communication_cost(placement, placement, loads, topo, pcie,
                                        constants=custom, src_gpu=0)
    layer = result.per_layer_current[0]
    # 40 tokens × 100 us (cross-NUMA PCIe at custom rate)
    assert abs(layer.estimated_overhead_us - 4000.0) < 0.01


# ---------------------------------------------------------------------------
# Improvement: recommended placement reduces overhead vs current
# ---------------------------------------------------------------------------

def test_recommended_placement_reduces_overhead():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    constants = CommunicationConstants(same_numa_us=2.0, cross_numa_pcie_us=15.0)
    loads = {(0, e): 100 for e in range(8)}
    # Current: all experts on GPU 4 (NUMA 1) → cross-NUMA dispatches at 15 us
    current = {(0, e): [4] for e in range(8)}
    # Recommended: all experts on GPU 1 (NUMA 0) → same-NUMA dispatches at 2 us
    recommended = {(0, e): [1] for e in range(8)}
    result = compute_communication_cost(current, recommended, loads, topo, pcie,
                                        constants=constants, src_gpu=0)
    assert result.delta_us < 0.0     # improvement
    assert result.recommended_overhead_us < result.current_overhead_us


# ---------------------------------------------------------------------------
# Multi-layer loads
# ---------------------------------------------------------------------------

def test_multi_layer_stats():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    # Layer 0: experts on same GPU → zero overhead
    # Layer 1: experts on GPU 4 (cross-NUMA) → overhead
    placement = {
        (0, 0): [0], (0, 1): [0],
        (1, 0): [4], (1, 1): [4],
    }
    loads = {(0, 0): 100, (0, 1): 100, (1, 0): 100, (1, 1): 100}
    result = compute_communication_cost(placement, placement, loads, topo, pcie, src_gpu=0)
    layers = {s.layer_id: s for s in result.per_layer_current}
    assert layers[0].estimated_overhead_us == 0.0
    assert layers[1].cross_numa_rate == 1.0
    assert layers[1].estimated_overhead_us > 0.0


# ---------------------------------------------------------------------------
# Missing placement entry defaults to src_gpu (no cross-GPU cost)
# ---------------------------------------------------------------------------

def test_missing_placement_entry_defaults_to_src_gpu():
    topo = _dual_numa_topology()
    pcie = _flat_pcie_topology()
    # placement is empty; all experts default to src_gpu=0 → zero overhead
    loads = {(0, e): 100 for e in range(4)}
    result = compute_communication_cost({}, {}, loads, topo, pcie, src_gpu=0)
    assert result.current_overhead_us == 0.0
