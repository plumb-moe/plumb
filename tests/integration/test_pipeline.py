"""Integration tests for the complete plumb analysis pipeline.

These tests validate end-to-end analysis using synthetic activations — no GPU
hardware is required.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from plumb.analysis.imbalance import compute_imbalance
from plumb.analysis.placement import recommend_placement
from plumb.counter import ActivationCounter
from plumb.report.generator import generate_report, generate_report_from_snapshot
from plumb.report.schema import ProfileReport
from plumb.simulation import from_profile
from plumb.topology import Topology

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_counter(load: np.ndarray, passes: int = 1000) -> ActivationCounter:
    """Feed a (num_layers, num_experts) load matrix into an ActivationCounter."""
    counter = ActivationCounter(window_size=10_000_000)
    for lid in range(load.shape[0]):
        for eid in range(load.shape[1]):
            count = int(load[lid, eid])
            if count > 0:
                counter.record(lid, eid, count)
    for _ in range(passes):
        counter.increment_pass()
    return counter


def _dual_epyc_topology() -> Topology:
    """8-GPU dual-EPYC topology: GPUs 0-3 on NUMA 0, GPUs 4-7 on NUMA 1."""
    return Topology({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1})


def _snapshot_from_load(
    load: np.ndarray,
    model_name: str = "DeepSeek-V3",
    pass_count: int = 1000,
) -> dict:
    now = time.time()
    return {
        "pid": 12345,
        "model_name": model_name,
        "pass_count": pass_count,
        "updated_at": now,
        "started_at": now - 300.0,
        "gpu_to_numa": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1, "7": 1},
        "expert_counts": {
            f"{lid}:{eid}": int(load[lid, eid])
            for lid in range(load.shape[0])
            for eid in range(load.shape[1])
            if load[lid, eid] > 0
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_deepseek_v3_full_pipeline():
    """DeepSeek V3 synthetic data passes through imbalance → placement → report."""
    load = from_profile("deepseek_v3", seed=42)
    assert load.shape == (61, 256)

    counter = _load_counter(load)

    # At least half of the 61 layers have imbalance ratio > 1.1
    imbalances = compute_imbalance(counter)
    assert len(imbalances) == 61
    ratios_above = sum(1 for i in imbalances if i.imbalance_ratio > 1.1)
    assert ratios_above >= len(imbalances) // 2, (
        f"Expected at least {len(imbalances) // 2} layers with ratio > 1.1, got {ratios_above}"
    )

    # Placement must be non-None
    topo = _dual_epyc_topology()
    rec = recommend_placement(counter, topo, num_gpus=8)
    assert rec is not None

    # generate_report produces a valid ProfileReport with all layers and placement
    report = generate_report(counter, topo, "DeepSeek-V3", 300.0, num_gpus=8)
    assert isinstance(report, ProfileReport)
    assert len(report.layers) == 61
    assert report.placement is not None


def test_mixtral_full_pipeline():
    """Mixtral-8x7B synthetic data (8 experts, 32 layers) passes end-to-end."""
    load = from_profile("mixtral_8x7b", seed=42)
    assert load.shape == (32, 8)

    counter = _load_counter(load)

    imbalances = compute_imbalance(counter)
    assert len(imbalances) == 32
    ratios_above = sum(1 for i in imbalances if i.imbalance_ratio > 1.1)
    assert ratios_above >= len(imbalances) // 2, (
        f"Expected at least {len(imbalances) // 2} layers with ratio > 1.1, got {ratios_above}"
    )

    topo = _dual_epyc_topology()
    rec = recommend_placement(counter, topo, num_gpus=8)
    assert rec is not None

    report = generate_report(counter, topo, "Mixtral-8x7B", 120.0, num_gpus=8)
    assert isinstance(report, ProfileReport)
    assert len(report.layers) == 32
    assert report.placement is not None


def test_snapshot_round_trip():
    """Snapshot dict from DeepSeek V3 survives generate_report_from_snapshot."""
    load = from_profile("deepseek_v3", seed=42)
    snapshot = _snapshot_from_load(load, model_name="DeepSeek-V3", pass_count=1000)

    report = generate_report_from_snapshot(snapshot)

    assert isinstance(report, ProfileReport)
    assert report.model_name == "DeepSeek-V3"
    assert report.total_forward_passes == 1000

    # 61 layers, each with 256 experts (all experts present in deepseek_v3 synthetic data)
    assert len(report.layers) == 61
    for layer in report.layers:
        assert len(layer.experts) == 256, (
            f"Layer {layer.layer_id} has {len(layer.experts)} experts, expected 256"
        )


def test_out101_json_schema_compliance():
    """OUT-101: serialised JSON contains all required fields at every level."""
    load = from_profile("deepseek_v3", seed=42)
    counter = _load_counter(load)
    topo = _dual_epyc_topology()
    report = generate_report(counter, topo, "DeepSeek-V3", 300.0, num_gpus=8)

    data = json.loads(report.model_dump_json())

    # Top-level required fields
    required_top = {
        "model_name",
        "hardware_config",
        "profiling_duration_seconds",
        "total_forward_passes",
        "layers",
        "generated_at",
    }
    assert required_top <= data.keys()

    # Per-layer required fields
    layer = data["layers"][0]
    required_layer = {"layer_id", "imbalance_ratio", "max_expert_id", "min_expert_id", "experts"}
    assert required_layer <= layer.keys()

    # Per-expert required fields
    expert = layer["experts"][0]
    assert {"expert_id", "token_count", "activation_fraction"} <= expert.keys()

    # Placement block required fields
    assert data["placement"] is not None
    placement = data["placement"]
    required_placement = {
        "method",
        "expert_placement",
        "estimated_improvement_pct_min",
        "estimated_improvement_pct_max",
    }
    assert required_placement <= placement.keys()


def test_placement_uses_topology():
    """Dual-EPYC topology: hottest expert in each layer lands on a NUMA-0 GPU (0-3)."""
    load = from_profile("deepseek_v3", seed=42)
    counter = _load_counter(load)

    topo = _dual_epyc_topology()
    rec = recommend_placement(counter, topo, num_gpus=8)
    assert rec is not None

    numa0_gpus = set(topo.gpus_in_numa(0))  # {0, 1, 2, 3}

    snap = counter.snapshot()
    for lid in range(load.shape[0]):
        layer_loads = {eid: snap.get((lid, eid), 0) for eid in range(load.shape[1])}
        if not any(layer_loads.values()):
            continue
        hottest_eid = max(layer_loads, key=layer_loads.__getitem__)
        gpus = rec.expert_placement.get((lid, hottest_eid))
        assert gpus is not None, f"No placement for (layer={lid}, expert={hottest_eid})"
        gpu = gpus[0] if isinstance(gpus, list) else gpus
        assert gpu in numa0_gpus, (
            f"Layer {lid}: hottest expert {hottest_eid} placed on GPU {gpu}, "
            f"expected one of NUMA-0 GPUs {numa0_gpus}"
        )


def test_token_conservation_through_pipeline():
    """Token counts in counter snapshot sum to batch_size * seq_len * active_k per layer."""
    # deepseek_v3: batch_size=32, seq_len=512, active_k=8 → 131072 tokens/layer
    expected_per_layer = 32 * 512 * 8  # 131072

    load = from_profile("deepseek_v3", seed=42)
    counter = _load_counter(load, passes=1000)

    snap = counter.snapshot()
    by_layer: dict[int, int] = {}
    for (lid, eid), cnt in snap.items():
        by_layer[lid] = by_layer.get(lid, 0) + cnt

    assert len(by_layer) == 61
    for lid in sorted(by_layer):
        assert by_layer[lid] == expected_per_layer, (
            f"Layer {lid}: got {by_layer[lid]} tokens, expected {expected_per_layer}"
        )


def test_report_summary_metrics():
    """report.summary() returns correct aggregate metrics."""
    load = from_profile("deepseek_v3", seed=42)
    counter = _load_counter(load, passes=1000)
    topo = _dual_epyc_topology()
    report = generate_report(counter, topo, "DeepSeek-V3", 300.0, num_gpus=8)

    summary = report.summary()

    assert summary["num_layers_profiled"] == 61
    assert summary["max_imbalance_ratio"] >= summary["mean_imbalance_ratio"]
    assert summary["total_forward_passes"] == 1000


def test_generate_report_from_snapshot_missing_expert_counts():
    """generate_report_from_snapshot raises ValueError when expert_counts is absent."""
    snapshot = {
        "pid": 99999,
        "model_name": "TestModel",
        "pass_count": 10,
        "updated_at": time.time(),
        "started_at": time.time() - 10.0,
        "gpu_to_numa": {"0": 0},
    }
    with pytest.raises(ValueError, match="expert_counts"):
        generate_report_from_snapshot(snapshot)
