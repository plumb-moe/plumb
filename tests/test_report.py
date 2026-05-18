import json

import pytest

from plumb.counter import ActivationCounter
from plumb.report.generator import generate_report
from plumb.report.schema import ExpertLoad, LayerReport, PlacementReport, ProfileReport
from plumb.topology import Topology


def _counter_with(data: dict[tuple[int, int], int], passes: int = 100) -> ActivationCounter:
    c = ActivationCounter(window_size=100_000)
    for (layer, expert), count in data.items():
        c.record(layer, expert, count)
    for _ in range(passes):
        c.increment_pass()
    return c


def _dual_epyc() -> Topology:
    return Topology({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1})


# ---------------------------------------------------------------------------
# top-level ProfileReport fields
# ---------------------------------------------------------------------------

def test_model_name_and_duration_preserved():
    c = _counter_with({(0, e): 100 for e in range(4)})
    report = generate_report(c, Topology.flat(4), "Mixtral-8x7B", 73.5)
    assert report.model_name == "Mixtral-8x7B"
    assert report.profiling_duration_seconds == pytest.approx(73.5)


def test_pass_count_matches_counter():
    c = _counter_with({(0, e): 100 for e in range(4)}, passes=250)
    report = generate_report(c, Topology.flat(4), "Model", 10.0)
    assert report.total_forward_passes == 250


def test_layer_count_matches_recorded_layers():
    c = _counter_with({(l, e): 100 for l in range(7) for e in range(4)})
    report = generate_report(c, Topology.flat(4), "Model", 10.0)
    assert len(report.layers) == 7


def test_expert_count_per_layer():
    c = _counter_with({(0, e): 100 for e in range(6)})
    report = generate_report(c, Topology.flat(4), "Model", 10.0)
    assert len(report.layers[0].experts) == 6


def test_empty_counter_produces_empty_report():
    c = ActivationCounter()
    report = generate_report(c, Topology.flat(4), "Model", 0.0)
    assert report.layers == []
    assert report.placement is None
    assert report.summary() == {}


# ---------------------------------------------------------------------------
# imbalance metrics
# ---------------------------------------------------------------------------

def test_imbalance_ratio_is_max_over_mean():
    # Expert 0: 800, experts 1-7: 100 → mean=1500/8=187.5, ratio=800/187.5≈4.267
    data = {(0, 0): 800}
    data.update({(0, e): 100 for e in range(1, 8)})
    report = generate_report(_counter_with(data), Topology.flat(8), "Model", 10.0)
    assert report.layers[0].imbalance_ratio == pytest.approx(800 / 187.5, rel=0.01)


def test_max_expert_id_is_hottest():
    data = {(0, e): (1000 if e == 5 else 50) for e in range(8)}
    report = generate_report(_counter_with(data), Topology.flat(8), "Model", 10.0)
    assert report.layers[0].max_expert_id == 5


def test_min_expert_id_is_coldest():
    data = {(0, e): (5 if e == 2 else 200) for e in range(8)}
    report = generate_report(_counter_with(data), Topology.flat(8), "Model", 10.0)
    assert report.layers[0].min_expert_id == 2


# ---------------------------------------------------------------------------
# activation fractions
# ---------------------------------------------------------------------------

def test_activation_fractions_sum_to_one_per_layer():
    c = _counter_with({(0, e): (e + 1) * 10 for e in range(8)})
    report = generate_report(c, Topology.flat(8), "Model", 10.0)
    total = sum(ex.activation_fraction for ex in report.layers[0].experts)
    assert total == pytest.approx(1.0, abs=0.001)


def test_activation_fraction_proportional_to_token_count():
    # Expert 0 gets 3× tokens of expert 1 → fraction must be 3×
    c = _counter_with({(0, 0): 300, (0, 1): 100})
    report = generate_report(c, Topology.flat(4), "Model", 10.0)
    by_expert = {ex.expert_id: ex for ex in report.layers[0].experts}
    assert by_expert[0].activation_fraction == pytest.approx(
        by_expert[1].activation_fraction * 3, rel=0.01
    )


# ---------------------------------------------------------------------------
# placement block
# ---------------------------------------------------------------------------

def test_placement_absent_on_empty_counter():
    report = generate_report(ActivationCounter(), Topology.flat(4), "Model", 0.0, num_gpus=4)
    assert report.placement is None


def test_placement_present_when_num_gpus_supplied():
    c = _counter_with({(l, e): 100 + e for l in range(4) for e in range(8)})
    report = generate_report(c, Topology.flat(8), "Model", 10.0, num_gpus=8)
    assert report.placement is not None


def test_placement_improvement_bounds_ordered():
    # Strongly imbalanced (expert 7 is 100× others) so method != 'none' and point
    # estimate falls within the paper's [min, max] range.
    c = _counter_with({(l, e): (1000 if e == 7 else 10) for l in range(4) for e in range(8)})
    report = generate_report(c, Topology.flat(8), "Model", 10.0, num_gpus=8)
    p = report.placement
    assert p.method != "none"
    assert p.estimated_improvement_pct_min <= p.estimated_improvement_pct <= p.estimated_improvement_pct_max


def test_placement_keys_are_layer_colon_expert_strings():
    c = _counter_with({(l, e): 50 for l in range(2) for e in range(4)})
    report = generate_report(c, Topology.flat(4), "Model", 5.0, num_gpus=4)
    if report.placement:
        for key in report.placement.expert_placement:
            layer_str, expert_str = key.split(":")
            assert layer_str.isdigit() and expert_str.isdigit()
            assert 0 <= int(layer_str) < 2
            assert 0 <= int(expert_str) < 4


def test_placement_gpu_values_are_valid_lists():
    c = _counter_with({(l, e): 100 + e * 10 for l in range(2) for e in range(8)})
    report = generate_report(c, Topology.flat(8), "Model", 10.0, num_gpus=8)
    if report.placement:
        for gpus in report.placement.expert_placement.values():
            assert isinstance(gpus, list) and len(gpus) >= 1
            assert all(0 <= g < 8 for g in gpus)


# ---------------------------------------------------------------------------
# cross-NUMA in report
# ---------------------------------------------------------------------------

def test_cross_numa_rate_present_on_multi_numa_topology():
    topology = _dual_epyc()
    c = _counter_with({(0, e): 100 for e in range(8)})
    report = generate_report(c, topology, "Model", 10.0, num_gpus=8)
    rate = report.layers[0].cross_numa_rate
    assert rate is not None
    assert 0.0 <= rate <= 1.0


def test_cross_numa_rate_zero_on_flat_topology():
    """Flat (single-NUMA) topology: all traffic is intra-NUMA, rate is 0.0."""
    c = _counter_with({(0, e): 100 for e in range(4)})
    report = generate_report(c, Topology.flat(4), "Model", 10.0, num_gpus=4)
    assert report.layers[0].cross_numa_rate == 0.0


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_layer_and_pass_counts():
    c = _counter_with({(l, e): 10 * (e + 1) for l in range(3) for e in range(8)}, passes=500)
    s = generate_report(c, Topology.flat(8), "Model", 30.0).summary()
    assert s["num_layers_profiled"] == 3
    assert s["total_forward_passes"] == 500


def test_summary_max_imbalance_gte_mean():
    c = _counter_with({(l, e): 10 * (e + 1) for l in range(4) for e in range(8)})
    s = generate_report(c, Topology.flat(8), "Model", 10.0).summary()
    assert s["max_imbalance_ratio"] >= s["mean_imbalance_ratio"]


def test_summary_worst_layer_has_max_ratio():
    # Layer 3 is the most imbalanced — spike one expert to 5000
    data = {(l, e): 100 for l in range(4) for e in range(8)}
    data[(3, 0)] = 5000
    s = generate_report(_counter_with(data), Topology.flat(8), "Model", 10.0).summary()
    assert s["worst_layer_id"] == 3


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_json_round_trip_preserves_token_counts():
    data = {(0, e): (e + 1) * 20 for e in range(4)}
    c = _counter_with(data, passes=50)
    report = generate_report(c, Topology.flat(4), "RoundTripModel", 5.0)
    parsed = json.loads(report.model_dump_json())

    assert parsed["model_name"] == "RoundTripModel"
    assert parsed["total_forward_passes"] == 50
    experts_by_id = {ex["expert_id"]: ex for ex in parsed["layers"][0]["experts"]}
    for eid in range(4):
        assert experts_by_id[eid]["token_count"] == (eid + 1) * 20


def test_json_round_trip_all_required_fields_present():
    c = _counter_with({(l, e): 100 for l in range(3) for e in range(8)})
    data = json.loads(generate_report(c, Topology.flat(8), "Model", 10.0, num_gpus=8).model_dump_json())

    assert {"model_name", "hardware_config", "profiling_duration_seconds",
            "total_forward_passes", "layers", "generated_at"} <= data.keys()
    assert {"layer_id", "imbalance_ratio", "max_expert_id", "min_expert_id",
            "experts"} <= data["layers"][0].keys()
    assert {"expert_id", "token_count", "activation_fraction"} <= data["layers"][0]["experts"][0].keys()
    assert {"method", "expert_placement", "estimated_improvement_pct_min",
            "estimated_improvement_pct_max"} <= data["placement"].keys()
