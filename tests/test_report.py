import json

from plumb.counter import ActivationCounter
from plumb.report.generator import generate_report
from plumb.report.schema import ProfileReport, PlacementReport
from plumb.topology import Topology


def _counter_with(data: dict[tuple[int, int], int], passes: int = 100) -> ActivationCounter:
    c = ActivationCounter(window_size=100_000)
    for (layer, expert), count in data.items():
        c.record(layer, expert, count)
    for _ in range(passes):
        c.increment_pass()
    return c


def _dual_epyc() -> Topology:
    return Topology({"0": 0, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1, "7": 1})


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_profile_report_schema_fields():
    c = _counter_with({(0, e): 100 for e in range(8)})
    report = generate_report(c, Topology.flat(8), "TestModel", 42.0)
    assert report.model_name == "TestModel"
    assert report.profiling_duration_seconds == 42.0
    assert report.total_forward_passes == 100
    assert len(report.layers) == 1
    assert report.generated_at is not None


def test_report_json_round_trip():
    c = _counter_with({(l, e): 50 * (e + 1) for l in range(3) for e in range(8)})
    report = generate_report(c, Topology.flat(8), "Mixtral-8x7B", 60.0, num_gpus=8)
    data = json.loads(report.model_dump_json())
    assert data["model_name"] == "Mixtral-8x7B"
    assert len(data["layers"]) == 3
    # Must be parseable by jq-equivalent
    assert isinstance(data["layers"][0]["experts"], list)


def test_report_has_all_required_fields():
    """OUT-101: report contains all required fields."""
    c = _counter_with({(l, e): 100 for l in range(4) for e in range(8)})
    report = generate_report(c, Topology.flat(8), "DeepSeek-V3", 120.0, num_gpus=8)
    data = json.loads(report.model_dump_json())

    required_top = {"model_name", "hardware_config", "profiling_duration_seconds",
                    "total_forward_passes", "layers", "generated_at"}
    assert required_top <= data.keys()

    layer = data["layers"][0]
    required_layer = {"layer_id", "imbalance_ratio", "max_expert_id", "min_expert_id", "experts"}
    assert required_layer <= layer.keys()

    expert = layer["experts"][0]
    assert {"expert_id", "token_count", "activation_fraction"} <= expert.keys()


def test_imbalance_ratio_in_report():
    # Expert 0 gets 8× more tokens than others
    data = {(0, 0): 800}
    for e in range(1, 8):
        data[(0, e)] = 100
    c = _counter_with(data)
    report = generate_report(c, Topology.flat(8), "Model", 10.0)
    layer = report.layers[0]
    assert layer.imbalance_ratio > 1.0
    assert layer.max_expert_id == 0


def test_placement_report_included_with_multi_gpu():
    c = _counter_with({(l, e): 100 + e for l in range(4) for e in range(8)})
    report = generate_report(c, Topology.flat(8), "Model", 10.0, num_gpus=8)
    assert report.placement is not None
    assert report.placement.estimated_improvement_pct_min > 0
    assert report.placement.estimated_improvement_pct_max > report.placement.estimated_improvement_pct_min


def test_placement_keys_are_json_safe():
    """PlacementReport keys must be JSON-serialisable strings (layer:expert format)."""
    c = _counter_with({(l, e): 50 for l in range(2) for e in range(4)})
    report = generate_report(c, Topology.flat(4), "Model", 5.0, num_gpus=4)
    if report.placement:
        for key in report.placement.expert_placement:
            assert ":" in key
            layer_str, expert_str = key.split(":")
            assert layer_str.isdigit()
            assert expert_str.isdigit()


def test_cross_numa_rate_in_report():
    topology = _dual_epyc()
    c = _counter_with({(0, e): 100 for e in range(8)})
    report = generate_report(c, topology, "Model", 10.0, num_gpus=8)
    layer = report.layers[0]
    assert layer.cross_numa_rate is not None
    assert 0.0 <= layer.cross_numa_rate <= 1.0


def test_summary_method():
    c = _counter_with({(l, e): 10 * (e + 1) for l in range(3) for e in range(8)})
    report = generate_report(c, Topology.flat(8), "Model", 30.0)
    s = report.summary()
    assert s["num_layers_profiled"] == 3
    assert s["total_forward_passes"] == 100
    assert s["max_imbalance_ratio"] >= s["mean_imbalance_ratio"]


def test_empty_counter_produces_empty_report():
    c = ActivationCounter()
    report = generate_report(c, Topology.flat(4), "Model", 0.0)
    assert report.layers == []
    assert report.placement is None
    assert report.summary() == {}
