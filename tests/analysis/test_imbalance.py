import pytest

from plumb.analysis.imbalance import compute_imbalance
from plumb.counter import ActivationCounter


def _counter_with(data: dict[tuple[int, int], int]) -> ActivationCounter:
    c = ActivationCounter(window_size=100_000)
    for (layer, expert), count in data.items():
        c.record(layer, expert, count)
    return c


# ---------------------------------------------------------------------------
# empty and trivial inputs
# ---------------------------------------------------------------------------

def test_empty_counter_returns_empty_list():
    assert compute_imbalance(ActivationCounter()) == []


def test_single_expert_ratio_is_one():
    c = _counter_with({(0, 0): 500})
    result = compute_imbalance(c)
    assert len(result) == 1
    assert result[0].imbalance_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ratio formula: imbalance_ratio = max_load / mean_load
# ---------------------------------------------------------------------------

def test_perfectly_balanced_ratio_is_one():
    c = _counter_with({(0, e): 100 for e in range(8)})
    assert compute_imbalance(c)[0].imbalance_ratio == pytest.approx(1.0, rel=1e-3)


def test_ratio_matches_max_over_mean():
    # loads: [10]*7 + [30] → mean=12.5, max=30, ratio=2.4
    data = {(0, e): 10 for e in range(7)}
    data[(0, 7)] = 30
    result = compute_imbalance(_counter_with(data))[0]
    assert result.imbalance_ratio == pytest.approx(2.4, rel=1e-3)


def test_extreme_imbalance_correct_ratio():
    # expert 0: 800, experts 1-7: 100 → mean=1500/8=187.5, max=800, ratio≈4.267
    data = {(0, 0): 800}
    data.update({(0, e): 100 for e in range(1, 8)})
    result = compute_imbalance(_counter_with(data))[0]
    assert result.imbalance_ratio == pytest.approx(800 / 187.5, rel=0.01)


# ---------------------------------------------------------------------------
# max_expert_id and min_expert_id
# ---------------------------------------------------------------------------

def test_max_expert_id_identifies_hottest():
    data = {(0, e): (1000 if e == 3 else 100) for e in range(8)}
    assert compute_imbalance(_counter_with(data))[0].max_expert_id == 3


def test_min_expert_id_identifies_coldest():
    data = {(0, e): (5 if e == 6 else 200) for e in range(8)}
    assert compute_imbalance(_counter_with(data))[0].min_expert_id == 6


def test_max_and_min_differ_when_imbalanced():
    data = {(0, e): (e + 1) * 10 for e in range(8)}   # expert 7 is hottest, 0 is coldest
    result = compute_imbalance(_counter_with(data))[0]
    assert result.max_expert_id == 7
    assert result.min_expert_id == 0


# ---------------------------------------------------------------------------
# expert_loads content
# ---------------------------------------------------------------------------

def test_expert_loads_contains_all_recorded_experts():
    data = {(0, e): (e + 1) * 10 for e in range(6)}
    result = compute_imbalance(_counter_with(data))[0]
    assert set(result.expert_loads.keys()) == {0, 1, 2, 3, 4, 5}


def test_expert_loads_values_match_input():
    data = {(0, 0): 300, (0, 1): 150, (0, 2): 50}
    result = compute_imbalance(_counter_with(data))[0]
    assert result.expert_loads == {0: 300, 1: 150, 2: 50}


# ---------------------------------------------------------------------------
# multi-layer
# ---------------------------------------------------------------------------

def test_multi_layer_produces_one_result_per_layer():
    c = ActivationCounter()
    for layer in range(5):
        for expert in range(4):
            c.record(layer, expert, 100 + expert)
    assert len(compute_imbalance(c)) == 5


def test_multi_layer_results_sorted_by_layer_id():
    c = ActivationCounter()
    for layer in [3, 0, 2, 1]:
        for expert in range(4):
            c.record(layer, expert, 100)
    result = compute_imbalance(c)
    assert [r.layer_id for r in result] == [0, 1, 2, 3]


def test_layer_imbalance_computed_independently():
    # Layer 0 balanced, layer 1 heavily imbalanced
    data = {(0, e): 100 for e in range(4)}
    data.update({(1, 0): 1000, (1, 1): 1, (1, 2): 1, (1, 3): 1})
    result = compute_imbalance(_counter_with(data))
    by_layer = {r.layer_id: r for r in result}
    assert by_layer[0].imbalance_ratio == pytest.approx(1.0, rel=1e-3)
    assert by_layer[1].imbalance_ratio > 3.0
