import pytest

from plumb.analysis.imbalance import compute_imbalance
from plumb.counter import ActivationCounter


def _counter_with(data: dict[tuple[int, int], int]) -> ActivationCounter:
    c = ActivationCounter(window_size=10_000)
    for (layer, expert), count in data.items():
        c.record(layer, expert, count)
    return c


def test_perfect_balance():
    c = _counter_with({(0, e): 100 for e in range(8)})
    result = compute_imbalance(c)
    assert len(result) == 1
    assert abs(result[0].imbalance_ratio - 1.0) < 1e-3


def test_total_imbalance():
    # One expert gets all tokens
    data = {(0, e): (1000 if e == 0 else 0) for e in range(8)}
    data[(0, 0)] = 1000
    for e in range(1, 8):
        data[(0, e)] = 1
    c = _counter_with(data)
    result = compute_imbalance(c)
    assert result[0].imbalance_ratio > 1.0
    assert result[0].max_expert_id == 0


def test_multiple_layers():
    c = ActivationCounter()
    for layer in range(4):
        for expert in range(8):
            c.record(layer, expert, 10 + expert)  # slight imbalance
    result = compute_imbalance(c)
    assert len(result) == 4
    for r in result:
        assert r.imbalance_ratio >= 1.0


def test_empty_counter():
    c = ActivationCounter()
    assert compute_imbalance(c) == []


def test_ratio_formula():
    # Experts: loads [10, 10, 10, 10, 10, 10, 10, 30]
    # mean=12.5, max=30, ratio=2.4
    data = {(0, e): 10 for e in range(7)}
    data[(0, 7)] = 30
    c = _counter_with(data)
    result = compute_imbalance(c)
    assert abs(result[0].imbalance_ratio - 2.4) < 0.01
