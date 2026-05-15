import threading

import pytest

from plumb.counter import ActivationCounter


def test_record_and_snapshot():
    c = ActivationCounter(window_size=100)
    c.record(0, 0, 10)
    c.record(0, 1, 5)
    c.record(1, 0, 3)
    snap = c.snapshot()
    assert snap[(0, 0)] == 10
    assert snap[(0, 1)] == 5
    assert snap[(1, 0)] == 3


def test_rolling_window():
    c = ActivationCounter(window_size=3)
    for i in range(5):
        c.record(0, 0, i + 1)  # 1,2,3,4,5 — window keeps last 3: 3,4,5
    assert c.snapshot()[(0, 0)] == 12  # 3+4+5


def test_pass_count():
    c = ActivationCounter()
    c.increment_pass()
    c.increment_pass()
    assert c.pass_count == 2


def test_thread_safety():
    c = ActivationCounter(window_size=10_000)
    errors: list[Exception] = []

    def worker(layer: int) -> None:
        try:
            for i in range(500):
                c.record(layer, i % 8)
                c.increment_pass()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    snap = c.snapshot()
    assert len(snap) > 0


def test_reset():
    c = ActivationCounter()
    c.record(0, 0)
    c.increment_pass()
    c.reset()
    assert c.snapshot() == {}
    assert c.pass_count == 0
