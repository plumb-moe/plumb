import threading

import pytest

from plumb.counter import ActivationCounter


# ---------------------------------------------------------------------------
# recording and accumulation
# ---------------------------------------------------------------------------

def test_single_record_appears_in_snapshot():
    c = ActivationCounter(window_size=100)
    c.record(layer_id=0, expert_id=3, token_count=42)
    assert c.snapshot()[(0, 3)] == 42


def test_multiple_records_to_same_expert_accumulate():
    c = ActivationCounter(window_size=100)
    c.record(0, 0, 10)
    c.record(0, 0, 25)
    assert c.snapshot()[(0, 0)] == 35


def test_default_token_count_is_one():
    c = ActivationCounter(window_size=100)
    c.record(0, 0)
    assert c.snapshot()[(0, 0)] == 1


def test_different_layer_expert_pairs_are_independent():
    c = ActivationCounter(window_size=100)
    c.record(0, 0, 10)
    c.record(0, 1, 20)
    c.record(1, 0, 30)
    snap = c.snapshot()
    assert snap[(0, 0)] == 10
    assert snap[(0, 1)] == 20
    assert snap[(1, 0)] == 30


def test_unrecorded_keys_absent_from_snapshot():
    c = ActivationCounter(window_size=100)
    c.record(0, 0, 5)
    snap = c.snapshot()
    assert (0, 1) not in snap
    assert len(snap) == 1


# ---------------------------------------------------------------------------
# rolling window
# ---------------------------------------------------------------------------

def test_window_evicts_oldest_observations():
    # window_size=3: inserts 1,2,3,4,5 → window holds [3,4,5]
    c = ActivationCounter(window_size=3)
    for val in [1, 2, 3, 4, 5]:
        c.record(0, 0, val)
    assert c.snapshot()[(0, 0)] == 12   # 3+4+5


def test_full_window_reflects_recent_entries_only():
    c = ActivationCounter(window_size=2)
    c.record(0, 0, 100)   # will be evicted
    c.record(0, 0, 10)
    c.record(0, 0, 20)
    assert c.snapshot()[(0, 0)] == 30


def test_window_independent_per_expert():
    c = ActivationCounter(window_size=2)
    c.record(0, 0, 100)
    c.record(0, 0, 200)   # (0,0) full
    c.record(0, 1, 50)    # (0,1) unaffected
    assert c.snapshot()[(0, 1)] == 50


# ---------------------------------------------------------------------------
# pass tracking
# ---------------------------------------------------------------------------

def test_pass_count_starts_at_zero():
    assert ActivationCounter().pass_count == 0


def test_each_increment_adds_one():
    c = ActivationCounter()
    for expected in range(1, 6):
        c.increment_pass()
        assert c.pass_count == expected


# ---------------------------------------------------------------------------
# layer_ids
# ---------------------------------------------------------------------------

def test_layer_ids_returns_sorted_unique_layers():
    c = ActivationCounter()
    c.record(2, 0, 1)
    c.record(0, 0, 1)
    c.record(2, 1, 1)   # duplicate layer
    c.record(5, 0, 1)
    assert c.layer_ids() == [0, 2, 5]


def test_layer_ids_empty_on_fresh_counter():
    assert ActivationCounter().layer_ids() == []


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_snapshot_and_pass_count():
    c = ActivationCounter()
    c.record(0, 0, 50)
    c.increment_pass()
    c.reset()
    assert c.snapshot() == {}
    assert c.pass_count == 0


def test_recording_resumes_normally_after_reset():
    c = ActivationCounter()
    c.record(0, 0, 10)
    c.reset()
    c.record(0, 0, 99)
    assert c.snapshot()[(0, 0)] == 99


# ---------------------------------------------------------------------------
# thread safety
# ---------------------------------------------------------------------------

def test_concurrent_recording_does_not_corrupt_state():
    c = ActivationCounter(window_size=100_000)
    errors: list[Exception] = []

    def worker(layer_id: int) -> None:
        try:
            for i in range(200):
                c.record(layer_id, i % 8, token_count=1)
                c.increment_pass()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(c.snapshot()) > 0
    assert c.pass_count == 8 * 200
