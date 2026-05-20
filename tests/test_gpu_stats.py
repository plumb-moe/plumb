"""Tests for GpuStatsPoller and compute_gpu_session_stats — no GPU hardware required."""
from __future__ import annotations

import io
import time
from unittest.mock import MagicMock, patch

import pytest

from plumb.exporters.gpu_stats import (
    GpuStatsPoller,
    compute_gpu_session_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dmon_output(*gpu_lines: str) -> str:
    header = "# gpu   sm  mem  enc  dec  jpg  ofa  pwr\n"
    return header + "\n".join(gpu_lines) + "\n"


def _make_proc(output: str, returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = io.StringIO(output)
    proc.returncode = returncode
    proc.wait.return_value = returncode
    proc.terminate = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# snapshot() with no data
# ---------------------------------------------------------------------------

def test_snapshot_empty_before_start():
    poller = GpuStatsPoller()
    assert poller.snapshot() == {}


def test_snapshot_missing_nvidia_smi():
    """FileNotFoundError → empty snapshot, no crash."""
    poller = GpuStatsPoller()
    with patch("subprocess.Popen", side_effect=FileNotFoundError):
        poller.start()
        time.sleep(0.1)
        poller.stop()
    assert poller.snapshot() == {}


def test_snapshot_permission_denied():
    poller = GpuStatsPoller()
    with patch("subprocess.Popen", side_effect=PermissionError("not allowed")):
        poller.start()
        time.sleep(0.1)
        poller.stop()
    assert poller.snapshot() == {}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_single_gpu():
    output = _dmon_output("    0   42   18    0    0    0    0  110")
    poller = GpuStatsPoller()
    with patch("subprocess.Popen", return_value=_make_proc(output)):
        poller.start()
        # Let the thread finish reading (process exits immediately after output)
        time.sleep(0.2)
        poller.stop()

    snap = poller.snapshot()
    assert 0 in snap
    assert len(snap[0]) == 1
    sample = snap[0][0]
    assert sample["sm"] == 42.0
    assert sample["mem"] == 18.0
    assert sample["pwr"] == 110.0


def test_parse_multi_gpu():
    output = _dmon_output(
        "    0   55   30    0    0    0    0  150",
        "    1   20   10    0    0    0    0   90",
    )
    poller = GpuStatsPoller()
    with patch("subprocess.Popen", return_value=_make_proc(output)):
        poller.start()
        time.sleep(0.2)
        poller.stop()

    snap = poller.snapshot()
    assert set(snap.keys()) == {0, 1}
    assert snap[0][0]["sm"] == 55.0
    assert snap[1][0]["sm"] == 20.0


def test_parse_multiple_samples():
    output = _dmon_output(
        "    0   10   5    0    0    0    0   80",
        "    0   20   8    0    0    0    0   85",
        "    0   30  12    0    0    0    0   90",
    )
    poller = GpuStatsPoller()
    with patch("subprocess.Popen", return_value=_make_proc(output)):
        poller.start()
        time.sleep(0.2)
        poller.stop()

    snap = poller.snapshot()
    # 3 lines → 3 frames, each with gpu 0 → 3 samples in gpu 0's list
    assert len(snap[0]) == 3
    assert [s["sm"] for s in snap[0]] == [10.0, 20.0, 30.0]


def test_rolling_buffer_respects_window():
    """Buffer keeps at most window_seconds samples per start/stop cycle."""
    output = _dmon_output(*[f"    0  {i:3d}   0    0    0    0    0  100" for i in range(20)])
    poller = GpuStatsPoller(window_seconds=5)
    with patch("subprocess.Popen", return_value=_make_proc(output)):
        poller.start()
        time.sleep(0.2)
        poller.stop()

    snap = poller.snapshot()
    assert len(snap[0]) <= 5


# ---------------------------------------------------------------------------
# Restart-once on crash
# ---------------------------------------------------------------------------

def test_crash_restarts_once_then_disables():
    """Process exits immediately (simulated crash) → restarts once then disables."""
    call_count = {"n": 0}

    def fake_popen(*args, **kwargs):
        call_count["n"] += 1
        proc = MagicMock()
        proc.stdout = io.StringIO("")  # empty → immediate EOF
        proc.wait.return_value = 1
        proc.terminate = MagicMock()
        return proc

    poller = GpuStatsPoller()
    with patch("subprocess.Popen", side_effect=fake_popen):
        poller.start()
        time.sleep(0.3)
        poller.stop()

    assert call_count["n"] == 2  # launched twice
    assert poller._disabled is True


# ---------------------------------------------------------------------------
# start() / stop() idempotency
# ---------------------------------------------------------------------------

def test_start_idempotent():
    poller = GpuStatsPoller()
    with patch("subprocess.Popen", side_effect=FileNotFoundError):
        poller.start()
        poller.start()  # second call is a no-op
        time.sleep(0.1)
        poller.stop()


def test_stop_before_start_is_safe():
    poller = GpuStatsPoller()
    poller.stop()  # should not raise


# ---------------------------------------------------------------------------
# compute_gpu_session_stats — bx-gfi
# ---------------------------------------------------------------------------

def test_session_stats_empty_buffer():
    stats = compute_gpu_session_stats({})
    assert stats.per_gpu == []
    assert stats.cluster_mean_sm_utilisation == 0.0
    assert stats.imbalance_confirmed is False


def test_session_stats_single_gpu():
    buf = {0: [{"sm": 40.0, "mem": 20.0, "pwr": 100.0},
               {"sm": 60.0, "mem": 25.0, "pwr": 110.0},
               {"sm": 80.0, "mem": 30.0, "pwr": 120.0}]}
    stats = compute_gpu_session_stats(buf)
    assert len(stats.per_gpu) == 1
    g = stats.per_gpu[0]
    assert g.gpu_index == 0
    assert g.sm_mean == pytest.approx(60.0, abs=0.1)
    assert g.sm_p50 == pytest.approx(60.0, abs=0.1)
    assert g.sm_p90 == pytest.approx(76.0, abs=1.0)
    assert g.sm_p95 == pytest.approx(78.0, abs=1.0)
    assert g.sm_peak == 80.0
    assert g.mem_mean == pytest.approx(25.0, abs=0.1)
    assert g.pwr_mean == pytest.approx(110.0, abs=0.1)
    assert stats.cluster_mean_sm_utilisation == pytest.approx(60.0, abs=0.1)


def test_session_stats_balanced_no_imbalance():
    buf = {
        0: [{"sm": 50.0, "mem": 20.0, "pwr": 100.0}],
        1: [{"sm": 52.0, "mem": 21.0, "pwr": 101.0}],
        2: [{"sm": 48.0, "mem": 19.0, "pwr": 99.0}],
        3: [{"sm": 50.0, "mem": 20.0, "pwr": 100.0}],
    }
    stats = compute_gpu_session_stats(buf)
    assert stats.imbalance_confirmed is False
    assert stats.cluster_mean_sm_utilisation == pytest.approx(50.0, abs=1.0)


def test_session_stats_imbalanced_confirmed():
    # GPU 0 at 90%, others at 10% → std dev >> 15pp
    buf = {
        0: [{"sm": 90.0, "mem": 80.0, "pwr": 200.0}],
        1: [{"sm": 10.0, "mem": 10.0, "pwr": 80.0}],
        2: [{"sm": 10.0, "mem": 10.0, "pwr": 80.0}],
        3: [{"sm": 10.0, "mem": 10.0, "pwr": 80.0}],
    }
    stats = compute_gpu_session_stats(buf)
    assert stats.imbalance_confirmed is True


def test_session_stats_missing_fields_graceful():
    """Samples with missing keys are skipped for that metric."""
    buf = {0: [{"sm": 55.0}, {"sm": 45.0, "mem": 10.0}]}
    stats = compute_gpu_session_stats(buf)
    assert stats.per_gpu[0].sm_mean == pytest.approx(50.0, abs=0.1)
    # mem only present in one sample
    assert stats.per_gpu[0].mem_mean == pytest.approx(10.0, abs=0.1)


def test_session_stats_gpu_indices_ordered():
    buf = {3: [{"sm": 30.0}], 1: [{"sm": 10.0}], 0: [{"sm": 0.0}]}
    stats = compute_gpu_session_stats(buf)
    assert [g.gpu_index for g in stats.per_gpu] == [0, 1, 3]
