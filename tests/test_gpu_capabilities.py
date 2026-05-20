"""Tests for numa_topology/gpu_capabilities.py — no GPU hardware required."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from numa_topology.gpu_capabilities import (
    _build_topology,
    _normalise_scores,
    _parse_csv,
    discover_gpu_capabilities,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ONE_GPU_CSV = "0, NVIDIA A100-SXM4-80GB, 81251 MiB, 79000 MiB, 8.0, 1410 MHz, 1593 MHz\n"

_TWO_GPU_HOMO_CSV = (
    "0, NVIDIA H100 SXM5 80GB, 81251 MiB, 78000 MiB, 9.0, 1980 MHz, 3201 MHz\n"
    "1, NVIDIA H100 SXM5 80GB, 81251 MiB, 78000 MiB, 9.0, 1980 MHz, 3201 MHz\n"
)

_TWO_GPU_HETERO_CSV = (
    "0, NVIDIA H100 SXM5 80GB, 81251 MiB, 78000 MiB, 9.0, 1980 MHz, 3201 MHz\n"
    "1, NVIDIA A100-SXM4-80GB, 81251 MiB, 79000 MiB, 8.0, 1410 MHz, 1593 MHz\n"
)

# AC-specified fixtures
_RTX3090_SYMMETRIC_CSV = (
    "0, NVIDIA GeForce RTX 3090, 24576 MiB, 22000 MiB, 8.6, 1695 MHz, 9751 MHz\n"
    "1, NVIDIA GeForce RTX 3090, 24576 MiB, 22000 MiB, 8.6, 1695 MHz, 9751 MHz\n"
    "2, NVIDIA GeForce RTX 3090, 24576 MiB, 22000 MiB, 8.6, 1695 MHz, 9751 MHz\n"
    "3, NVIDIA GeForce RTX 3090, 24576 MiB, 22000 MiB, 8.6, 1695 MHz, 9751 MHz\n"
)

_HETERO_3060_1080_1050_CSV = (
    "0, NVIDIA GeForce RTX 3060, 12288 MiB, 10000 MiB, 8.6, 1777 MHz, 7501 MHz\n"
    "1, NVIDIA GeForce GTX 1080, 8192 MiB,  7000 MiB, 6.1, 1733 MHz, 5005 MHz\n"
    "2, NVIDIA GeForce GTX 1050, 4096 MiB,  3500 MiB, 6.1, 1392 MHz, 3504 MHz\n"
)


def _make_run_result(stdout: str, returncode: int = 0) -> MagicMock:
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    return r


# ---------------------------------------------------------------------------
# _parse_csv
# ---------------------------------------------------------------------------

def test_parse_single_gpu():
    gpus = _parse_csv(_ONE_GPU_CSV)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.index == 0
    assert "A100" in g.name
    assert g.memory_total_mib == 81251
    assert g.memory_free_mib == 79000
    assert g.compute_cap == "8.0"
    assert g.max_sm_clock_mhz == 1410
    assert g.max_mem_clock_mhz == 1593


def test_parse_two_gpus():
    gpus = _parse_csv(_TWO_GPU_HETERO_CSV)
    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[1].index == 1


def test_parse_malformed_line_skipped():
    bad = "0, TooFewFields\n"
    gpus = _parse_csv(bad)
    assert gpus == []


def test_parse_empty_output():
    assert _parse_csv("") == []
    assert _parse_csv("   \n\n") == []


# ---------------------------------------------------------------------------
# _normalise_scores
# ---------------------------------------------------------------------------

def test_normalise_fastest_is_1():
    gpus = _parse_csv(_TWO_GPU_HETERO_CSV)
    gpus = _normalise_scores(gpus)
    scores = [g.relative_compute_score for g in gpus]
    assert max(scores) == pytest.approx(1.0)


def test_normalise_slower_gpu_below_1():
    gpus = _parse_csv(_TWO_GPU_HETERO_CSV)
    gpus = _normalise_scores(gpus)
    # H100 (index 0) is faster; A100 (index 1) should be < 1
    assert gpus[0].relative_compute_score == pytest.approx(1.0)
    assert gpus[1].relative_compute_score < 1.0


def test_normalise_homogeneous_all_1():
    gpus = _parse_csv(_TWO_GPU_HOMO_CSV)
    gpus = _normalise_scores(gpus)
    assert all(g.relative_compute_score == pytest.approx(1.0) for g in gpus)


# ---------------------------------------------------------------------------
# _build_topology
# ---------------------------------------------------------------------------

def test_homogeneous_flag():
    gpus = _normalise_scores(_parse_csv(_TWO_GPU_HOMO_CSV))
    topo = _build_topology(gpus)
    assert topo.is_homogeneous is True


def test_heterogeneous_flag():
    gpus = _normalise_scores(_parse_csv(_TWO_GPU_HETERO_CSV))
    topo = _build_topology(gpus)
    assert topo.is_homogeneous is False


def test_compute_score_range():
    gpus = _normalise_scores(_parse_csv(_TWO_GPU_HETERO_CSV))
    topo = _build_topology(gpus)
    lo, hi = topo.compute_score_range
    assert hi == pytest.approx(1.0)
    assert lo < hi


def test_mixed_vendor_false_for_nvidia_only():
    gpus = _normalise_scores(_parse_csv(_TWO_GPU_HETERO_CSV))
    topo = _build_topology(gpus)
    assert topo.mixed_vendor is False


def test_mixed_vendor_detected():
    csv = (
        "0, NVIDIA H100 SXM5 80GB, 81251 MiB, 78000 MiB, 9.0, 1980 MHz, 3201 MHz\n"
        "1, AMD Instinct MI300X, 192000 MiB, 190000 MiB, 9.4, 2100 MHz, 2500 MHz\n"
    )
    gpus = _normalise_scores(_parse_csv(csv))
    topo = _build_topology(gpus)
    assert topo.mixed_vendor is True


# ---------------------------------------------------------------------------
# discover_gpu_capabilities — integration + fallback
# ---------------------------------------------------------------------------

def test_discover_returns_topology_on_success():
    with patch("subprocess.run", return_value=_make_run_result(_TWO_GPU_HETERO_CSV)):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 2
    assert topo.is_homogeneous is False


def test_discover_fallback_on_missing_nvidia_smi():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1
    assert topo.gpus[0].relative_compute_score == pytest.approx(1.0)
    assert topo.is_homogeneous is True


def test_discover_fallback_on_permission_error():
    with patch("subprocess.run", side_effect=PermissionError("denied")):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1


def test_discover_fallback_on_nonzero_exit():
    with patch("subprocess.run", return_value=_make_run_result("", returncode=1)):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1


def test_discover_fallback_on_empty_output():
    with patch("subprocess.run", return_value=_make_run_result("")):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1


def test_discover_normalises_scores():
    with patch("subprocess.run", return_value=_make_run_result(_TWO_GPU_HETERO_CSV)):
        topo = discover_gpu_capabilities()
    scores = [g.relative_compute_score for g in topo.gpus]
    assert max(scores) == pytest.approx(1.0)


def test_discover_single_gpu():
    with patch("subprocess.run", return_value=_make_run_result(_ONE_GPU_CSV)):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1
    assert topo.gpus[0].relative_compute_score == pytest.approx(1.0)
    assert topo.is_homogeneous is True


# ---------------------------------------------------------------------------
# AC-specified fixture: symmetric RTX 3090 cluster
# ---------------------------------------------------------------------------

def test_symmetric_rtx3090_homogeneous():
    gpus = _normalise_scores(_parse_csv(_RTX3090_SYMMETRIC_CSV))
    topo = _build_topology(gpus)
    assert topo.is_homogeneous is True
    assert all(g.relative_compute_score == pytest.approx(1.0) for g in topo.gpus)
    assert topo.mixed_vendor is False
    assert topo.compute_score_range == pytest.approx((1.0, 1.0))


def test_symmetric_rtx3090_discover():
    with patch("subprocess.run", return_value=_make_run_result(_RTX3090_SYMMETRIC_CSV)):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 4
    assert topo.is_homogeneous is True


# ---------------------------------------------------------------------------
# AC-specified fixture: heterogeneous 3060 + 1080 + 1050
# ---------------------------------------------------------------------------

def test_hetero_3060_1080_1050_scores_ordered():
    gpus = _normalise_scores(_parse_csv(_HETERO_3060_1080_1050_CSV))
    # 3060 has highest SM clock × compute_cap → should score 1.0
    scores_by_name = {g.name: g.relative_compute_score for g in gpus}
    assert scores_by_name["NVIDIA GeForce RTX 3060"] == pytest.approx(1.0)
    assert scores_by_name["NVIDIA GeForce GTX 1080"] < 1.0
    assert scores_by_name["NVIDIA GeForce GTX 1050"] < scores_by_name["NVIDIA GeForce GTX 1080"]


def test_hetero_3060_1080_1050_not_homogeneous():
    gpus = _normalise_scores(_parse_csv(_HETERO_3060_1080_1050_CSV))
    topo = _build_topology(gpus)
    assert topo.is_homogeneous is False
    assert topo.mixed_vendor is False  # all NVIDIA
    lo, hi = topo.compute_score_range
    assert hi == pytest.approx(1.0)
    assert lo < 1.0
