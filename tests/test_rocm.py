"""Tests for ROCm (AMD GPU) support in numa_topology.

All tests mock rocm-smi subprocess calls — no AMD hardware required.
Covers: GPU counting, NUMA discovery, capability parsing, and auto-detect
fallback from nvidia-smi → rocm-smi.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from numa_topology import Topology, _count_gpus_rocm, _rocm_smi_pci, _sysfs_numa_rocm
from numa_topology.gpu_capabilities import (
    GPUCapability,
    _discover_rocm,
    _parse_rocm_bytes_to_mib,
    _parse_rocm_clock_mhz,
    _parse_rocm_smi_capabilities,
    discover_gpu_capabilities,
)


# ---------------------------------------------------------------------------
# Mock rocm-smi JSON fixtures
# ---------------------------------------------------------------------------

# RX 7900 XTX: 24 GB GDDR6, 96 CUs, 2615 MHz sclk
_RX7900XTX_JSON = json.dumps({
    "card0": {
        "PCI Bus": "0000:03:00.0",
        "Card Series": "AMD Radeon RX 7900 XTX",
        "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
        "VRAM Total Memory (B)": "25769803776",   # 24576 MiB
        "VRAM Total Used Memory (B)": "1073741824",
        "VRAM Free Memory (B)": "24696061952",    # 23552 MiB
        "Current clock speed for sclk": "2615 Mhz",
        "Current clock speed for mclk": "1249 Mhz",
        "CU Count": "96",
    },
    "system": {"Driver version": "6.0.0"},
})

# MI300X: 192 GB HBM3, 304 CUs, 2100 MHz sclk
_MI300X_JSON = json.dumps({
    "card0": {
        "PCI Bus": "0000:83:00.0",
        "Card Series": "AMD Instinct MI300X",
        "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
        "VRAM Total Memory (B)": "206158430208",   # 196608 MiB
        "VRAM Total Used Memory (B)": "4294967296",
        "VRAM Free Memory (B)": "201863462912",    # 192512 MiB
        "Current clock speed for sclk": "2100 Mhz",
        "Current clock speed for mclk": "3200 Mhz",
        "CU Count": "304",
    },
})

# Strix Halo APU: unified memory, "Strix Halo" in product name triggers unified_memory=True
_STRIX_HALO_JSON = json.dumps({
    "card0": {
        "PCI Bus": "0000:c4:00.0",
        "Card Series": "AMD Radeon 890M (Strix Halo)",
        "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
        "VRAM Total Memory (B)": "137438953472",   # 131072 MiB (128 GB unified)
        "VRAM Total Used Memory (B)": "1073741824",
        "VRAM Free Memory (B)": "136365211648",    # 130048 MiB
        "Current clock speed for sclk": "3100 Mhz",
        "Current clock speed for mclk": "0 Mhz",
        "CU Count": "40",
    },
})

# Two-card system: RX 7900 XTX + MI300X
_TWO_CARD_JSON = json.dumps({
    "card0": {
        "PCI Bus": "0000:03:00.0",
        "Card Series": "AMD Radeon RX 7900 XTX",
        "VRAM Total Memory (B)": "25769803776",
        "VRAM Free Memory (B)": "24696061952",
        "Current clock speed for sclk": "2615 Mhz",
        "CU Count": "96",
    },
    "card1": {
        "PCI Bus": "0000:83:00.0",
        "Card Series": "AMD Instinct MI300X",
        "VRAM Total Memory (B)": "206158430208",
        "VRAM Free Memory (B)": "201863462912",
        "Current clock speed for sclk": "2100 Mhz",
        "CU Count": "304",
    },
})


def _run_ok(stdout: str) -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stdout = stdout
    return r


def _run_fail(returncode: int = 1) -> MagicMock:
    r = MagicMock()
    r.returncode = returncode
    r.stdout = ""
    return r


# ---------------------------------------------------------------------------
# _parse_rocm_bytes_to_mib / _parse_rocm_clock_mhz helpers
# ---------------------------------------------------------------------------

def test_parse_rocm_bytes_to_mib_exact():
    assert _parse_rocm_bytes_to_mib("25769803776") == 24576

def test_parse_rocm_bytes_to_mib_integer():
    assert _parse_rocm_bytes_to_mib(25769803776) == 24576

def test_parse_rocm_bytes_to_mib_zero():
    assert _parse_rocm_bytes_to_mib("0") == 0

def test_parse_rocm_bytes_to_mib_bad_input():
    assert _parse_rocm_bytes_to_mib("N/A") == 0

def test_parse_rocm_clock_mhz_with_unit():
    assert _parse_rocm_clock_mhz("2615 Mhz") == 2615

def test_parse_rocm_clock_mhz_bare_int():
    assert _parse_rocm_clock_mhz("2100") == 2100

def test_parse_rocm_clock_mhz_bad_input():
    assert _parse_rocm_clock_mhz("N/A") == 0


# ---------------------------------------------------------------------------
# _parse_rocm_smi_capabilities
# ---------------------------------------------------------------------------

def test_parse_rx7900xtx_name():
    gpus = _parse_rocm_smi_capabilities(_RX7900XTX_JSON)
    assert len(gpus) == 1
    assert gpus[0].name == "AMD Radeon RX 7900 XTX"


def test_parse_rx7900xtx_vram():
    gpus = _parse_rocm_smi_capabilities(_RX7900XTX_JSON)
    g = gpus[0]
    assert g.memory_total_mib == 24576
    assert g.memory_free_mib == 23552


def test_parse_rx7900xtx_clock_and_cu():
    gpus = _parse_rocm_smi_capabilities(_RX7900XTX_JSON)
    g = gpus[0]
    assert g.max_sm_clock_mhz == 2615
    assert g.cu_count == 96


def test_parse_rx7900xtx_not_unified_memory():
    gpus = _parse_rocm_smi_capabilities(_RX7900XTX_JSON)
    assert gpus[0].unified_memory is False


def test_parse_mi300x_vram():
    gpus = _parse_rocm_smi_capabilities(_MI300X_JSON)
    g = gpus[0]
    assert g.memory_total_mib == 196608
    assert g.memory_free_mib == 192512


def test_parse_mi300x_cu_count():
    gpus = _parse_rocm_smi_capabilities(_MI300X_JSON)
    assert gpus[0].cu_count == 304


def test_parse_strix_halo_unified_memory():
    gpus = _parse_rocm_smi_capabilities(_STRIX_HALO_JSON)
    assert len(gpus) == 1
    assert gpus[0].unified_memory is True


def test_parse_strix_halo_cu_count():
    gpus = _parse_rocm_smi_capabilities(_STRIX_HALO_JSON)
    assert gpus[0].cu_count == 40


def test_parse_two_cards_count():
    gpus = _parse_rocm_smi_capabilities(_TWO_CARD_JSON)
    assert len(gpus) == 2


def test_parse_two_cards_indices():
    gpus = _parse_rocm_smi_capabilities(_TWO_CARD_JSON)
    indices = {g.index for g in gpus}
    assert indices == {0, 1}


def test_parse_skips_non_card_keys():
    # "system" key should be skipped
    gpus = _parse_rocm_smi_capabilities(_RX7900XTX_JSON)
    assert all(isinstance(g.index, int) for g in gpus)
    assert len(gpus) == 1


def test_parse_empty_json():
    gpus = _parse_rocm_smi_capabilities("{}")
    assert gpus == []


def test_parse_invalid_json():
    gpus = _parse_rocm_smi_capabilities("not json")
    assert gpus == []


def test_parse_compute_cap_is_na():
    gpus = _parse_rocm_smi_capabilities(_RX7900XTX_JSON)
    assert gpus[0].compute_cap == "N/A"


# ---------------------------------------------------------------------------
# Score normalisation for ROCm GPUs
# ---------------------------------------------------------------------------

def test_mi300x_scores_higher_than_rx7900xtx():
    gpus = _parse_rocm_smi_capabilities(_TWO_CARD_JSON)
    assert len(gpus) == 2
    by_name = {g.name: g for g in gpus}
    from numa_topology.gpu_capabilities import _normalise_scores
    gpus = _normalise_scores(gpus)
    by_name = {g.name: g for g in gpus}
    # MI300X: 2100 × 304 = 638400; RX 7900 XTX: 2615 × 96 = 251040
    assert by_name["AMD Instinct MI300X"].relative_compute_score == pytest.approx(1.0)
    assert by_name["AMD Radeon RX 7900 XTX"].relative_compute_score < 1.0


# ---------------------------------------------------------------------------
# _count_gpus_rocm
# ---------------------------------------------------------------------------

def test_count_gpus_rocm_single_card():
    with patch("subprocess.run", return_value=_run_ok(_RX7900XTX_JSON)):
        assert _count_gpus_rocm() == 1


def test_count_gpus_rocm_two_cards():
    with patch("subprocess.run", return_value=_run_ok(_TWO_CARD_JSON)):
        assert _count_gpus_rocm() == 2


def test_count_gpus_rocm_no_rocm_smi():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert _count_gpus_rocm() == 0


def test_count_gpus_rocm_nonzero_exit():
    with patch("subprocess.run", return_value=_run_fail(1)):
        assert _count_gpus_rocm() == 0


def test_count_gpus_rocm_skips_system_key():
    # _RX7900XTX_JSON contains a "system" key — should not be counted
    with patch("subprocess.run", return_value=_run_ok(_RX7900XTX_JSON)):
        assert _count_gpus_rocm() == 1


# ---------------------------------------------------------------------------
# _rocm_smi_pci
# ---------------------------------------------------------------------------

def test_rocm_smi_pci_returns_bus_id():
    bus_json = json.dumps({"card0": {"PCI Bus": "0000:03:00.0"}})
    with patch("subprocess.run", return_value=_run_ok(bus_json)):
        assert _rocm_smi_pci(0) == "0000:03:00.0"


def test_rocm_smi_pci_returns_none_on_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert _rocm_smi_pci(0) is None


def test_rocm_smi_pci_returns_none_on_bad_exit():
    with patch("subprocess.run", return_value=_run_fail(1)):
        assert _rocm_smi_pci(0) is None


def test_rocm_smi_pci_returns_none_for_unknown_index():
    bus_json = json.dumps({"card0": {"PCI Bus": "0000:03:00.0"}})
    with patch("subprocess.run", return_value=_run_ok(bus_json)):
        assert _rocm_smi_pci(5) is None


# ---------------------------------------------------------------------------
# _sysfs_numa_rocm
# ---------------------------------------------------------------------------

def test_sysfs_numa_rocm_returns_none_when_rocm_smi_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert _sysfs_numa_rocm(0) is None


# ---------------------------------------------------------------------------
# discover_gpu_capabilities — ROCm auto-detect fallback
# ---------------------------------------------------------------------------

def test_discover_tries_rocm_when_nvidia_smi_missing():
    with patch("subprocess.run", side_effect=[
        FileNotFoundError,              # nvidia-smi call raises
        _run_ok(_RX7900XTX_JSON),       # rocm-smi capabilities call
    ]):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1
    assert topo.gpus[0].name == "AMD Radeon RX 7900 XTX"


def test_discover_tries_rocm_when_nvidia_returns_empty():
    with patch("subprocess.run", side_effect=[
        _run_ok(""),                    # nvidia-smi returns 0 GPUs
        _run_ok(_MI300X_JSON),          # rocm-smi capabilities call
    ]):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1
    assert "MI300X" in topo.gpus[0].name


def test_discover_rocm_flat_fallback_when_both_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        topo = discover_gpu_capabilities()
    assert len(topo.gpus) == 1
    assert topo.gpus[0].relative_compute_score == pytest.approx(1.0)


def test_discover_rocm_topology_is_homogeneous_for_single_card():
    with patch("subprocess.run", side_effect=[
        FileNotFoundError,
        _run_ok(_RX7900XTX_JSON),
    ]):
        topo = discover_gpu_capabilities()
    assert topo.is_homogeneous is True


def test_discover_rocm_topology_heterogeneous_for_two_cards():
    with patch("subprocess.run", side_effect=[
        FileNotFoundError,
        _run_ok(_TWO_CARD_JSON),
    ]):
        topo = discover_gpu_capabilities()
    assert topo.is_homogeneous is False


# ---------------------------------------------------------------------------
# Topology.discover — ROCm path
# ---------------------------------------------------------------------------

def test_topology_discover_uses_rocm_when_cuda_absent():
    # _count_gpus returns 0 (no torch, nvidia-smi raises); _count_gpus_rocm finds 1
    with patch("numa_topology._count_gpus", return_value=0), \
         patch("numa_topology._count_gpus_rocm", return_value=1), \
         patch("numa_topology._sysfs_numa_rocm", return_value=0):
        topo = Topology.discover()
    assert 0 in topo.gpu_to_numa


def test_topology_discover_flat_when_both_absent():
    with patch("numa_topology._count_gpus", return_value=0), \
         patch("numa_topology._count_gpus_rocm", return_value=0):
        topo = Topology.discover()
    assert topo.gpu_to_numa == {0: 0}
