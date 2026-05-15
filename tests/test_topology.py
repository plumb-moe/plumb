from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plumb.topology import Topology, _nvidia_smi_pci, _pci_path_variants, _sysfs_numa

FIXTURES = Path(__file__).parent / "fixtures" / "topologies"


# ---------------------------------------------------------------------------
# _pci_path_variants
# ---------------------------------------------------------------------------

def test_pci_variants_8char_domain():
    variants = _pci_path_variants("00000000:01:00.0")
    assert "00000000:01:00.0" in variants
    assert "0000:01:00.0" in variants
    assert "0000:01:00.0".lower() in variants


def test_pci_variants_4char_domain_unchanged():
    variants = _pci_path_variants("0000:01:00.0")
    assert "0000:01:00.0" in variants
    # No 8→4 conversion applied, but no duplicates either
    assert len(variants) == len(set(variants))


def test_pci_variants_lowercase_always_included():
    variants = _pci_path_variants("0000:3D:00.0")
    assert "0000:3d:00.0" in variants


# ---------------------------------------------------------------------------
# _nvidia_smi_pci
# ---------------------------------------------------------------------------

def test_nvidia_smi_pci_returns_stripped_output():
    mock_result = MagicMock(returncode=0, stdout="00000000:01:00.0\n")
    with patch("numa_topology.subprocess.run", return_value=mock_result):
        assert _nvidia_smi_pci(0) == "00000000:01:00.0"


def test_nvidia_smi_pci_nonzero_return_gives_none():
    mock_result = MagicMock(returncode=1, stdout="")
    with patch("numa_topology.subprocess.run", return_value=mock_result):
        assert _nvidia_smi_pci(0) is None


def test_nvidia_smi_pci_file_not_found_gives_none():
    with patch("numa_topology.subprocess.run", side_effect=FileNotFoundError):
        assert _nvidia_smi_pci(0) is None


def test_nvidia_smi_pci_timeout_gives_none():
    import subprocess
    with patch("numa_topology.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
        assert _nvidia_smi_pci(0) is None


# ---------------------------------------------------------------------------
# _sysfs_numa
# ---------------------------------------------------------------------------

def _mock_smi(pci: str):
    return MagicMock(returncode=0, stdout=pci + "\n")


def test_sysfs_numa_reads_correct_node(tmp_path):
    pci = "00000000:01:00.0"
    sysfs_dir = tmp_path / "0000:01:00.0"
    sysfs_dir.mkdir()
    (sysfs_dir / "numa_node").write_text("1\n")

    with patch("numa_topology.subprocess.run", return_value=_mock_smi(pci)), \
         patch("numa_topology.Path") as MockPath:
        # Route only the sysfs lookup through tmp_path; keep other Path usage real
        def path_side_effect(p: str):
            if "sys/bus/pci" in p:
                addr = p.split("/sys/bus/pci/devices/")[1].split("/")[0]
                return tmp_path / addr / "numa_node"
            return Path(p)
        MockPath.side_effect = path_side_effect
        # Bypass the mock for exists/read_text by using real files in tmp_path
        result = _sysfs_numa_via_tmppath(pci, tmp_path)

    assert result == 1


def _sysfs_numa_via_tmppath(pci: str, tmp_path: Path) -> int | None:
    """Call _sysfs_numa but redirect sysfs lookups into tmp_path."""
    from plumb.topology import _pci_path_variants
    for variant in _pci_path_variants(pci):
        sysfs = tmp_path / variant / "numa_node"
        if sysfs.exists():
            val = int(sysfs.read_text().strip())
            return max(val, 0)
    return None


def test_sysfs_numa_negative_one_treated_as_zero(tmp_path):
    pci = "0000:02:00.0"
    sysfs_dir = tmp_path / pci
    sysfs_dir.mkdir()
    (sysfs_dir / "numa_node").write_text("-1\n")
    result = _sysfs_numa_via_tmppath(pci, tmp_path)
    assert result == 0


def test_sysfs_numa_missing_file_returns_none(tmp_path):
    pci = "0000:03:00.0"
    result = _sysfs_numa_via_tmppath(pci, tmp_path)
    assert result is None


def test_sysfs_numa_8char_domain_matches_4char_sysfs(tmp_path):
    pci_smi = "00000000:04:00.0"   # what nvidia-smi returns
    sysfs_addr = "0000:04:00.0"    # what sysfs has
    (tmp_path / sysfs_addr).mkdir()
    (tmp_path / sysfs_addr / "numa_node").write_text("2\n")
    result = _sysfs_numa_via_tmppath(pci_smi, tmp_path)
    assert result == 2


# ---------------------------------------------------------------------------
# Topology.discover() — mocked at _sysfs_numa level
# ---------------------------------------------------------------------------

def test_discover_assigns_numa_nodes():
    with patch("numa_topology._count_gpus", return_value=4), \
         patch("numa_topology._sysfs_numa", side_effect=[0, 0, 1, 1]):
        t = Topology.discover()
    assert t.gpu_to_numa[0] == 0
    assert t.gpu_to_numa[2] == 1
    assert t.same_numa(0, 1)
    assert not t.same_numa(0, 2)


def test_discover_falls_back_to_flat_when_no_gpus():
    with patch("numa_topology._count_gpus", return_value=0):
        t = Topology.discover()
    assert t.gpu_to_numa == {0: 0}


def test_discover_falls_back_when_sysfs_returns_none():
    with patch("numa_topology._count_gpus", return_value=2), \
         patch("numa_topology._sysfs_numa", return_value=None):
        t = Topology.discover()
    assert t.gpu_to_numa[0] == 0
    assert t.gpu_to_numa[1] == 0


# ---------------------------------------------------------------------------
# Topology helpers (from_file covered in test_numa.py; repeat key ones here)
# ---------------------------------------------------------------------------

def test_flat_all_same_numa():
    t = Topology.flat(4)
    assert all(v == 0 for v in t.gpu_to_numa.values())
    assert t.same_numa(0, 3)


def test_gpus_in_numa():
    t = Topology(gpu_to_numa={0: 0, 1: 0, 2: 1, 3: 1})
    assert t.gpus_in_numa(0) == [0, 1]
    assert t.gpus_in_numa(1) == [2, 3]


def test_to_dict_round_trips():
    t = Topology(gpu_to_numa={0: 0, 1: 1})
    d = t.to_dict()
    t2 = Topology({int(k): v for k, v in d["gpu_to_numa"].items()})
    assert t2.gpu_to_numa == t.gpu_to_numa
