"""numa-topology — discover GPU NUMA affinity with zero dependencies.

Reads /sys/bus/pci/devices/<pci-id>/numa_node for each GPU visible to
nvidia-smi.  Falls back gracefully when running without GPUs or on
non-NUMA machines.

Typical usage::

    from numa_topology import Topology

    topology = Topology.discover()
    print(topology.gpu_to_numa)   # {0: 0, 1: 0, 2: 1, 3: 1}
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["Topology", "GPUCapability", "HeterogeneousTopology", "discover_gpu_capabilities"]

from .gpu_capabilities import (  # noqa: E402
    GPUCapability,
    HeterogeneousTopology,
    discover_gpu_capabilities,
)

logger = logging.getLogger(__name__)


class Topology:
    """Maps GPU device index to NUMA node.

    Attributes:
        gpu_to_numa: ``{gpu_index: numa_node}`` mapping.
    """

    def __init__(self, gpu_to_numa: dict[int, int]) -> None:
        self.gpu_to_numa = gpu_to_numa

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls) -> "Topology":
        """Return a Topology by reading sysfs NUMA affinity for each GPU.

        Requires ``nvidia-smi`` on ``$PATH``.  Falls back to a flat
        single-node topology when GPUs are unavailable or NUMA info cannot
        be read.

        Works without PyTorch installed — falls back to querying
        ``nvidia-smi`` directly for the GPU count.
        """
        num_gpus = _count_gpus()
        if num_gpus == 0:
            return cls.flat(0)

        gpu_to_numa: dict[int, int] = {}
        for idx in range(num_gpus):
            node = _sysfs_numa(idx)
            gpu_to_numa[idx] = node if node is not None else 0

        logger.info("Topology discovered: %s", gpu_to_numa)
        return cls(gpu_to_numa)

    @classmethod
    def flat(cls, num_gpus: int) -> "Topology":
        """Return a topology where every GPU is on NUMA node 0."""
        return cls({i: 0 for i in range(max(num_gpus, 1))})

    @classmethod
    def from_file(cls, path: Path) -> "Topology":
        """Load a topology from a JSON file (``{"gpu_to_numa": {"0": 0, ...}}``)."""
        data = json.loads(Path(path).read_text())
        return cls({int(k): int(v) for k, v in data["gpu_to_numa"].items()})

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def numa_nodes(self) -> list[int]:
        """Sorted list of distinct NUMA node IDs."""
        return sorted(set(self.gpu_to_numa.values()))

    def gpus_in_numa(self, node: int) -> list[int]:
        """Sorted list of GPU indices assigned to *node*."""
        return sorted(g for g, n in self.gpu_to_numa.items() if n == node)

    def same_numa(self, gpu_a: int, gpu_b: int) -> bool:
        """Return True if both GPUs share a NUMA node."""
        return self.gpu_to_numa.get(gpu_a, 0) == self.gpu_to_numa.get(gpu_b, 0)

    def to_dict(self) -> dict:
        """Serialisable representation (string keys for JSON compat)."""
        return {"gpu_to_numa": {str(k): v for k, v in self.gpu_to_numa.items()}}

    def __repr__(self) -> str:
        return f"Topology({self.gpu_to_numa!r})"


# ------------------------------------------------------------------
# Internal helpers (exported for introspection / testing)
# ------------------------------------------------------------------

def _count_gpus() -> int:
    """Return the number of CUDA GPUs visible on this machine.

    Tries ``torch.cuda.device_count()`` first; falls back to parsing
    ``nvidia-smi`` output so the package works without PyTorch.
    """
    try:
        import torch  # optional
        return torch.cuda.device_count()
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        pass

    return 0


def _sysfs_numa(gpu_idx: int) -> int | None:
    """Read the NUMA node for *gpu_idx* from sysfs. Returns None on failure."""
    pci = _nvidia_smi_pci(gpu_idx)
    if pci is None:
        return None
    for variant in _pci_path_variants(pci):
        sysfs = Path(f"/sys/bus/pci/devices/{variant}/numa_node")
        if sysfs.exists():
            try:
                val = int(sysfs.read_text().strip())
                return max(val, 0)  # -1 → no NUMA affinity → treat as node 0
            except ValueError:
                pass
    logger.debug("No numa_node sysfs entry for GPU %d (pci=%s)", gpu_idx, pci)
    return None


def _nvidia_smi_pci(gpu_idx: int) -> str | None:
    """Return the PCI bus ID for *gpu_idx* from nvidia-smi, or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", f"--id={gpu_idx}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("nvidia-smi unavailable: %s", e)
        return None


def _pci_path_variants(bus_id: str) -> list[str]:
    """nvidia-smi emits 8-char domain (00000000:01:00.0); sysfs uses 4-char (0000:01:00.0)."""
    variants: list[str] = [bus_id, bus_id.lower()]
    parts = bus_id.split(":")
    if len(parts) == 3 and len(parts[0]) == 8:
        short = parts[0][4:] + ":" + ":".join(parts[1:])
        variants += [short, short.lower()]
    return list(dict.fromkeys(variants))  # deduplicate, preserve order
