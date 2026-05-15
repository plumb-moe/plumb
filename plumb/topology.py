"""Thin re-export — Topology lives in the standalone numa_topology package."""
from numa_topology import (  # noqa: F401
    Topology,
    _nvidia_smi_pci,
    _pci_path_variants,
    _sysfs_numa,
)

__all__ = ["Topology", "_nvidia_smi_pci", "_pci_path_variants", "_sysfs_numa"]
