"""PCIe topology discovery — link speed, width, and theoretical bandwidth per GPU.

Reads /sys/bus/pci/devices/<bus-id>/current_link_speed and current_link_width.
Falls back gracefully when running without GPUs or on machines without sysfs.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

from . import _count_gpus, _nvidia_smi_pci, _pci_path_variants

logger = logging.getLogger(__name__)

# Unidirectional GB/s per lane for each PCIe generation transfer rate.
# Gen 1-2: 8b/10b encoding (80% efficiency).
# Gen 3-5: 128b/130b encoding (~98.5% efficiency).
# Gen 6: 242b/256b PAM4 encoding (~94.5% efficiency).
_SPEED_GBS_PER_LANE: dict[float, float] = {
    2.5:  0.250,
    5.0:  0.500,
    8.0:  0.985,
    16.0: 1.969,
    32.0: 3.938,
    64.0: 7.563,
}

_SYSFS_PCI = Path("/sys/bus/pci/devices")


@dataclass
class GPUPCIeInfo:
    gpu_idx: int
    bus_id: str
    link_speed_gts: float
    link_width: int
    theoretical_bw_gbs: float
    nvlink: bool = False


@dataclass
class PCIeTopology:
    gpus: list[GPUPCIeInfo]
    is_symmetric: bool
    min_bw_gpu: int
    max_bw_gpu: int
    bandwidth_ratio: float

    @classmethod
    def discover(cls, sysfs_root: Path = _SYSFS_PCI) -> "PCIeTopology":
        """Return a PCIeTopology by reading sysfs link speed/width for each GPU.

        Requires nvidia-smi on $PATH.  Falls back to a flat symmetric topology
        when GPUs are unavailable or sysfs cannot be read.
        """
        num_gpus = _count_gpus()
        if num_gpus == 0:
            warnings.warn(
                "No GPUs detected; returning flat PCIe topology",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls._flat_fallback(0)

        gpus: list[GPUPCIeInfo] = []
        for idx in range(num_gpus):
            info = _read_pcie_info(idx, sysfs_root)
            if info is not None:
                gpus.append(info)

        if not gpus:
            warnings.warn(
                "Could not read PCIe link info for any GPU; returning flat topology",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls._flat_fallback(num_gpus)

        return cls._from_gpu_list(gpus)

    @classmethod
    def _flat_fallback(cls, num_gpus: int) -> "PCIeTopology":
        n = max(num_gpus, 1)
        gpus = [
            GPUPCIeInfo(gpu_idx=i, bus_id="", link_speed_gts=0.0,
                        link_width=0, theoretical_bw_gbs=0.0)
            for i in range(n)
        ]
        return cls(gpus=gpus, is_symmetric=True, min_bw_gpu=0, max_bw_gpu=0, bandwidth_ratio=1.0)

    @classmethod
    def _from_gpu_list(cls, gpus: list[GPUPCIeInfo]) -> "PCIeTopology":
        bandwidths = [g.theoretical_bw_gbs for g in gpus]
        min_bw = min(bandwidths)
        max_bw = max(bandwidths)
        min_idx = bandwidths.index(min_bw)
        max_idx = bandwidths.index(max_bw)
        ratio = (max_bw / min_bw) if min_bw > 0 else 1.0
        return cls(
            gpus=gpus,
            is_symmetric=(min_bw == max_bw),
            min_bw_gpu=gpus[min_idx].gpu_idx,
            max_bw_gpu=gpus[max_idx].gpu_idx,
            bandwidth_ratio=round(ratio, 6),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_pcie_info(gpu_idx: int, sysfs_root: Path) -> GPUPCIeInfo | None:
    bus_id = _nvidia_smi_pci(gpu_idx)
    if bus_id is None:
        return None

    for variant in _pci_path_variants(bus_id):
        dev = sysfs_root / variant
        speed_file = dev / "current_link_speed"
        if not speed_file.exists():
            continue

        try:
            speed_gts = _parse_speed_gts(speed_file.read_text().strip())
            width_file = dev / "current_link_width"
            width = int(width_file.read_text().strip()) if width_file.exists() else 0
            bw = _compute_bw(speed_gts, width)
            nvlink = (dev / "nvlink").exists()
            return GPUPCIeInfo(
                gpu_idx=gpu_idx,
                bus_id=bus_id,
                link_speed_gts=speed_gts,
                link_width=width,
                theoretical_bw_gbs=bw,
                nvlink=nvlink,
            )
        except (ValueError, OSError) as exc:
            logger.debug("PCIe read failed GPU %d at %s: %s", gpu_idx, dev, exc)
            continue

    logger.debug("No PCIe sysfs entry for GPU %d (bus_id=%s)", gpu_idx, bus_id)
    return None


def _parse_speed_gts(speed_str: str) -> float:
    """Parse '8 GT/s PCIe' → 8.0; returns 0.0 on parse failure."""
    try:
        return float(speed_str.split()[0])
    except (ValueError, IndexError):
        return 0.0


def _compute_bw(speed_gts: float, width: int) -> float:
    """Unidirectional theoretical bandwidth in GB/s."""
    gbs_per_lane = _SPEED_GBS_PER_LANE.get(speed_gts)
    if gbs_per_lane is None:
        # Unknown generation — assume 128b/130b framing as a conservative estimate
        gbs_per_lane = speed_gts * (128 / 130) / 8
    return round(gbs_per_lane * width, 3)
