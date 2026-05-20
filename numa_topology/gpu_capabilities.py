"""GPU capability discovery via nvidia-smi.

Provides GPUCapability and HeterogeneousTopology dataclasses populated by
parsing `nvidia-smi --query-gpu=... --format=csv,noheader` output.

When nvidia-smi is unavailable a flat symmetric single-GPU topology is
returned with a warning so callers never need to special-case the import.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_QUERY_FIELDS = (
    "index",
    "name",
    "memory.total",
    "memory.free",
    "compute_cap",
    "clocks.max.sm",
    "clocks.max.memory",
)

_NVIDIA_SMI_CMD = [
    "nvidia-smi",
    f"--query-gpu={','.join(_QUERY_FIELDS)}",
    "--format=csv,noheader",
]


@dataclass
class GPUCapability:
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int
    compute_cap: str           # e.g. "8.9"
    max_sm_clock_mhz: int
    max_mem_clock_mhz: int
    relative_compute_score: float = 1.0  # 1.0 = fastest GPU in the system


@dataclass
class HeterogeneousTopology:
    gpus: list[GPUCapability] = field(default_factory=list)
    is_homogeneous: bool = True
    mixed_vendor: bool = False
    compute_score_range: tuple[float, float] = (1.0, 1.0)


def discover_gpu_capabilities() -> HeterogeneousTopology:
    """Query nvidia-smi and return a HeterogeneousTopology.

    Falls back to a single-GPU flat topology (all scores = 1.0) with a
    warning when nvidia-smi is missing, returns PermissionError, or produces
    unparseable output.
    """
    try:
        result = subprocess.run(
            _NVIDIA_SMI_CMD,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logger.warning("gpu-capabilities: nvidia-smi not found; using flat fallback topology")
        return _flat_fallback()
    except PermissionError as exc:
        logger.warning("gpu-capabilities: permission denied running nvidia-smi: %s; using flat fallback", exc)
        return _flat_fallback()
    except subprocess.TimeoutExpired:
        logger.warning("gpu-capabilities: nvidia-smi timed out; using flat fallback topology")
        return _flat_fallback()

    if result.returncode != 0:
        logger.warning(
            "gpu-capabilities: nvidia-smi exited %d; using flat fallback topology",
            result.returncode,
        )
        return _flat_fallback()

    gpus = _parse_csv(result.stdout)
    if not gpus:
        logger.warning("gpu-capabilities: no GPUs reported by nvidia-smi; using flat fallback topology")
        return _flat_fallback()

    gpus = _normalise_scores(gpus)
    return _build_topology(gpus)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_csv(output: str) -> list[GPUCapability]:
    gpus: list[GPUCapability] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_QUERY_FIELDS):
            logger.debug("gpu-capabilities: skipping malformed line: %r", line)
            continue
        try:
            idx           = int(parts[0])
            name          = parts[1]
            mem_total     = _parse_mib(parts[2])
            mem_free      = _parse_mib(parts[3])
            compute_cap   = parts[4]
            sm_clock      = _parse_int_mhz(parts[5])
            mem_clock     = _parse_int_mhz(parts[6])
        except (ValueError, IndexError) as exc:
            logger.debug("gpu-capabilities: parse error on line %r: %s", line, exc)
            continue
        gpus.append(GPUCapability(
            index=idx,
            name=name,
            memory_total_mib=mem_total,
            memory_free_mib=mem_free,
            compute_cap=compute_cap,
            max_sm_clock_mhz=sm_clock,
            max_mem_clock_mhz=mem_clock,
        ))
    return gpus


def _parse_mib(s: str) -> int:
    """Parse '80513 MiB' or '80513' → int MiB."""
    return int(s.split()[0])


def _parse_int_mhz(s: str) -> int:
    """Parse '1980 MHz' or '1980' → int MHz."""
    return int(s.split()[0])


def _raw_score(gpu: GPUCapability) -> float:
    """SM clock × compute_cap float (unnormalised)."""
    try:
        cap = float(gpu.compute_cap)
    except ValueError:
        cap = 1.0
    return gpu.max_sm_clock_mhz * cap


def _normalise_scores(gpus: list[GPUCapability]) -> list[GPUCapability]:
    """Set relative_compute_score so the fastest GPU = 1.0."""
    scores = [_raw_score(g) for g in gpus]
    max_score = max(scores) if scores else 1.0
    if max_score <= 0:
        max_score = 1.0
    for gpu, raw in zip(gpus, scores):
        gpu.relative_compute_score = round(raw / max_score, 4)
    return gpus


def _build_topology(gpus: list[GPUCapability]) -> HeterogeneousTopology:
    names = [g.name for g in gpus]
    is_homogeneous = len(set(names)) == 1

    # Vendor detection: look for AMD/ROCm indicators in GPU name
    has_nvidia = any("nvidia" in n.lower() or "geforce" in n.lower() or "quadro" in n.lower()
                     or "tesla" in n.lower() or "rtx" in n.lower() or "gtx" in n.lower()
                     or "a100" in n.lower() or "h100" in n.lower() or "v100" in n.lower()
                     for n in names)
    has_amd = any("amd" in n.lower() or "radeon" in n.lower() or "instinct" in n.lower()
                  for n in names)
    mixed_vendor = has_nvidia and has_amd

    scores = [g.relative_compute_score for g in gpus]
    score_range = (round(min(scores), 4), round(max(scores), 4)) if scores else (1.0, 1.0)

    return HeterogeneousTopology(
        gpus=gpus,
        is_homogeneous=is_homogeneous,
        mixed_vendor=mixed_vendor,
        compute_score_range=score_range,
    )


def _flat_fallback() -> HeterogeneousTopology:
    """Single synthetic GPU with all scores = 1.0."""
    gpu = GPUCapability(
        index=0,
        name="unknown",
        memory_total_mib=0,
        memory_free_mib=0,
        compute_cap="0.0",
        max_sm_clock_mhz=0,
        max_mem_clock_mhz=0,
        relative_compute_score=1.0,
    )
    return HeterogeneousTopology(
        gpus=[gpu],
        is_homogeneous=True,
        mixed_vendor=False,
        compute_score_range=(1.0, 1.0),
    )
