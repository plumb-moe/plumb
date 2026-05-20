"""nvidia-smi dmon poller with rolling buffer of raw per-GPU samples."""
from __future__ import annotations

import collections
import logging
import math
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Columns we ask dmon to export (-s u = utilisation + power)
_DMON_FIELDS = ("-s", "u")
_DMON_INTERVAL = "-d", "1"

_MISSING_SENTINEL = object()  # marks nvidia-smi absent — do not restart


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------

@dataclass
class PerGpuStats:
    gpu_index: int
    sm_mean: float
    sm_p50: float
    sm_p90: float
    sm_p95: float
    sm_peak: float
    mem_mean: float
    pwr_mean: float


@dataclass
class GpuSessionStats:
    per_gpu: list[PerGpuStats] = field(default_factory=list)
    cluster_mean_sm_utilisation: float = 0.0
    imbalance_confirmed: bool = False   # std-dev of per-GPU mean SM util > 15 pp


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile on a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def compute_gpu_session_stats(
    buffer: dict[int, list[dict[str, float]]],
) -> GpuSessionStats:
    """Compute session statistics from a GpuStatsPoller.snapshot() buffer.

    Args:
        buffer: Mapping of GPU index → list of sample dicts (sm/mem/pwr).

    Returns:
        GpuSessionStats with per-GPU percentiles and cluster-level summary.
    """
    if not buffer:
        return GpuSessionStats()

    per_gpu: list[PerGpuStats] = []
    gpu_means: list[float] = []

    for gpu_idx in sorted(buffer):
        samples = buffer[gpu_idx]
        sm_vals  = [s["sm"]  for s in samples if "sm"  in s]
        mem_vals = [s["mem"] for s in samples if "mem" in s]
        pwr_vals = [s["pwr"] for s in samples if "pwr" in s]

        sm_mean = sum(sm_vals) / len(sm_vals) if sm_vals else 0.0
        per_gpu.append(PerGpuStats(
            gpu_index=gpu_idx,
            sm_mean=round(sm_mean, 2),
            sm_p50=round(_percentile(sm_vals, 50), 2),
            sm_p90=round(_percentile(sm_vals, 90), 2),
            sm_p95=round(_percentile(sm_vals, 95), 2),
            sm_peak=round(max(sm_vals, default=0.0), 2),
            mem_mean=round(sum(mem_vals) / len(mem_vals) if mem_vals else 0.0, 2),
            pwr_mean=round(sum(pwr_vals) / len(pwr_vals) if pwr_vals else 0.0, 2),
        ))
        gpu_means.append(sm_mean)

    cluster_mean = sum(gpu_means) / len(gpu_means) if gpu_means else 0.0
    variance = sum((x - cluster_mean) ** 2 for x in gpu_means) / len(gpu_means) if gpu_means else 0.0
    stddev = math.sqrt(variance)

    return GpuSessionStats(
        per_gpu=per_gpu,
        cluster_mean_sm_utilisation=round(cluster_mean, 2),
        imbalance_confirmed=stddev > 15.0,
    )


class GpuStatsPoller:
    """Poll `nvidia-smi dmon` and keep a rolling buffer of per-GPU samples.

    Each sample is a dict mapping GPU index (int) to a dict of metric→value
    (floats), e.g. ``{0: {"sm": 42.0, "mem": 18.0, "pwr": 110.0}, ...}``.

    Args:
        window_seconds: How many seconds of samples to retain (default 300).
        poll_interval:  Passed to ``dmon -d``; typically 1 second.
    """

    def __init__(self, window_seconds: int = 300, poll_interval: int = 1) -> None:
        self._window = window_seconds
        self._poll_interval = poll_interval
        self._buf: collections.deque[dict[int, dict[str, float]]] = collections.deque(
            maxlen=window_seconds
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._disabled = False  # set after two consecutive launch failures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-stats-poller")
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to exit and join."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def snapshot(self) -> dict[int, list[dict[str, float]]]:
        """Return all buffered samples grouped by GPU index.

        Returns an empty dict when nvidia-smi is unavailable.
        Each GPU maps to a list of sample-dicts ordered oldest→newest.
        """
        with self._lock:
            samples = list(self._buf)

        by_gpu: dict[int, list[dict[str, float]]] = {}
        for frame in samples:
            for gpu_idx, metrics in frame.items():
                by_gpu.setdefault(gpu_idx, []).append(metrics)
        return by_gpu

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        attempts = 0
        while not self._stop.is_set():
            result = self._launch_and_read()
            if result is _MISSING_SENTINEL:
                return  # nvidia-smi absent — give up permanently
            attempts += 1
            if attempts >= 2:
                logger.warning("gpu-stats: nvidia-smi crashed twice; disabling poller")
                self._disabled = True
                return
            logger.warning("gpu-stats: nvidia-smi exited unexpectedly; restarting (attempt %d)", attempts)

    def _launch_and_read(self) -> Any:
        """Launch dmon and parse lines until _stop is set or process exits.

        Returns _MISSING_SENTINEL when nvidia-smi is not on PATH or raises
        PermissionError; returns None on normal/crash exit.
        """
        cmd = [
            "nvidia-smi", "dmon",
            *_DMON_FIELDS,
            _DMON_INTERVAL[0], str(self._poll_interval),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            logger.warning("gpu-stats: nvidia-smi not found; GPU stats unavailable")
            return _MISSING_SENTINEL
        except PermissionError as exc:
            logger.warning("gpu-stats: permission denied launching nvidia-smi: %s", exc)
            return _MISSING_SENTINEL

        header: list[str] | None = None
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if self._stop.is_set():
                    proc.terminate()
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    # Header lines: "# gpu   sm  mem  enc  dec  jpg  ofa  pwr gTemp mTemp"
                    tokens = line.lstrip("# ").split()
                    if "gpu" in tokens:
                        header = tokens
                    continue
                if header is None:
                    continue
                self._parse_line(line, header)
        finally:
            proc.wait()

        return None  # process exited (possibly crashed)

    def _parse_line(self, line: str, header: list[str]) -> None:
        tokens = line.split()
        if len(tokens) < len(header):
            return
        try:
            gpu_idx = int(tokens[header.index("gpu")])
        except (ValueError, IndexError):
            return

        metrics: dict[str, float] = {}
        for col in ("sm", "mem", "pwr"):
            if col in header:
                try:
                    val = tokens[header.index(col)]
                    metrics[col] = float(val)
                except (ValueError, IndexError):
                    pass

        if not metrics:
            return

        frame = {gpu_idx: metrics}
        with self._lock:
            self._buf.append(frame)
