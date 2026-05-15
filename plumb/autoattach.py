from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SCAN_INTERVAL = int(os.environ.get("SAI_PROFILER_SCAN_INTERVAL", "8"))  # seconds between gc sweeps
_MIN_LAYERS = 2         # ignore tiny test models
_attached_ids: set[int] = set()
_lock = threading.Lock()
_scan_active = True     # set to False once a model is found; ends GC polling


def start_background_profiler() -> None:
    """Spawn the background profiling thread. Called from sitecustomize on SAI_PROFILER_AUTO=1."""
    t = threading.Thread(target=_profiler_main, name="plumb-auto", daemon=True)
    t.start()
    logger.debug("plumb background thread started (pid=%d)", os.getpid())


def _profiler_main() -> None:
    global _scan_active
    while _scan_active:
        time.sleep(_SCAN_INTERVAL)
        try:
            _scan_and_attach()
        except Exception:
            logger.debug("autoattach scan error", exc_info=True)


def _scan_and_attach() -> None:
    try:
        import torch.nn as nn
    except ImportError:
        return

    from .hook import _BLOCK_EXTRACTORS, _VLLM_GATE_BLOCKS

    _ALL_MOE_TYPES = frozenset(_BLOCK_EXTRACTORS) | frozenset(_VLLM_GATE_BLOCKS)

    for obj in gc.get_objects():
        if not isinstance(obj, nn.Module):
            continue
        oid = id(obj)
        with _lock:
            if oid in _attached_ids:
                continue

        # Count MoE layers across both transformers block-level and vLLM gate-block patterns
        n_moe = sum(1 for _, m in obj.named_modules() if type(m).__name__ in _ALL_MOE_TYPES)
        if n_moe < _MIN_LAYERS:
            continue

        with _lock:
            if oid in _attached_ids:
                continue
            _attached_ids.add(oid)

        if _attach_session(obj, n_moe):
            # Stop polling — model found and hooked, GC scan no longer needed
            global _scan_active
            _scan_active = False
            return


def _attach_session(model, n_moe: int) -> bool:
    from .counter import ActivationCounter
    from .hook import ProfilingHooks
    from .registry import SessionInfo, REGISTRY_DIR, register
    from .topology import Topology

    model_name = type(model).__name__
    logger.info("plumb: attaching to %s (%d MoE layers)", model_name, n_moe)

    counter = ActivationCounter(window_size=5000)
    hooks = ProfilingHooks(counter)
    n = hooks.attach(model, top_k=2)
    if n == 0:
        logger.debug("plumb: no hooks attached to %s, skipping", model_name)
        return False

    topology = Topology.discover()

    pid = os.getpid()
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = str(REGISTRY_DIR / f"{pid}_snapshot.json")

    info = SessionInfo(
        pid=pid,
        model_name=model_name,
        n_layers=n,
        socket_path="",
        started_at=time.time(),
        snapshot_path=snapshot_path,
    )
    register(info)

    # Background thread: write snapshot every 2 seconds
    def _snapshot_writer() -> None:
        from .analysis.imbalance import compute_imbalance

        while True:
            time.sleep(2)
            try:
                snap = counter.snapshot()
                imbalance = compute_imbalance(counter)
                payload = {
                    "pid": pid,
                    "model_name": model_name,
                    "n_layers": n,
                    "pass_count": counter.pass_count,
                    "updated_at": time.time(),
                    "started_at": info.started_at,
                    "imbalance": [
                        {
                            "layer_id": r.layer_id,
                            "ratio": round(r.imbalance_ratio, 4),
                            "max_expert": r.max_expert_id,
                        }
                        for r in sorted(imbalance, key=lambda x: x.layer_id)
                    ],
                    # Raw per-expert token counts — used by `plumb report`
                    "expert_loads": {
                        f"{lid}:{eid}": count for (lid, eid), count in snap.items()
                    },
                    "gpu_to_numa": topology.gpu_to_numa,
                    # Raw counts enable full report generation from outside the process
                    "expert_counts": {
                        f"{lid}:{eid}": count
                        for (lid, eid), count in snap.items()
                    },
                }
                Path(snapshot_path).write_text(json.dumps(payload))
            except Exception:
                logger.debug("snapshot write error", exc_info=True)

    t = threading.Thread(target=_snapshot_writer, name="plumb-snapshot", daemon=True)
    t.start()
    logger.info("plumb: session registered (snapshot → %s)", snapshot_path)

    prom_port_str = os.environ.get("SAI_PROFILER_PROMETHEUS_PORT")
    if prom_port_str:
        try:
            from .exporters.prometheus import PrometheusExporter
            exporter = PrometheusExporter(counter, port=int(prom_port_str))
            exporter.start()
        except Exception as exc:
            logger.warning("plumb: Prometheus exporter failed to start: %s", exc)

    return True
