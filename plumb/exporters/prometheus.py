from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..counter import ActivationCounter

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Exports per-expert activation counts and imbalance ratios as Prometheus metrics.

    Metrics:
      vllm:moe_expert_activation_count{layer, expert}  — cumulative token count (Counter)
      vllm:moe_imbalance_ratio{layer}                  — current imbalance ratio (Gauge)
    """

    def __init__(self, counter: ActivationCounter, port: int = 9000,
                 interval: float = 5.0) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge

        self._counter = counter
        self._port = port
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_snapshot: dict[tuple[int, int], int] = {}

        self._registry = CollectorRegistry()
        self._act_counter = Counter(
            "vllm:moe_expert_activation_count",
            "Cumulative token count routed to each MoE expert",
            ["layer", "expert"],
            registry=self._registry,
        )
        self._imbalance_gauge = Gauge(
            "vllm:moe_imbalance_ratio",
            "Expert load imbalance ratio per layer (max_load / mean_load)",
            ["layer"],
            registry=self._registry,
        )

    def start(self) -> None:
        """Start the HTTP server and background update thread."""
        from prometheus_client import start_http_server

        start_http_server(self._port, registry=self._registry)
        logger.info("plumb: Prometheus metrics at http://0.0.0.0:%d/metrics", self._port)

        self._thread = threading.Thread(
            target=self._loop, name="plumb-prometheus", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background update thread to stop."""
        self._stop.set()

    def update(self) -> None:
        """Read current ActivationCounter state and push to Prometheus metrics."""
        from ..analysis.imbalance import compute_imbalance

        current = self._counter.snapshot()

        for (layer_id, expert_id), count in current.items():
            prev = self._prev_snapshot.get((layer_id, expert_id), 0)
            delta = count - prev
            if delta > 0:
                self._act_counter.labels(
                    layer=str(layer_id), expert=str(expert_id)
                ).inc(delta)

        self._prev_snapshot = current

        for imb in compute_imbalance(self._counter):
            self._imbalance_gauge.labels(layer=str(imb.layer_id)).set(imb.imbalance_ratio)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                self.update()
            except Exception:
                logger.debug("prometheus update error", exc_info=True)
