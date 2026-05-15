from __future__ import annotations

import logging
import time
from typing import Any

import torch.nn as nn

from .counter import ActivationCounter
from .hook import ProfilingHooks
from .report.generator import generate_report
from .report.schema import ProfileReport
from .topology import Topology

logger = logging.getLogger(__name__)


class Session:
    """High-level handle: attach hooks, run inference, produce report."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        top_k: int = 2,
        window_size: int = 1000,
    ) -> None:
        self.model_name = model_name
        self._counter = ActivationCounter(window_size=window_size)
        self._hooks = ProfilingHooks(self._counter)
        self._topology = Topology.discover()
        self._model = model
        self._top_k = top_k
        self._start: float = 0.0
        self._n_layers: int = 0

    def start(self) -> int:
        self._n_layers = self._hooks.attach(self._model, top_k=self._top_k)
        self._start = time.time()
        return self._n_layers

    def stop(self) -> None:
        self._hooks.detach()

    def report(self, num_gpus: int | None = None) -> ProfileReport:
        return generate_report(
            counter=self._counter,
            topology=self._topology,
            model_name=self.model_name,
            duration_seconds=time.time() - self._start,
            num_gpus=num_gpus,
        )

    def __enter__(self) -> Session:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
