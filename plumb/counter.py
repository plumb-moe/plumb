from __future__ import annotations

import threading
from collections import defaultdict, deque


class ActivationCounter:
    """Rolling-window token count per (layer_id, expert_id). Thread-safe."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self._counts: dict[tuple[int, int], deque[int]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._pass_count = 0
        self._lock = threading.Lock()

    def record(self, layer_id: int, expert_id: int, token_count: int = 1) -> None:
        with self._lock:
            self._counts[(layer_id, expert_id)].append(token_count)

    def increment_pass(self) -> None:
        with self._lock:
            self._pass_count += 1

    @property
    def pass_count(self) -> int:
        with self._lock:
            return self._pass_count

    def snapshot(self) -> dict[tuple[int, int], int]:
        """Summed token counts per (layer_id, expert_id) over the current window."""
        with self._lock:
            return {k: int(sum(v)) for k, v in self._counts.items() if v}

    def layer_ids(self) -> list[int]:
        with self._lock:
            return sorted({k[0] for k in self._counts})

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._pass_count = 0
