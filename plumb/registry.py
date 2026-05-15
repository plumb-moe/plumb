from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

REGISTRY_DIR = Path("/tmp/plumb")


@dataclass
class SessionInfo:
    pid: int
    model_name: str
    n_layers: int
    socket_path: str
    started_at: float
    snapshot_path: str

    def is_alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False


def register(info: SessionInfo) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    (REGISTRY_DIR / f"{info.pid}.json").write_text(json.dumps(asdict(info)))


def deregister(pid: int) -> None:
    try:
        (REGISTRY_DIR / f"{pid}.json").unlink()
    except FileNotFoundError:
        pass


def list_sessions() -> list[SessionInfo]:
    if not REGISTRY_DIR.exists():
        return []

    sessions: list[SessionInfo] = []
    for path in REGISTRY_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            info = SessionInfo(**data)
        except Exception:
            continue

        if not info.is_alive():
            continue

        sessions.append(info)

    return sorted(sessions, key=lambda s: s.started_at)
