"""Persistance d'état : un seul serveur géré à la fois → serve/state.json."""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, asdict
from pathlib import Path


STATE_PATH = Path(__file__).resolve().parent / "state.json"
LOG_DIR = Path(__file__).resolve().parent / "logs"


@dataclass
class ServerState:
    pid: int
    model: str
    model_path: str
    port: int
    host: str
    started_at: str
    log_file: str

    def as_dict(self) -> dict:
        return asdict(self)


def save(state: ServerState) -> None:
    STATE_PATH.write_text(json.dumps(state.as_dict(), indent=2))


def load() -> ServerState | None:
    if not STATE_PATH.exists():
        return None
    try:
        d = json.loads(STATE_PATH.read_text())
        return ServerState(**d)
    except Exception:
        return None


def clear() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def kill(pid: int, timeout: float = 5.0) -> bool:
    """SIGTERM puis SIGKILL si besoin. Retourne True si le process est mort."""
    import time
    if not pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    time.sleep(0.2)
    return not pid_alive(pid)
