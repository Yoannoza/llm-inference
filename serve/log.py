"""Logger commun — étapes numérotées, sous-lignes indentées, durées."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager


class StepLogger:
    def __init__(self, total: int, stream=sys.stdout) -> None:
        self.total = total
        self.current = 0
        self.stream = stream
        self._step_start: float | None = None

    def step(self, title: str) -> None:
        if self._step_start is not None:
            dt = time.perf_counter() - self._step_start
            print(f"      ✓ ({dt:.1f}s)", file=self.stream, flush=True)
        self.current += 1
        self._step_start = time.perf_counter()
        print(f"\n[{self.current}/{self.total}] {title}",
              file=self.stream, flush=True)

    def info(self, msg: str) -> None:
        print(f"      → {msg}", file=self.stream, flush=True)

    def warn(self, msg: str) -> None:
        print(f"      ⚠ {msg}", file=self.stream, flush=True)

    def error(self, msg: str) -> None:
        print(f"      ✗ {msg}", file=self.stream, flush=True)

    def done(self, msg: str = "Prêt") -> None:
        if self._step_start is not None:
            dt = time.perf_counter() - self._step_start
            print(f"      ✓ ({dt:.1f}s)", file=self.stream, flush=True)
            self._step_start = None
        print(f"\n✓ {msg}\n", file=self.stream, flush=True)


@contextmanager
def timed(log: StepLogger, label: str):
    t0 = time.perf_counter()
    log.info(f"{label}...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info(f"{label} terminé ({dt:.1f}s)")
