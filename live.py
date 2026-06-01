"""Mini HTTP+SSE server pour suivre un `report` en live depuis un navigateur.

Stdlib uniquement. Démarre un thread HTTP qui sert :
  GET /          → dashboard.html (avec auto-connexion SSE en mode live)
  GET /events    → flux text/event-stream, replay de l'historique aux nouveaux clients
  GET /report.json → snapshot JSON courant (rapport partiel ou final)

Le pipeline `report` tourne en parallèle dans un autre thread et `publish()`
chaque événement sur le bus → fanout vers tous les clients connectés.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


class EventBus:
    """File de diffusion thread-safe avec replay des events passés."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.done = threading.Event()
        self.final_report: dict | None = None

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._history.append(event)
            subs = list(self._subs)
            if event.get("type") == "done":
                self.final_report = event.get("report")
                self.done.set()
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1024)
        with self._lock:
            for e in self._history:
                q.put_nowait(e)
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


def adapt_event(stage: str, payload: Any) -> dict[str, Any]:
    """Convertit les events internes (tuples/dataclasses) en JSON-safe."""
    if stage == "stage":
        return {"type": "stage", "name": payload}
    if stage == "latency_run":
        i, n, r = payload
        return {
            "type": "latency_run", "i": i, "n": n,
            "result": asdict(r) if is_dataclass(r) else r,
        }
    if stage == "throughput_point":
        return {
            "type": "throughput_point",
            "point": asdict(payload) if is_dataclass(payload) else payload,
        }
    if stage == "done":
        return {
            "type": "done",
            "report": asdict(payload) if is_dataclass(payload) else payload,
        }
    return {"type": stage, "payload": str(payload)}


def _make_handler(bus: EventBus):
    class Handler(BaseHTTPRequestHandler):
        # Silence le log par défaut (sinon chaque ping pollue stdout)
        def log_message(self, *_a) -> None:
            return

        def _send(self, status: int, ctype: str, body: bytes,
                  extra_headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                try:
                    data = DASHBOARD_PATH.read_bytes()
                except FileNotFoundError:
                    self._send(404, "text/plain", b"dashboard.html introuvable")
                    return
                self._send(200, "text/html; charset=utf-8", data)
                return

            if path == "/report.json":
                snapshot = bus.final_report or {"status": "in_progress"}
                self._send(200, "application/json",
                           json.dumps(snapshot, default=str).encode())
                return

            if path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                q = bus.subscribe()
                try:
                    while True:
                        try:
                            ev = q.get(timeout=15)
                        except queue.Empty:
                            # heartbeat pour garder la connexion ouverte
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        line = f"data: {json.dumps(ev, default=str)}\n\n"
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    bus.unsubscribe(q)
                return

            self._send(404, "text/plain", b"not found")

    return Handler


def start_server(bus: EventBus, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), _make_handler(bus))
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
