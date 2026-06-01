"""Cycle de vie du serveur llama-server."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .detect import Env
from .log import StepLogger
from .state import LOG_DIR, ServerState, save


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def build_command(
    binary: str, model_path: str, host: str, port: int,
    ctx: int, ngl: int | str, env: Env,
) -> list[str]:
    if ngl == "auto":
        # NVIDIA → offload total ; Metal → 99 (= toutes les couches sur GPU) ;
        # CPU pur → 0.
        if env.gpu_kind == "nvidia":
            ngl_v = 99
        elif env.gpu_kind == "metal":
            ngl_v = 99
        else:
            ngl_v = 0
    else:
        ngl_v = int(ngl)

    return [
        binary,
        "-m", model_path,
        "--host", host,
        "--port", str(port),
        "-c", str(ctx),
        "-ngl", str(ngl_v),
    ]


def health_check(host: str, port: int, log: StepLogger,
                 model_name: str, timeout: float = 60.0) -> None:
    base = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    url = f"{base}/v1/models"
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    log.info(f"GET /v1/models → 200 ✓")
                    break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(0.5)
    else:
        raise RuntimeError(f"timeout — /v1/models n'a pas répondu ({last_err})")

    # Petit ping de complétion.
    import json
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status == 200:
                log.info("POST /v1/chat/completions → 200 ✓")
    except Exception as e:
        log.warn(f"complétion test a échoué ({e}) — endpoint up mais à vérifier")


def launch(
    binary: str, model_repo: str, model_path: str,
    host: str, port: int, ctx: int, ngl: int | str,
    env: Env, log: StepLogger, dry_run: bool = False,
) -> ServerState:
    cmd = build_command(binary, model_path, host, port, ctx, ngl, env)
    log.info(" ".join(cmd))

    if dry_run:
        log.info("(dry-run — pas de lancement)")
        return ServerState(
            pid=0, model=model_repo, model_path=model_path,
            port=port, host=host, started_at=_now(),
            log_file="(dry-run)",
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"llama-server-{_ts_tag()}.log"
    fh = open(log_file, "wb")

    proc = subprocess.Popen(
        cmd,
        stdout=fh, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # détache du shell
        env={**os.environ},
    )
    log.info(f"PID {proc.pid} | logs → {log_file}")

    state = ServerState(
        pid=proc.pid, model=model_repo, model_path=model_path,
        port=port, host=host, started_at=_now(), log_file=str(log_file),
    )
    save(state)

    # Vérif rapide que le process n'est pas mort tout de suite.
    time.sleep(0.5)
    if proc.poll() is not None:
        raise RuntimeError(
            f"llama-server a quitté immédiatement (code {proc.returncode}). "
            f"Voir {log_file}"
        )
    return state
