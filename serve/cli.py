"""CLI infer-serve.

Sous-commandes :
  up        Tout-en-un : détecte, installe, télécharge, lance, vérifie.
  down      Stoppe le serveur courant (via state.json).
  status    Affiche l'état du serveur courant.
  detect    Affiche l'env détecté (debug).
  install   Installe llama.cpp seulement.
  pull      Télécharge un modèle GGUF seulement.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from .detect import detect_env
from .install import ensure_installed, llama_server_path
from .log import StepLogger
from .model import fetch_model
from .server import health_check, launch
from .state import clear, kill, load, pid_alive


DEFAULT_MODEL = "unsloth/gemma-4-E4B-it-GGUF"


# ───────────────────────── cmd: up ────────────────────────────────────

def cmd_up(args) -> int:
    log = StepLogger(total=5)

    log.step("Détection de l'environnement")
    env = detect_env()
    log.info(env.summary())

    log.step("Installation de llama.cpp")
    binary = ensure_installed(env, log)
    log.info(f"binaire : {binary}")

    log.step("Téléchargement du modèle")
    pick = fetch_model(env, args.model, args.quant, log)
    log.info(f"local : {pick.local_path}")

    log.step("Démarrage du serveur")
    state = launch(
        binary=binary, model_repo=args.model, model_path=pick.local_path,
        host=args.host, port=args.port, ctx=args.ctx, ngl=args.ngl,
        env=env, log=log, dry_run=args.dry_run,
    )
    if args.dry_run:
        log.done("Dry-run terminé — rien n'a été lancé")
        return 0

    log.step("Vérification")
    try:
        health_check(args.host, args.port, log, model_name=args.model)
    except Exception as e:
        log.error(str(e))
        log.warn("le serveur tourne mais la vérif a échoué — voir logs")
        return 1

    log.done(f"Prêt sur http://{args.host}:{args.port}/v1")
    print(f"  → Test : python -m cli latency \\\n"
          f"      --base-url http://{args.host}:{args.port}/v1 \\\n"
          f"      --model {args.model}\n")
    return 0


# ───────────────────────── cmd: down ──────────────────────────────────

def cmd_down(args) -> int:
    state = load()
    if not state:
        print("Aucun serveur géré (state.json absent).")
        return 0
    if not pid_alive(state.pid):
        print(f"PID {state.pid} déjà mort — nettoyage du state.")
        clear()
        return 0
    print(f"Arrêt PID {state.pid} (modèle {state.model})...")
    if kill(state.pid):
        clear()
        print("✓ Arrêté.")
        return 0
    print("✗ Échec de l'arrêt.", file=sys.stderr)
    return 1


# ───────────────────────── cmd: status ────────────────────────────────

def cmd_status(args) -> int:
    state = load()
    if not state:
        print("Aucun serveur géré.")
        return 0
    alive = pid_alive(state.pid)
    print(f"  PID       : {state.pid} ({'vivant' if alive else 'MORT'})")
    print(f"  Modèle    : {state.model}")
    print(f"  Endpoint  : http://{state.host}:{state.port}/v1")
    print(f"  Démarré   : {state.started_at}")
    print(f"  Logs      : {state.log_file}")
    if alive:
        try:
            url = f"http://{'127.0.0.1' if state.host == '0.0.0.0' else state.host}:{state.port}/v1/models"
            with urllib.request.urlopen(url, timeout=2) as r:
                print(f"  /v1/models: {r.status} ✓")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"  /v1/models: pas de réponse ({e})")
    return 0


# ───────────────────────── cmd: detect / install / pull ───────────────

def cmd_detect(args) -> int:
    env = detect_env()
    print(env.summary())
    print(json.dumps(env.as_dict(), indent=2))
    return 0


def cmd_install(args) -> int:
    log = StepLogger(total=1)
    log.step("Installation de llama.cpp")
    env = detect_env()
    log.info(env.summary())
    binary = ensure_installed(env, log)
    log.done(f"llama-server → {binary}")
    return 0


def cmd_pull(args) -> int:
    log = StepLogger(total=1)
    log.step(f"Téléchargement de {args.model}")
    env = detect_env()
    pick = fetch_model(env, args.model, args.quant, log)
    log.done(f"{pick.filename} → {pick.local_path}")
    return 0


# ───────────────────────── parser ─────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer-serve",
        description="Host un modèle GGUF en endpoint OpenAI-compat via llama.cpp.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # up
    p = sub.add_parser("up", help="Detect → install → pull → serve → check")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Repo HF GGUF (défaut : {DEFAULT_MODEL})")
    p.add_argument("--quant", default=None,
                   help="Quantisation (ex : Q4_K_M). Auto si non fourni.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ngl", default="auto",
                   help="Couches sur GPU (entier ou 'auto').")
    p.add_argument("--dry-run", action="store_true",
                   help="Affiche les actions sans lancer le serveur.")
    p.set_defaults(func=cmd_up)

    # down
    p = sub.add_parser("down", help="Stoppe le serveur courant")
    p.set_defaults(func=cmd_down)

    # status
    p = sub.add_parser("status", help="État du serveur courant")
    p.set_defaults(func=cmd_status)

    # detect
    p = sub.add_parser("detect", help="Affiche l'environnement détecté")
    p.set_defaults(func=cmd_detect)

    # install
    p = sub.add_parser("install", help="Installe llama.cpp seulement")
    p.set_defaults(func=cmd_install)

    # pull
    p = sub.add_parser("pull", help="Télécharge un modèle GGUF")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--quant", default=None)
    p.set_defaults(func=cmd_pull)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
