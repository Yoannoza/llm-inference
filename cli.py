"""
CLI infer-metrics — 4 familles de métriques + rapport unifié.

Sous-commandes :
  latency     v0.1 — TTFT + TPOT
  throughput  v0.2 — courbe latence/débit système
  cost        v0.3 — $/1M tokens output observé
  resources   v0.4 — VRAM peak (GPU NVIDIA local)
  report      v1.0 — tout d'un coup, sortie texte + JSON

Toutes les sous-commandes parlent une API OpenAI-compatible
(--base-url + --model). `cost` et `resources` peuvent aussi être
combinés à `latency` via `report`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .cost import compute_cost, cost_from_results
from .latency import measure_latency_once, summarize
from .report import report_to_json, run_full_report
from .resources import VRAMMonitor
from .throughput import sweep_throughput


DEFAULT_PROMPT = (
    "Explique en 3 phrases ce qu'est l'inférence d'un LLM, "
    "et pourquoi le TTFT et le TPOT sont des métriques différentes."
)


# ───────────────────────── arguments communs ──────────────────────────

def _add_endpoint_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=128)


# ───────────────────────── cmd: latency ───────────────────────────────

def cmd_latency(args) -> int:
    print(f"\n→ Modèle : {args.model}")
    print(f"→ Endpoint : {args.base_url}")
    print(f"→ {args.warmup} warmup + {args.runs} runs mesurés\n")

    for i in range(args.warmup):
        print(f"  [warmup {i+1}/{args.warmup}] ...", end=" ", flush=True)
        r = measure_latency_once(
            base_url=args.base_url, model=args.model, prompt=args.prompt,
            api_key=args.api_key, max_tokens=args.max_tokens,
        )
        print(str(r))

    results = []
    for i in range(args.runs):
        print(f"  [run     {i+1}/{args.runs}] ...", end=" ", flush=True)
        r = measure_latency_once(
            base_url=args.base_url, model=args.model, prompt=args.prompt,
            api_key=args.api_key, max_tokens=args.max_tokens,
        )
        results.append(r)
        print(str(r))

    s = summarize(results)
    print("\n" + "═" * 60)
    print(f"  LATENCE — résumé sur {s['n_runs']} runs (médianes)")
    print("═" * 60)
    print(f"  TTFT médian       : {s['ttft_median_ms']:7.1f} ms"
          f"   (min {s['ttft_min_ms']:.0f} / max {s['ttft_max_ms']:.0f})")
    print(f"  TPOT médian       : {s['tpot_median_ms']:7.1f} ms/token")
    print(f"  → débit observé   : {s['tokens_per_sec_median']:7.1f} tokens/s")
    print("═" * 60 + "\n")
    return 0


# ───────────────────────── cmd: throughput ────────────────────────────

def cmd_throughput(args) -> int:
    concurrencies = [int(c) for c in args.concurrencies.split(",")]
    print(f"\n→ Modèle : {args.model}")
    print(f"→ Endpoint : {args.base_url}")
    print(f"→ Sweep concurrence : {concurrencies} "
          f"({args.requests} requêtes par niveau)\n")

    def on_point(p):
        print(f"  {p}")

    points = sweep_throughput(
        base_url=args.base_url, model=args.model, prompt=args.prompt,
        concurrencies=concurrencies, requests_per_level=args.requests,
        api_key=args.api_key, max_tokens=args.max_tokens,
        on_point=on_point,
    )

    best = max(points, key=lambda p: p.system_tokens_per_sec)
    print("\n" + "═" * 60)
    print("  DÉBIT — courbe latence/débit (système)")
    print("═" * 60)
    for p in points:
        marker = "  ★" if p is best else "   "
        print(f"{marker} conc={p.concurrency:>3} → "
              f"{p.system_tokens_per_sec:7.1f} tok/s "
              f"(TTFT méd. {p.ttft_median_ms:5.0f} ms, "
              f"TPOT méd. {p.tpot_median_ms:5.1f} ms/tok)")
    print(f"\n  → Pic : {best.system_tokens_per_sec:.1f} tokens/s "
          f"à concurrence {best.concurrency}")
    print("═" * 60 + "\n")
    return 0


# ───────────────────────── cmd: cost ──────────────────────────────────

def cmd_cost(args) -> int:
    if args.from_run:
        # Mesure live + calcule le coût observé.
        print(f"\n→ Run unique pour observation du coût...")
        r = measure_latency_once(
            base_url=args.base_url, model=args.model, prompt=args.prompt,
            api_key=args.api_key, max_tokens=args.max_tokens,
        )
        cb = cost_from_results(
            [r], model=args.model,
            price_in_per_mtok=args.price_in,
            price_out_per_mtok=args.price_out,
            fallback_prompt_tokens=max(1, len(args.prompt) // 4),
        )
    else:
        # Calcul analytique à partir des comptes passés en CLI.
        cb = compute_cost(
            model=args.model,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            price_in_per_mtok=args.price_in,
            price_out_per_mtok=args.price_out,
        )

    print("\n" + "═" * 60)
    print("  COÛT")
    print("═" * 60)
    print(f"  Modèle             : {cb.model}")
    print(f"  Tarif input        : ${cb.price_in_per_mtok:.4f} / 1M tokens")
    print(f"  Tarif output       : ${cb.price_out_per_mtok:.4f} / 1M tokens")
    print(f"  Tokens in / out    : {cb.input_tokens} / {cb.output_tokens}")
    print(f"  Coût total         : ${cb.cost_total_usd:.6f}")
    print(f"  → Effectif         : ${cb.effective_dollars_per_mtok_out:.2f} "
          f"/ 1M tokens output")
    print("═" * 60 + "\n")
    return 0


# ───────────────────────── cmd: resources ─────────────────────────────

def cmd_resources(args) -> int:
    print(f"\n→ Monitoring VRAM GPU #{args.gpu_index} pendant {args.runs} runs...\n")
    with VRAMMonitor(gpu_index=args.gpu_index, interval_s=args.interval) as mon:
        for i in range(args.runs):
            print(f"  [run {i+1}/{args.runs}] ...", end=" ", flush=True)
            r = measure_latency_once(
                base_url=args.base_url, model=args.model, prompt=args.prompt,
                api_key=args.api_key, max_tokens=args.max_tokens,
            )
            print(str(r))

    u = mon.result()
    print("\n" + "═" * 60)
    print("  RESSOURCES — VRAM")
    print("═" * 60)
    if not u.available:
        print(f"  Indisponible : {u.reason}")
    else:
        print(f"  GPU #{u.gpu_index} — {u.n_samples} échantillons "
              f"sur {u.duration_s:.1f}s")
        print(f"  VRAM peak          : {u.vram_used_peak_mib} MiB "
              f"/ {u.vram_total_mib} MiB "
              f"({100*u.vram_used_peak_mib/u.vram_total_mib:.1f} %)")
        print(f"  VRAM moyenne       : {u.vram_used_mean_mib:.0f} MiB")
        print(f"  Util. GPU pic      : {u.util_gpu_peak_pct} %")
    print("═" * 60 + "\n")
    return 0


# ───────────────────────── cmd: report ────────────────────────────────

def cmd_report(args) -> int:
    concurrencies = (
        [int(c) for c in args.concurrencies.split(",")]
        if args.concurrencies else None
    )

    def on_event(stage, payload):
        if stage == "stage":
            print(f"\n── {payload.upper()} ──")
        elif stage == "latency_run":
            i, n, r = payload
            print(f"  [run {i}/{n}] {r}")
        elif stage == "throughput_point":
            print(f"  {payload}")

    report = run_full_report(
        base_url=args.base_url, model=args.model, prompt=args.prompt,
        api_key=args.api_key, max_tokens=args.max_tokens,
        latency_runs=args.runs, warmup=args.warmup,
        concurrencies=concurrencies,
        requests_per_level=args.requests,
        price_in_per_mtok=args.price_in,
        price_out_per_mtok=args.price_out,
        measure_resources=not args.no_resources,
        gpu_index=args.gpu_index,
        on_event=on_event,
    )

    # Résumé texte final.
    print("\n" + "═" * 60)
    print(f"  RAPPORT UNIFIÉ — {report.model}")
    print("═" * 60)
    s = report.latency_summary or {}
    if s:
        print(f"  Latence   : TTFT {s.get('ttft_median_ms', float('nan')):.0f} ms"
              f"  |  TPOT {s.get('tpot_median_ms', float('nan')):.1f} ms/tok")
    if report.throughput_best_tokens_per_sec is not None:
        print(f"  Débit pic : {report.throughput_best_tokens_per_sec:.1f} tok/s"
              f" @ conc {report.throughput_best_concurrency}")
    if report.cost and "error" not in report.cost:
        c = report.cost
        print(f"  Coût      : ${c['effective_dollars_per_mtok_out']:.2f}"
              " / 1M tokens output")
    elif report.cost:
        print(f"  Coût      : (skipped — {report.cost['error']})")
    if report.resources:
        u = report.resources
        if u.get("available"):
            print(f"  VRAM peak : {u['vram_used_peak_mib']} MiB "
                  f"/ {u['vram_total_mib']} MiB")
        else:
            print(f"  VRAM      : (skipped — {u.get('reason')})")
    print("═" * 60 + "\n")

    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(report_to_json(report))
        print(f"  Rapport JSON écrit dans {args.json_out}\n")
    return 0


# ───────────────────────── parser principal ───────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer-metrics",
        description="Mesure les métriques d'inférence LLM sur API OpenAI-compat.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # latency
    p = sub.add_parser("latency", help="v0.1 — TTFT + TPOT")
    _add_endpoint_args(p)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.set_defaults(func=cmd_latency)

    # throughput
    p = sub.add_parser("throughput", help="v0.2 — courbe latence/débit")
    _add_endpoint_args(p)
    p.add_argument("--concurrencies", default="1,2,4,8",
                   help="Niveaux de concurrence séparés par virgules.")
    p.add_argument("--requests", type=int, default=8,
                   help="Nb de requêtes lancées à chaque niveau.")
    p.set_defaults(func=cmd_throughput)

    # cost
    p = sub.add_parser("cost", help="v0.3 — $/1M tokens output")
    p.add_argument("--model", required=True,
                   help="Nom du modèle (sert au lookup de tarif).")
    p.add_argument("--from-run", action="store_true",
                   help="Lance un vrai run pour mesurer in/out tokens.")
    # Args pour --from-run (endpoint réel)
    p.add_argument("--base-url")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=128)
    # Args pour calcul analytique
    p.add_argument("--input-tokens", type=int, default=0)
    p.add_argument("--output-tokens", type=int, default=0)
    # Override tarifs
    p.add_argument("--price-in", type=float, default=None,
                   help="$ / 1M tokens input (override catalogue).")
    p.add_argument("--price-out", type=float, default=None,
                   help="$ / 1M tokens output (override catalogue).")
    p.set_defaults(func=cmd_cost)

    # resources
    p = sub.add_parser("resources", help="v0.4 — VRAM peak (NVIDIA local)")
    _add_endpoint_args(p)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.1,
                   help="Intervalle de polling nvidia-smi (s).")
    p.set_defaults(func=cmd_resources)

    # report
    p = sub.add_parser("report", help="v1.0 — toutes les familles + JSON")
    _add_endpoint_args(p)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--concurrencies", default="1,4",
                   help="Sweep débit (vide pour skip).")
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--price-in", type=float, default=None)
    p.add_argument("--price-out", type=float, default=None)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--no-resources", action="store_true")
    p.add_argument("--json-out", default=None,
                   help="Chemin de sortie JSON (optionnel).")
    p.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Validation custom pour `cost` (combos d'args incompatibles).
    if args.cmd == "cost":
        if args.from_run and not args.base_url:
            print("erreur : --from-run nécessite --base-url", file=sys.stderr)
            return 2
        if not args.from_run and (args.input_tokens == 0 and args.output_tokens == 0):
            print("erreur : passe soit --from-run, soit --input-tokens/--output-tokens",
                  file=sys.stderr)
            return 2

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
