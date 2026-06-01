"""
RAPPORT UNIFIÉ (v1.0) — exécute les 4 familles et produit un résumé.

Sortie en deux formats :
  - texte humain (stdout)
  - JSON structuré (optionnel, via --json-out FILE) pour archivage / CI.

Le JSON est volontairement plat et stable : c'est ce qui permettra plus tard
de comparer des runs entre eux (avant/après une optim, deux fournisseurs...).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from cost import CostBreakdown, cost_from_results
from latency import LatencyResult, measure_latency_once, summarize
from resources import ResourceUsage, VRAMMonitor
from throughput import ThroughputPoint, sweep_throughput


@dataclass
class UnifiedReport:
    """Snapshot complet d'un modèle sous un endpoint donné."""
    model: str
    base_url: str
    timestamp_utc: str
    host: dict[str, str]

    # v0.1 — latence
    latency_summary: dict[str, float] | None = None
    latency_runs: list[dict[str, Any]] = field(default_factory=list)

    # v0.2 — débit
    throughput_curve: list[dict[str, Any]] = field(default_factory=list)
    throughput_best_tokens_per_sec: float | None = None
    throughput_best_concurrency: int | None = None

    # v0.3 — coût
    cost: dict[str, Any] | None = None

    # v0.4 — ressources
    resources: dict[str, Any] | None = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _host_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def run_full_report(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None = None,
    max_tokens: int = 128,
    latency_runs: int = 5,
    warmup: int = 1,
    concurrencies: list[int] | None = None,
    requests_per_level: int = 8,
    price_in_per_mtok: float | None = None,
    price_out_per_mtok: float | None = None,
    measure_resources: bool = True,
    gpu_index: int = 0,
    on_event=None,
) -> UnifiedReport:
    """
    Pipeline complet. `on_event(stage, payload)` est appelé pour le streaming
    des résultats à l'écran. Tout est best-effort : coût/ressources sont
    optionnels et n'arrêtent pas le rapport s'ils échouent.
    """
    report = UnifiedReport(
        model=model,
        base_url=base_url,
        timestamp_utc=_now_iso(),
        host=_host_info(),
    )
    notify = on_event or (lambda *_: None)

    monitor_ctx = VRAMMonitor(gpu_index=gpu_index) if measure_resources else None
    if monitor_ctx is not None:
        monitor_ctx.__enter__()

    try:
        # ── v0.1 — Latence ──────────────────────────────────────────────
        notify("stage", "latency")
        for _ in range(warmup):
            measure_latency_once(
                base_url=base_url, model=model, prompt=prompt,
                api_key=api_key, max_tokens=max_tokens,
            )
        results: list[LatencyResult] = []
        for i in range(latency_runs):
            r = measure_latency_once(
                base_url=base_url, model=model, prompt=prompt,
                api_key=api_key, max_tokens=max_tokens,
            )
            results.append(r)
            notify("latency_run", (i + 1, latency_runs, r))
        report.latency_summary = summarize(results)
        report.latency_runs = [asdict(r) for r in results]

        # ── v0.2 — Débit ────────────────────────────────────────────────
        if concurrencies:
            notify("stage", "throughput")
            points: list[ThroughputPoint] = sweep_throughput(
                base_url=base_url, model=model, prompt=prompt,
                concurrencies=concurrencies,
                requests_per_level=requests_per_level,
                api_key=api_key, max_tokens=max_tokens,
                on_point=lambda p: notify("throughput_point", p),
            )
            report.throughput_curve = [
                {
                    "concurrency": p.concurrency,
                    "n_requests": p.n_requests,
                    "wall_time_s": p.wall_time_s,
                    "total_output_tokens": p.total_output_tokens,
                    "system_tokens_per_sec": p.system_tokens_per_sec,
                    "ttft_median_ms": p.ttft_median_ms,
                    "tpot_median_ms": p.tpot_median_ms,
                }
                for p in points
            ]
            if points:
                best = max(points, key=lambda p: p.system_tokens_per_sec)
                report.throughput_best_tokens_per_sec = best.system_tokens_per_sec
                report.throughput_best_concurrency = best.concurrency

        # ── v0.3 — Coût ─────────────────────────────────────────────────
        notify("stage", "cost")
        try:
            cb: CostBreakdown = cost_from_results(
                results, model=model,
                price_in_per_mtok=price_in_per_mtok,
                price_out_per_mtok=price_out_per_mtok,
                fallback_prompt_tokens=max(1, len(prompt) // 4),
            )
            report.cost = asdict(cb)
        except ValueError as e:
            report.cost = {"error": str(e)}

    finally:
        # ── v0.4 — Ressources ───────────────────────────────────────────
        if monitor_ctx is not None:
            monitor_ctx.__exit__(None, None, None)
            usage: ResourceUsage = monitor_ctx.result()
            report.resources = asdict(usage)

    notify("done", report)
    return report


def report_to_json(report: UnifiedReport) -> str:
    return json.dumps(asdict(report), indent=2, default=str)
