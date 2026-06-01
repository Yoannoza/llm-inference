"""
Famille DÉBIT — métrique-mère : output tokens / second (système).

Le débit système est différent du débit par requête : ce qui compte pour
un opérateur d'inférence, c'est combien de tokens son serveur produit
**globalement** quand on lui envoie plusieurs requêtes en parallèle.

On mesure ça via un sweep de concurrence : on lance N requêtes simultanées,
on attend qu'elles finissent, et on calcule (tokens_totaux / temps_wall).

La « courbe latence/débit » classique apparaît en répétant pour plusieurs
niveaux de concurrence (1, 2, 4, 8, ...). À concurrence faible : latence
basse mais débit faible. À concurrence haute : débit qui sature et latence
qui explose (file d'attente). L'opérateur cherche le coude.

Aligné §6 du draft IETF draft-gaikwad-llm-benchmarking-terminology.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from statistics import median

from latency import LatencyResult, measure_latency_once


@dataclass
class ThroughputPoint:
    """Un point de la courbe latence/débit, à un niveau de concurrence donné."""
    concurrency: int
    n_requests: int
    wall_time_s: float            # temps total pour traiter le batch
    total_output_tokens: int
    system_tokens_per_sec: float  # métrique-mère : débit système
    ttft_median_ms: float
    tpot_median_ms: float
    per_request_results: list[LatencyResult] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"conc={self.concurrency:>3} | "
            f"débit={self.system_tokens_per_sec:7.1f} tok/s | "
            f"TTFT_med={self.ttft_median_ms:6.0f} ms | "
            f"TPOT_med={self.tpot_median_ms:5.1f} ms/tok"
        )


def measure_throughput_at(
    *,
    base_url: str,
    model: str,
    prompt: str,
    concurrency: int,
    n_requests: int,
    api_key: str | None = None,
    max_tokens: int = 128,
) -> ThroughputPoint:
    """
    Lance `n_requests` requêtes en utilisant un pool de `concurrency` workers.

    Le débit système est mesuré sur le temps wall-clock du batch entier
    — le bon dénominateur car c'est ce que voit l'opérateur.
    """
    if n_requests < concurrency:
        n_requests = concurrency

    def one() -> LatencyResult:
        return measure_latency_once(
            base_url=base_url, model=model, prompt=prompt,
            api_key=api_key, max_tokens=max_tokens,
        )

    t0 = time.perf_counter()
    results: list[LatencyResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one) for _ in range(n_requests)]
        for f in as_completed(futures):
            results.append(f.result())
    wall = time.perf_counter() - t0

    total_tokens = sum(r.n_output_tokens for r in results)
    ttfts = [r.ttft_ms for r in results]
    tpots = [r.tpot_ms for r in results if r.tpot_ms is not None]

    return ThroughputPoint(
        concurrency=concurrency,
        n_requests=n_requests,
        wall_time_s=wall,
        total_output_tokens=total_tokens,
        system_tokens_per_sec=total_tokens / wall if wall > 0 else 0.0,
        ttft_median_ms=median(ttfts) if ttfts else float("nan"),
        tpot_median_ms=median(tpots) if tpots else float("nan"),
        per_request_results=results,
    )


def sweep_throughput(
    *,
    base_url: str,
    model: str,
    prompt: str,
    concurrencies: list[int],
    requests_per_level: int,
    api_key: str | None = None,
    max_tokens: int = 128,
    on_point=None,
) -> list[ThroughputPoint]:
    """Balaye plusieurs niveaux de concurrence — produit la courbe."""
    points: list[ThroughputPoint] = []
    for c in concurrencies:
        p = measure_throughput_at(
            base_url=base_url, model=model, prompt=prompt,
            concurrency=c, n_requests=max(requests_per_level, c),
            api_key=api_key, max_tokens=max_tokens,
        )
        points.append(p)
        if on_point:
            on_point(p)
    return points
