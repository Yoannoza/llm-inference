"""
Famille LATENCE — métriques-mères : TTFT et TPOT.

Mesure en streaming sur n'importe quelle API compatible OpenAI :
  - OpenAI, Anthropic (via proxy compat), Together, Groq, Fireworks, DeepSeek...
  - Serveur local : vLLM, llama.cpp en mode serveur, Ollama (avec /v1)

Définitions (alignées sur le draft IETF draft-gaikwad-llm-benchmarking-terminology) :
  - TTFT (Time To First Token) : délai entre la requête et le 1er token reçu.
  - TPOT (Time Per Output Token) : (latence_totale - TTFT) / (n_tokens - 1).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from statistics import median
from typing import Iterable

import httpx


@dataclass
class LatencyResult:
    """Résultat d'UNE requête mesurée."""
    ttft_ms: float            # Time To First Token, en millisecondes
    tpot_ms: float | None     # Time Per Output Token (None si <2 tokens générés)
    total_ms: float           # latence de bout en bout
    n_output_tokens: int      # nombre de tokens générés (compté côté client)
    prompt_chars: int         # taille du prompt envoyé (pour traçabilité)
    # Tokens rapportés par l'API si dispo (stream_options.include_usage).
    # Précis pour le calcul de coût ; sinon on retombe sur n_output_tokens.
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None

    def __str__(self) -> str:
        tpot = f"{self.tpot_ms:.1f}" if self.tpot_ms is not None else "n/a"
        return (
            f"TTFT={self.ttft_ms:7.1f} ms | "
            f"TPOT={tpot:>6} ms/tok | "
            f"total={self.total_ms:7.1f} ms | "
            f"tokens={self.n_output_tokens}"
        )


def measure_latency_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None = None,
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> LatencyResult:
    """
    Envoie UNE requête en streaming et chronomètre TTFT + TPOT.

    Paramètres
    ----------
    base_url : URL racine de l'API compatible OpenAI (sans /chat/completions).
        Ex. "https://api.openai.com/v1", "http://localhost:8000/v1".
    model : identifiant du modèle (ex. "gpt-4o-mini", "Qwen/Qwen2.5-3B-Instruct").
    prompt : texte utilisateur unique (on construit un message {role: "user"}).
    api_key : clé d'API si nécessaire ; None pour un serveur local sans auth.
    max_tokens : plafond de tokens générés.
    temperature : 0.0 par défaut pour rendre les mesures reproductibles.
    timeout : délai max global de la requête.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Demande à l'API d'envoyer un chunk final "usage" (tokens exacts).
        # Ignoré silencieusement par les APIs qui ne le supportent pas.
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first: float | None = None
    n_tokens = 0
    usage_in: int | None = None
    usage_out: int | None = None

    with httpx.stream(
        "POST", url, headers=headers, json=payload, timeout=timeout
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            # Le format SSE OpenAI préfixe chaque ligne par "data: ".
            line = raw_line.removeprefix("data: ").strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if usage:
                usage_in = usage.get("prompt_tokens")
                usage_out = usage.get("completion_tokens")

            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {}) if choices else {}
            content = delta.get("content")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                n_tokens += 1

    t_end = time.perf_counter()

    if t_first is None:
        raise RuntimeError("Aucun token reçu — vérifier l'API, le modèle, la clé.")

    ttft_ms = (t_first - t_start) * 1000
    total_ms = (t_end - t_start) * 1000
    tpot_ms = (total_ms - ttft_ms) / (n_tokens - 1) if n_tokens >= 2 else None

    return LatencyResult(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_ms=total_ms,
        n_output_tokens=n_tokens,
        prompt_chars=len(prompt),
        usage_input_tokens=usage_in,
        usage_output_tokens=usage_out,
    )


def summarize(results: Iterable[LatencyResult]) -> dict[str, float]:
    """Statistiques agrégées (médianes, plus robustes que les moyennes)."""
    results = list(results)
    ttfts = [r.ttft_ms for r in results]
    tpots = [r.tpot_ms for r in results if r.tpot_ms is not None]
    return {
        "n_runs": len(results),
        "ttft_median_ms": median(ttfts),
        "ttft_min_ms": min(ttfts),
        "ttft_max_ms": max(ttfts),
        "tpot_median_ms": median(tpots) if tpots else float("nan"),
        "tokens_per_sec_median": 1000 / median(tpots) if tpots else float("nan"),
    }
