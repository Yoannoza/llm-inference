"""
Famille COÛT — métrique-mère : $ / 1M tokens de sortie.

L'idée : à partir des tokens consommés (input + output) sur un run réel,
calculer le coût observé en $ et le ramener à une unité standard
($/1M tokens output) pour comparer des fournisseurs entre eux.

Important : les tarifs ci-dessous sont des **valeurs publiques connues**
au moment de la rédaction. Ils peuvent bouger ; le `--price-in` et
`--price-out` permettent toujours de surcharger à la main pour un modèle
non listé ou un tarif négocié.
"""

from __future__ import annotations

from dataclasses import dataclass

from latency import LatencyResult


# Tarifs publics en $ / 1M tokens (input, output).
# Source : pages tarifaires officielles. À surcharger si périmé.
PRICE_CATALOG: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":             (2.50, 10.00),
    "gpt-4o-mini":        (0.15,  0.60),
    "gpt-4.1":            (2.00,  8.00),
    "gpt-4.1-mini":       (0.40,  1.60),
    "gpt-4.1-nano":       (0.10,  0.40),
    "o1":                 (15.00, 60.00),
    "o1-mini":            (3.00, 12.00),
    "o3-mini":            (1.10,  4.40),
    # Anthropic (via API compat)
    "claude-3-5-sonnet":  (3.00, 15.00),
    "claude-3-5-haiku":   (0.80,  4.00),
    "claude-3-opus":     (15.00, 75.00),
    # DeepSeek
    "deepseek-chat":      (0.27,  1.10),
    "deepseek-reasoner":  (0.55,  2.19),
    # Groq (Llama)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant":    (0.05, 0.08),
    # Together (quelques modèles populaires)
    "Qwen/Qwen2.5-72B-Instruct-Turbo": (1.20, 1.20),
}


@dataclass
class CostBreakdown:
    """Coût observé pour un run (ou un batch agrégé)."""
    model: str
    price_in_per_mtok: float    # $ / 1M tokens input
    price_out_per_mtok: float   # $ / 1M tokens output
    input_tokens: int
    output_tokens: int
    cost_input_usd: float
    cost_output_usd: float
    cost_total_usd: float
    # Métrique-mère : ramené à 1M tokens de sortie pour la comparaison.
    effective_dollars_per_mtok_out: float

    def __str__(self) -> str:
        return (
            f"in={self.input_tokens} tok @ ${self.price_in_per_mtok}/Mtok | "
            f"out={self.output_tokens} tok @ ${self.price_out_per_mtok}/Mtok | "
            f"total=${self.cost_total_usd:.6f} | "
            f"=> ${self.effective_dollars_per_mtok_out:.2f} / 1M tok output"
        )


def lookup_price(model: str) -> tuple[float, float] | None:
    """Recherche tarif exact puis fallback par préfixe (ex. provider/model)."""
    if model in PRICE_CATALOG:
        return PRICE_CATALOG[model]
    # Match suffix : ex. "groq/llama-3.1-8b-instant" → cherche la fin.
    for key, price in PRICE_CATALOG.items():
        if model.endswith(key) or model.endswith(key.split("/")[-1]):
            return price
    return None


def compute_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    price_in_per_mtok: float | None = None,
    price_out_per_mtok: float | None = None,
) -> CostBreakdown:
    """Calcule le coût d'un run. Prix explicites prioritaires sur le catalogue."""
    if price_in_per_mtok is None or price_out_per_mtok is None:
        looked = lookup_price(model)
        if looked is None:
            raise ValueError(
                f"Tarif inconnu pour le modèle '{model}'. "
                f"Passe --price-in et --price-out, ou ajoute-le au catalogue."
            )
        price_in_per_mtok = price_in_per_mtok or looked[0]
        price_out_per_mtok = price_out_per_mtok or looked[1]

    cost_in = (input_tokens / 1_000_000) * price_in_per_mtok
    cost_out = (output_tokens / 1_000_000) * price_out_per_mtok
    total = cost_in + cost_out

    # Ramène à 1M tokens output : multiplie par (1M / output_tokens).
    if output_tokens > 0:
        effective = total * (1_000_000 / output_tokens)
    else:
        effective = float("nan")

    return CostBreakdown(
        model=model,
        price_in_per_mtok=price_in_per_mtok,
        price_out_per_mtok=price_out_per_mtok,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_input_usd=cost_in,
        cost_output_usd=cost_out,
        cost_total_usd=total,
        effective_dollars_per_mtok_out=effective,
    )


def cost_from_results(
    results: list[LatencyResult],
    *,
    model: str,
    price_in_per_mtok: float | None = None,
    price_out_per_mtok: float | None = None,
    fallback_prompt_tokens: int = 0,
) -> CostBreakdown:
    """
    Agrège plusieurs runs et calcule le coût total + effectif.

    Préfère les tokens rapportés par l'API (`usage_*`) ; sinon retombe sur
    le compte de chunks côté client pour l'output, et `fallback_prompt_tokens`
    pour l'input (estimation grossière du prompt).
    """
    in_tok = sum(
        r.usage_input_tokens if r.usage_input_tokens is not None
        else fallback_prompt_tokens
        for r in results
    )
    out_tok = sum(
        r.usage_output_tokens if r.usage_output_tokens is not None
        else r.n_output_tokens
        for r in results
    )
    return compute_cost(
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        price_in_per_mtok=price_in_per_mtok,
        price_out_per_mtok=price_out_per_mtok,
    )
