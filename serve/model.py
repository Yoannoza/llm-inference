"""Download GGUF + choix de quantisation.

Le modèle par défaut est `unsloth/gemma-4-E4B-it-GGUF` (override avec --model).
On choisit le fichier .gguf selon la mémoire dispo en cherchant un budget de
~80% de la VRAM (NVIDIA) ou ~60% de la RAM (Metal/CPU, mémoire partagée).

Préférence de quantisations (du meilleur au plus léger) :
  Q8_0 > Q6_K > Q5_K_M > Q4_K_M > Q3_K_M > Q2_K
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .detect import Env
from .log import StepLogger


CACHE_DIR = Path.home() / ".cache" / "infer-serve" / "models"

QUANT_PREFERENCE = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
                    "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K"]


@dataclass
class ModelPick:
    repo_id: str
    filename: str
    quant: str
    size_mib: int
    local_path: str


def _budget_mib(env: Env) -> int:
    if env.vram_total_mib:
        return int(env.vram_total_mib * 0.80)
    # Metal/CPU : mémoire partagée → on prend une fraction de la RAM.
    return int(env.ram_gb * 1024 * 0.60)


def _quant_of(filename: str) -> str | None:
    m = re.search(r"(Q\d(?:_K(?:_[SM])?|_\d)?)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


def list_gguf_files(repo_id: str) -> list[tuple[str, int]]:
    """[(filename, size_bytes)] pour tous les .gguf du repo."""
    api = HfApi()
    info = api.repo_info(repo_id, files_metadata=True)
    out = []
    for f in info.siblings:
        if f.rfilename.lower().endswith(".gguf") and f.size:
            out.append((f.rfilename, f.size))
    return out


def pick_file(env: Env, repo_id: str, requested_quant: str | None,
              log: StepLogger) -> tuple[str, int, str]:
    """Choisit (filename, size_mib, quant) selon le budget mémoire."""
    files = list_gguf_files(repo_id)
    if not files:
        raise RuntimeError(f"Aucun .gguf trouvé dans {repo_id}")

    budget = _budget_mib(env)
    log.info(f"budget mémoire estimé : {budget} MiB")

    annotated = []
    for fn, size in files:
        q = _quant_of(fn)
        size_mib = size // (1024 * 1024)
        annotated.append((fn, size_mib, q))

    if requested_quant:
        q = requested_quant.upper()
        matches = [(fn, sz, qq) for fn, sz, qq in annotated if qq == q]
        if not matches:
            raise RuntimeError(
                f"Quantisation {q} introuvable dans {repo_id}. "
                f"Disponibles : {sorted({qq for _, _, qq in annotated if qq})}"
            )
        fn, sz, qq = matches[0]
        if sz > budget:
            log.warn(f"{fn} ({sz} MiB) > budget ({budget} MiB) — forcé quand même")
        return fn, sz, qq

    by_quant = {qq: (fn, sz) for fn, sz, qq in annotated if qq}
    for q in QUANT_PREFERENCE:
        if q in by_quant:
            fn, sz = by_quant[q]
            if sz <= budget:
                return fn, sz, q

    # Rien ne rentre : on prend le plus petit dispo et on prévient.
    fn, sz, q = min(annotated, key=lambda t: t[1])
    log.warn(f"aucune quant ne rentre dans le budget — fallback {fn} ({sz} MiB)")
    return fn, sz, q or "?"


def download(repo_id: str, filename: str, log: StepLogger) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"download → {repo_id} :: {filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(CACHE_DIR),
    )
    return path


def fetch_model(env: Env, repo_id: str, requested_quant: str | None,
                log: StepLogger) -> ModelPick:
    filename, size_mib, quant = pick_file(env, repo_id, requested_quant, log)
    log.info(f"sélection : {filename} ({size_mib} MiB, {quant})")
    local = download(repo_id, filename, log)
    return ModelPick(
        repo_id=repo_id, filename=filename, quant=quant,
        size_mib=size_mib, local_path=local,
    )
