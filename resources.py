"""
Famille RESSOURCES — métrique-mère : VRAM utilisée (MiB), peak observé.

Pour un déploiement local (vLLM, llama.cpp, Ollama), savoir combien de
VRAM le modèle consomme conditionne directement :
  - le batch size maximum,
  - le KV-cache disponible (donc la longueur de contexte exploitable),
  - le choix GPU (24 Go ? 48 Go ? 80 Go ?).

Implémentation : on poll `nvidia-smi` à intervalle régulier dans un thread
pendant qu'un workload tourne, et on renvoie le pic. Volontairement
simple : pas de NVML, pas de cgroups — juste un sous-processus.

Limite : `nvidia-smi` ne distingue pas notre processus des autres
locataires de la GPU. À utiliser sur une GPU dédiée au test, ou faire
la différence avant/après.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class ResourceUsage:
    """Utilisation observée pendant un workload."""
    available: bool          # False si nvidia-smi absent (ex. CPU-only / Mac)
    gpu_index: int
    vram_used_peak_mib: int  # métrique-mère
    vram_used_mean_mib: float
    vram_total_mib: int
    util_gpu_peak_pct: int   # % d'utilisation calcul, pic observé
    n_samples: int
    duration_s: float
    reason: str | None = None  # explication si available=False


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query_once(gpu_index: int) -> tuple[int, int, int] | None:
    """Retourne (vram_used_mib, vram_total_mib, util_pct) ou None si erreur."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None


class VRAMMonitor:
    """
    Context-manager qui sample nvidia-smi en arrière-plan.

    Usage :
        with VRAMMonitor(gpu_index=0, interval_s=0.1) as mon:
            ... workload ...
        print(mon.result())
    """

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.1):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples_used: list[int] = []
        self._samples_util: list[int] = []
        self._vram_total: int = 0
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self._reason: str | None = None

    def __enter__(self):
        if not _has_nvidia_smi():
            self._reason = "nvidia-smi introuvable (CPU-only ou pas de GPU NVIDIA)"
            return self
        # Sanity check : un seul query pour valider l'index.
        probe = _query_once(self.gpu_index)
        if probe is None:
            self._reason = f"impossible d'interroger GPU #{self.gpu_index}"
            return self
        self._vram_total = probe[1]
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            q = _query_once(self.gpu_index)
            if q is not None:
                self._samples_used.append(q[0])
                self._samples_util.append(q[2])
            self._stop.wait(self.interval_s)

    def __exit__(self, exc_type, exc, tb):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._t_end = time.perf_counter()
        return False

    def result(self) -> ResourceUsage:
        if not self._samples_used:
            return ResourceUsage(
                available=False, gpu_index=self.gpu_index,
                vram_used_peak_mib=0, vram_used_mean_mib=0.0,
                vram_total_mib=self._vram_total,
                util_gpu_peak_pct=0, n_samples=0, duration_s=0.0,
                reason=self._reason or "aucun échantillon collecté",
            )
        return ResourceUsage(
            available=True, gpu_index=self.gpu_index,
            vram_used_peak_mib=max(self._samples_used),
            vram_used_mean_mib=sum(self._samples_used) / len(self._samples_used),
            vram_total_mib=self._vram_total,
            util_gpu_peak_pct=max(self._samples_util) if self._samples_util else 0,
            n_samples=len(self._samples_used),
            duration_s=self._t_end - self._t_start,
        )
