"""Détection de l'environnement : OS, arch, RAM, GPU, VRAM."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Literal


GpuKind = Literal["nvidia", "metal", "none"]


@dataclass
class Env:
    os: str                  # "Darwin" / "Linux"
    arch: str                # "arm64" / "x86_64"
    ram_gb: int
    gpu_kind: GpuKind
    gpu_name: str | None
    vram_total_mib: int | None   # None si GPU absent / Metal (shared mem)
    cuda_available: bool

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        gpu = self.gpu_name or "aucun"
        vram = f" ({self.vram_total_mib} MiB)" if self.vram_total_mib else ""
        return (f"OS: {self.os} ({self.arch}) | RAM: {self.ram_gb} GB | "
                f"GPU: {gpu}{vram} | CUDA: {'oui' if self.cuda_available else 'non'}")


def _ram_gb() -> int:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(int(out.strip()) / (1024**3))
        # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return int(kb / (1024**2))
    except Exception:
        pass
    return 0


def _detect_nvidia() -> tuple[str | None, int | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        first = out.strip().splitlines()[0]
        name, mem = [x.strip() for x in first.split(",")]
        return name, int(mem)
    except Exception:
        return None, None


def _detect_metal() -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        )
        brand = out.strip()
        if "Apple" in brand:
            return f"Apple {brand.replace('Apple ', '')}"
    except Exception:
        pass
    return "Apple Silicon"


def detect_env() -> Env:
    os_name = platform.system()
    arch = platform.machine()
    ram = _ram_gb()

    nv_name, nv_vram = _detect_nvidia()
    if nv_name:
        return Env(
            os=os_name, arch=arch, ram_gb=ram,
            gpu_kind="nvidia", gpu_name=nv_name,
            vram_total_mib=nv_vram, cuda_available=True,
        )

    metal = _detect_metal()
    if metal and os_name == "Darwin":
        return Env(
            os=os_name, arch=arch, ram_gb=ram,
            gpu_kind="metal", gpu_name=metal,
            vram_total_mib=None, cuda_available=False,
        )

    return Env(
        os=os_name, arch=arch, ram_gb=ram,
        gpu_kind="none", gpu_name=None,
        vram_total_mib=None, cuda_available=False,
    )
