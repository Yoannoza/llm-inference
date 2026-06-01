"""Installation de llama.cpp.

Stratégie :
- Mac (Darwin) : `brew install llama.cpp` (binaire `llama-server` dispo).
- Linux + NVIDIA : build depuis source avec `cmake -DGGML_CUDA=ON`.
- Linux sans GPU : build CPU.

On considère llama.cpp installé si `llama-server` est dans le PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .detect import Env
from .log import StepLogger


LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
BUILD_DIR = Path.home() / ".cache" / "infer-serve" / "llama.cpp"


def llama_server_path() -> str | None:
    """Trouve `llama-server` dans le PATH ou dans notre build local."""
    p = shutil.which("llama-server")
    if p:
        return p
    local = BUILD_DIR / "build" / "bin" / "llama-server"
    if local.exists():
        return str(local)
    return None


def _run(cmd: list[str], log: StepLogger, cwd: Path | None = None) -> None:
    log.info(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error(proc.stderr.strip().splitlines()[-1] if proc.stderr else "échec")
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def install_mac(log: StepLogger) -> str:
    if not shutil.which("brew"):
        raise RuntimeError(
            "Homebrew introuvable. Installe-le depuis https://brew.sh puis relance."
        )
    _run(["brew", "install", "llama.cpp"], log)
    p = shutil.which("llama-server")
    if not p:
        raise RuntimeError("brew a réussi mais llama-server reste introuvable")
    return p


def install_linux(env: Env, log: StepLogger) -> str:
    for tool in ("git", "cmake", "make"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"`{tool}` requis pour build llama.cpp. "
                "Sur Debian/Ubuntu : `sudo apt install -y git cmake build-essential`"
            )

    BUILD_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not BUILD_DIR.exists():
        log.info(f"clone → {BUILD_DIR}")
        _run(["git", "clone", "--depth", "1", LLAMA_REPO, str(BUILD_DIR)], log)
    else:
        log.info(f"repo déjà présent → {BUILD_DIR} (pull)")
        _run(["git", "-C", str(BUILD_DIR), "pull", "--ff-only"], log)

    build = BUILD_DIR / "build"
    cmake_args = ["cmake", "-S", str(BUILD_DIR), "-B", str(build)]
    if env.cuda_available:
        log.info("build avec CUDA (GGML_CUDA=ON)")
        cmake_args += ["-DGGML_CUDA=ON"]
    else:
        log.info("build CPU (pas de GPU détecté)")

    _run(cmake_args, log)
    _run(["cmake", "--build", str(build), "--target", "llama-server",
          "-j", str(max(1, (Path("/proc/cpuinfo").read_text().count('processor') or 4) // 2))],
         log)

    bin_path = build / "bin" / "llama-server"
    if not bin_path.exists():
        raise RuntimeError(f"build terminé mais binaire introuvable : {bin_path}")
    return str(bin_path)


def ensure_installed(env: Env, log: StepLogger) -> str:
    existing = llama_server_path()
    if existing:
        log.info(f"llama-server déjà installé → {existing}")
        return existing

    if env.os == "Darwin":
        return install_mac(log)
    if env.os == "Linux":
        return install_linux(env, log)
    raise RuntimeError(f"OS non supporté pour install auto : {env.os}")
