"""Central configuration: paths, seeds, display options, and version checks."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED: int = 1123

# ── Debug mode ─────────────────────────────────────────────────────────────────
DEBUG: bool = os.getenv("DEBUG", "0") == "1"

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
DIAGNOSTICS_DIR: Path = OUTPUTS_DIR / "diagnostics"

# Ensure output directories exist at import time
for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, DIAGNOSTICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Display ────────────────────────────────────────────────────────────────────
DISPLAY: dict = {
    "max_rows": 50,
    "max_cols": 20,
    "float_format": "{:.4f}",
    "str_len": 80,
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Version requirements ────────────────────────────────────────────────────────
_REQUIRED: dict[str, str] = {
    "marimo": "0.9.0",
    "polars": "1.0.0",
    "numpy": "1.26.0",
    "sklearn": "1.5.0",
    "chainladder": "0.8.0",
}


def _parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split(".")[:3])


def check_versions(raise_on_mismatch: bool = False) -> dict[str, str]:
    """Print installed versions and warn if below the required minimums."""
    import importlib

    _module_map = {
        "marimo": "marimo",
        "polars": "polars",
        "numpy": "numpy",
        "sklearn": "sklearn",
        "chainladder": "chainladder",
    }

    results: dict[str, str] = {}
    all_ok = True

    col_w = max(len(k) for k in _required_keys()) + 2
    print(f"\n{'Package':<{col_w}} {'Installed':<12} {'Required':<12} Status")
    print("-" * (col_w + 36))

    for name, mod_name in _module_map.items():
        try:
            mod = importlib.import_module(mod_name)
            installed = getattr(mod, "__version__", "unknown")
            results[name] = installed
            required = _REQUIRED[name]
            ok = _parse_version(installed) >= _parse_version(required)
            status = "OK" if ok else "WARN (below minimum)"
            if not ok:
                all_ok = False
            print(f"{name:<{col_w}} {installed:<12} {required:<12} {status}")
        except ImportError:
            results[name] = "NOT INSTALLED"
            all_ok = False
            print(f"{name:<{col_w}} {'NOT INSTALLED':<12} {_REQUIRED[name]:<12} MISSING")

    print()
    if not all_ok:
        msg = "One or more packages are missing or below their required version."
        if raise_on_mismatch:
            raise RuntimeError(msg)
        logger.warning(msg)

    return results


def _required_keys() -> list[str]:
    return list(_REQUIRED.keys())
