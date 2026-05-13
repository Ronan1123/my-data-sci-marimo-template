"""Chainladder reserving workflows: CL, Mack, Bootstrap, diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import polars as pl

logger = logging.getLogger(__name__)


def _require_cl():
    try:
        import chainladder as cl
        return cl
    except ImportError as exc:
        raise ImportError("chainladder is required for reserving workflows.") from exc


# ── Development methods ────────────────────────────────────────────────────────

def fit_chainladder(triangle):
    """Fit a basic volume-weighted chain-ladder development model."""
    cl = _require_cl()
    model = cl.Chainladder().fit(triangle)
    logger.info("Chainladder fitted | tail: %s", triangle.shape)
    return model


def fit_mack(triangle):
    """Fit a Mack chain-ladder model (variance + confidence intervals)."""
    cl = _require_cl()
    model = cl.MackChainladder().fit(triangle)
    logger.info("Mack model fitted")
    return model


def fit_bootstrap(triangle, n_sims: int = 1000, random_state: int | None = 42):
    """Fit a Bootstrap ODP chain-ladder for reserve distributions."""
    cl = _require_cl()
    model = cl.BootstrapODPChain(n_sims=n_sims, random_state=random_state).fit(triangle)
    logger.info("Bootstrap ODP fitted | n_sims=%d", n_sims)
    return model


# ── Reserve summaries ──────────────────────────────────────────────────────────

def summarise_reserves(model) -> pl.DataFrame:
    """
    Extract IBNR by origin period from a fitted chainladder model.
    Returns a polars DataFrame with columns [origin, ibnr, ultimate].
    """
    ibnr = model.ibnr_.to_frame(origin_as_datetime=False)
    ultimate = model.ultimate_.to_frame(origin_as_datetime=False)

    ibnr_df = pl.from_pandas(ibnr.reset_index()).rename({"index": "origin"})
    ult_df = pl.from_pandas(ultimate.reset_index()).rename({"index": "origin"})

    ibnr_col = ibnr_df.columns[-1]
    ult_col = ult_df.columns[-1]

    result = ibnr_df.join(ult_df, on="origin").rename(
        {ibnr_col: "ibnr", ult_col: "ultimate"}
    )
    total_ibnr = result["ibnr"].sum()
    logger.info("Total IBNR: {:,.0f}".format(total_ibnr))
    return result


# ── Diagnostics ────────────────────────────────────────────────────────────────

def triangle_diagnostics(triangle, save_dir: str | Path | None = None) -> None:
    """
    Plot development factors, tail factors, and residuals.
    Saves figures to save_dir (defaults to outputs/diagnostics/).
    """
    from src.utils.config import DIAGNOSTICS_DIR

    cl = _require_cl()
    save_dir = Path(save_dir) if save_dir else DIAGNOSTICS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    dev = cl.Development().fit(triangle)
    ldfs = dev.ldf_.to_frame(origin_as_datetime=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Development factors
    axes[0].bar(range(len(ldfs.columns)), ldfs.iloc[0].values)
    axes[0].set_title("Age-to-Age Development Factors")
    axes[0].set_xlabel("Development Period")
    axes[0].set_ylabel("LDF")
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=0.8)

    # Cumulative development
    cdf = ldfs.iloc[0].cumprod()[::-1]
    axes[1].plot(range(len(cdf)), cdf.values, marker="o")
    axes[1].set_title("Cumulative Development Factors (CDF to Ultimate)")
    axes[1].set_xlabel("Development Period")
    axes[1].set_ylabel("CDF")
    axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out = save_dir / "triangle_development_factors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnostics plot to %s", out)


# ── Extensibility ──────────────────────────────────────────────────────────────

def apply_custom_transform(triangle, transform_fn: Callable):
    """Apply a user-supplied transform function to a triangle and return the result."""
    logger.debug("Applying custom triangle transform: %s", transform_fn.__name__)
    return transform_fn(triangle)
