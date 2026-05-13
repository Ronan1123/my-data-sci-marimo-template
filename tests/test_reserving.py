"""Tests for src/models/reserving.py"""

import pytest

cl = pytest.importorskip("chainladder")

from src.models.reserving import (
    apply_custom_transform,
    fit_bootstrap,
    fit_chainladder,
    fit_mack,
    summarise_reserves,
    triangle_diagnostics,
)


@pytest.fixture
def raa_triangle():
    return cl.load_sample("raa")


def test_fit_chainladder_returns_model(raa_triangle):
    model = fit_chainladder(raa_triangle)
    assert hasattr(model, "ibnr_")


def test_fit_mack_returns_model(raa_triangle):
    model = fit_mack(raa_triangle)
    assert hasattr(model, "ibnr_")


def test_fit_bootstrap_returns_model(raa_triangle):
    model = fit_bootstrap(raa_triangle, n_sims=100)
    assert hasattr(model, "ibnr_")


def test_summarise_reserves_columns(raa_triangle):
    import polars as pl

    model = fit_chainladder(raa_triangle)
    df = summarise_reserves(model)
    assert isinstance(df, pl.DataFrame)
    assert "origin" in df.columns
    assert "ibnr" in df.columns
    assert "ultimate" in df.columns


def test_summarise_reserves_ibnr_positive(raa_triangle):
    model = fit_chainladder(raa_triangle)
    df = summarise_reserves(model)
    total = df["ibnr"].sum()
    assert total > 0


def test_triangle_diagnostics_saves_file(raa_triangle, tmp_path):
    triangle_diagnostics(raa_triangle, save_dir=tmp_path)
    files = list(tmp_path.glob("*.png"))
    assert len(files) >= 1


def test_apply_custom_transform(raa_triangle):
    result = apply_custom_transform(raa_triangle, lambda t: t.grain("CY"))
    assert result is not None
