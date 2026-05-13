"""Tests for src/data/loaders.py"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.data.loaders import inspect_schema, load_csv, load_parquet, save_processed


@pytest.fixture
def sample_csv(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("a,b,c\n1,2.0,x\n3,4.0,y\n5,,z\n")
    return p


@pytest.fixture
def sample_df():
    return pl.DataFrame({"a": [1, 3, 5], "b": [2.0, 4.0, None], "c": ["x", "y", "z"]})


def test_load_csv_returns_lazyframe(sample_csv):
    lf = load_csv(sample_csv)
    assert isinstance(lf, pl.LazyFrame)


def test_load_csv_correct_shape(sample_csv):
    df = load_csv(sample_csv).collect()
    assert df.shape == (3, 3)


def test_load_parquet_roundtrip(tmp_path, sample_df):
    p = tmp_path / "sample.parquet"
    sample_df.write_parquet(p)
    lf = load_parquet(p)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().shape == sample_df.shape


def test_inspect_schema_returns_dict(sample_csv):
    lf = load_csv(sample_csv)
    info = inspect_schema(lf)
    assert "schema" in info
    assert "n_rows" in info
    assert "null_counts" in info
    assert info["n_rows"] == 3


def test_inspect_schema_null_counts(sample_csv):
    lf = load_csv(sample_csv)
    info = inspect_schema(lf)
    assert info["null_counts"]["b"] == 1


def test_load_triangle_shape():
    pytest.importorskip("chainladder")
    import chainladder as cl

    triangle = cl.load_sample("raa")
    assert triangle.shape is not None
    assert len(triangle.shape) == 4
