"""Tests for src/features/transformers.py"""

import numpy as np
import polars as pl
import pytest

from src.features.transformers import (
    PolarsTransformer,
    cast_columns,
    drop_nulls_threshold,
    log_transform,
    select_numeric,
)


@pytest.fixture
def base_df():
    return pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [1.0, None, None, None, 5.0],
        "c": ["x", "y", None, None, "z"],
        "d": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


def test_cast_columns(base_df):
    lf = base_df.lazy()
    result = cast_columns(lf, {"a": pl.Float64}).collect()
    assert result["a"].dtype == pl.Float64


def test_drop_nulls_threshold_removes_high_null_col(base_df):
    lf = base_df.lazy()
    result = drop_nulls_threshold(lf, threshold=0.5).collect()
    assert "b" not in result.columns  # 3/5 = 0.6 > 0.5
    assert "a" in result.columns


def test_drop_nulls_threshold_keeps_low_null_col(base_df):
    lf = base_df.lazy()
    result = drop_nulls_threshold(lf, threshold=0.7).collect()
    assert "b" in result.columns  # 3/5 = 0.6 < 0.7


def test_log_transform_adds_columns(base_df):
    lf = base_df.select(["a", "d"]).lazy()
    result = log_transform(lf, ["a", "d"]).collect()
    assert "a_log1p" in result.columns
    assert "d_log1p" in result.columns


def test_log_transform_values_correct(base_df):
    lf = base_df.select(["d"]).lazy()
    result = log_transform(lf, ["d"]).collect()
    expected = np.log1p(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    np.testing.assert_allclose(result["d_log1p"].to_numpy(), expected)


def test_select_numeric(base_df):
    result = select_numeric(base_df.lazy()).collect()
    assert "c" not in result.columns
    assert "a" in result.columns
    assert "d" in result.columns


def test_polars_transformer_roundtrip():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    transformer = PolarsTransformer(transforms=[])
    result = transformer.fit_transform(df)
    assert result.shape == (3, 2)
