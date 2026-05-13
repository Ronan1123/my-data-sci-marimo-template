"""Feature engineering: polars expressions and an sklearn-compatible transformer."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def cast_columns(lf: pl.LazyFrame, schema_map: dict[str, pl.PolarsDataType]) -> pl.LazyFrame:
    """Cast named columns to specified polars dtypes."""
    exprs = [pl.col(col).cast(dtype) for col, dtype in schema_map.items()]
    logger.debug("Casting %d columns", len(schema_map))
    return lf.with_columns(exprs)


def drop_nulls_threshold(lf: pl.LazyFrame, threshold: float = 0.5) -> pl.LazyFrame:
    """Drop columns where the null fraction exceeds `threshold` (0–1)."""
    df = lf.collect()
    n = len(df)
    cols_to_drop = [col for col in df.columns if df[col].null_count() / n > threshold]
    if cols_to_drop:
        logger.info("Dropping %d high-null columns: %s", len(cols_to_drop), cols_to_drop)
    return lf.drop(cols_to_drop)


def log_transform(lf: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    """Apply log1p to numeric columns (handles zeros safely)."""
    exprs = [pl.col(c).log1p().alias(f"{c}_log1p") for c in cols]
    logger.debug("Applying log1p to: %s", cols)
    return lf.with_columns(exprs)


def select_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep only numeric columns."""
    numeric_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64]
    cols = [c for c, dtype in lf.schema.items() if dtype in numeric_types]
    return lf.select(cols)


class PolarsTransformer(BaseEstimator, TransformerMixin):
    """
    Wrap a sequence of polars LazyFrame transforms into an sklearn Transformer.

    Parameters
    ----------
    transforms : list of callables (LazyFrame -> LazyFrame)
    """

    def __init__(self, transforms: list[Callable[[pl.LazyFrame], pl.LazyFrame]]):
        self.transforms = transforms

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Accept a polars DataFrame/LazyFrame or numpy array, return numpy array."""
        if isinstance(X, np.ndarray):
            lf = pl.from_numpy(X).lazy()
        elif isinstance(X, pl.DataFrame):
            lf = X.lazy()
        elif isinstance(X, pl.LazyFrame):
            lf = X
        else:
            raise TypeError(f"Expected polars DataFrame/LazyFrame or ndarray, got {type(X)}")

        for fn in self.transforms:
            lf = fn(lf)

        return lf.collect().to_numpy()
