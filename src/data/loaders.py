"""Data loading utilities: CSV/Parquet via polars LazyFrames, chainladder triangles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import polars as pl

logger = logging.getLogger(__name__)


def load_csv(path: str | Path, **scan_kwargs) -> pl.LazyFrame:
    """Scan a CSV file as a polars LazyFrame."""
    path = Path(path)
    logger.info("Scanning CSV: %s", path)
    return pl.scan_csv(path, **scan_kwargs)


def load_parquet(path: str | Path, **scan_kwargs) -> pl.LazyFrame:
    """Scan a Parquet file as a polars LazyFrame."""
    path = Path(path)
    logger.info("Scanning Parquet: %s", path)
    return pl.scan_parquet(path, **scan_kwargs)


def save_processed(df: pl.DataFrame, filename: str) -> Path:
    """Write a collected DataFrame to data/processed/ as Parquet."""
    from src.utils.config import DATA_PROCESSED_DIR

    out = DATA_PROCESSED_DIR / filename
    df.write_parquet(out)
    logger.info("Saved processed data to %s", out)
    return out


def inspect_schema(lf: pl.LazyFrame) -> dict:
    """Collect schema info: dtypes, row count, null counts."""
    schema = lf.schema
    df = lf.collect()
    null_counts = {col: df[col].null_count() for col in df.columns}
    n_rows = len(df)

    logger.info("Schema: %d rows × %d cols", n_rows, len(schema))
    print(f"\nRows: {n_rows:,}  |  Columns: {len(schema)}")
    print(f"\n{'Column':<30} {'Dtype':<20} {'Nulls':>8}")
    print("-" * 62)
    for col, dtype in schema.items():
        print(f"{col:<30} {str(dtype):<20} {null_counts[col]:>8}")
    print()

    return {"schema": schema, "n_rows": n_rows, "null_counts": null_counts}


def load_triangle(
    df: pl.DataFrame,
    index: str | list[str],
    columns: str,
    values: str,
    aggfunc: str = "sum",
    origin_format: str | None = None,
    development_format: str | None = None,
):
    """
    Build a chainladder Triangle from a polars DataFrame.

    Parameters
    ----------
    df : collected polars DataFrame
    index : origin period column(s)
    columns : development period column
    values : loss/claim amount column
    aggfunc : aggregation function name ('sum', 'mean', etc.)
    origin_format : strftime format for origin column if string dates
    development_format : strftime format for development column if string dates
    """
    try:
        import chainladder as cl
    except ImportError as exc:
        raise ImportError("chainladder is required for triangle loading.") from exc

    pandas_df = df.to_pandas()
    logger.info(
        "Building Triangle — index=%s, columns=%s, values=%s", index, columns, values
    )
    triangle = cl.Triangle(
        data=pandas_df,
        origin=index,
        development=columns,
        columns=values,
        aggfunc=aggfunc,
        origin_format=origin_format,
        development_format=development_format,
    )
    logger.info("Triangle shape: %s", triangle.shape)
    return triangle
