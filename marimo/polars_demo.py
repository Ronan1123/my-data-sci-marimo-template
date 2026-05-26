import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="Polars with Marimo")


# ── 1. IMPORTS ─────────────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    return alt, mo, np, pl


# ── 2. TITLE ───────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polars with Marimo's Dataframe Transformer

    This notebook explores marimo's data UI capabilities alongside the power of Polars.
    We cover four approaches to interacting with data:

    1. Referencing a `pl.LazyFrame` directly
    2. Collecting to a `pl.DataFrame`
    3. Using `mo.ui.table` for row selection
    4. Using `mo.ui.dataframe` for interactive transformations

    Reference: [marimo dataframe guide](https://docs.marimo.io/guides/working_with_data/dataframes/)
    """)
    return


# ── 3. SYNTHETIC DATA ──────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loading Data

    We generate a synthetic orders dataset and load it as a `pl.LazyFrame` — so all
    transformations are evaluated lazily.

    Read more about `.lazy()` here: [Polars lazy API](https://docs.pola.rs/user-guide/lazy/)
    """)
    return


@app.cell
def _(np, pl):
    from datetime import date, timedelta

    rng = np.random.default_rng(42)
    n = 200

    _families = ["Electronics", "Clothing", "Food", "Home", "Sports"]
    _products = {
        "Electronics": ["Laptop", "Phone", "Tablet"],
        "Clothing": ["Shirt", "Pants", "Jacket"],
        "Food": ["Coffee", "Snacks", "Tea"],
        "Home": ["Lamp", "Chair", "Desk"],
        "Sports": ["Ball", "Racket", "Weights"],
    }

    _family_col = rng.choice(_families, size=n).tolist()
    _product_col = [rng.choice(_products[f]) for f in _family_col]
    _start = date(2024, 1, 1)
    _date_col = [_start + timedelta(days=int(d)) for d in rng.integers(0, 365, size=n)]

    demand: pl.LazyFrame = pl.DataFrame({
        "order_date": _date_col,
        "product_family": _family_col,
        "product": _product_col,
        "order_quantity": rng.integers(1, 100, size=n).tolist(),
    }).lazy()

    demand
    return date, demand, timedelta


# ── 4. LAZYFRAME VIEW ──────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you reference a `pl.LazyFrame` directly, marimo gives you two tabs out of the box:

    - **Table** — click `Preview data` for a quick look at your data
    - **Query Plan** — shows Polars' optimised execution plan

    💡 Try both tabs above!
    """)
    return


# ── 5. DATAFRAME VIEW ─────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference a `pl.DataFrame`

    Call `.collect()` to materialise the LazyFrame. Notice the richer column-level
    features: sorting, filtering, searching, and mini distribution charts on numeric columns.
    """)
    return


@app.cell
def _(demand):
    demand.collect()
    return


# ── 6. MO.UI.TABLE ────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `mo.ui.table` — Select Rows for Downstream Use

    `mo.ui.table` lets you select rows and pass them to cells further down the notebook.
    A useful pattern is to first summarise, then select categories you want to drill into.
    """)
    return


@app.cell
def _(demand, mo, pl):
    summary: pl.DataFrame = demand.group_by("product_family").agg(
        pl.mean("order_quantity").alias("mean"),
        pl.sum("order_quantity").alias("sum"),
        pl.std("order_quantity").alias("std"),
        pl.min("order_quantity").alias("min"),
        pl.max("order_quantity").alias("max"),
    ).sort("product_family").collect()
    summary_table = mo.ui.table(summary, label="Product Family Summary")
    summary_table
    return summary, summary_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Select one or more rows above. The cell below joins your selection back to the
    full dataset so you can explore the chosen families in detail.
    """)
    return


@app.cell
def _(demand, summary_table):
    selection_keys = (
        summary_table.value.select("product_family").unique()
    )
    selection = selection_keys.lazy().join(demand, on="product_family", how="left")
    selection.collect()
    return (selection_keys,)


# ── 7. MO.UI.DATAFRAME ────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `mo.ui.dataframe` — Interactive Transformations

    `mo.ui.dataframe` gives you a visual query builder. Use it to apply filters,
    group-bys, sorts, and column selections — and see the generated Polars code
    under the **Python Code** tab.

    💡 Try grouping by `product_family` and averaging `order_quantity`.
    """)
    return


@app.cell
def _(demand, mo):
    demand_cached = demand.collect()
    mo_dataframe = mo.ui.dataframe(demand_cached)
    mo_dataframe
    return demand_cached, mo_dataframe


# ── 8. POLARS AGGREGATION + CHART ─────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Polars Code: Group-By Aggregation

    Here is the same group-by aggregation written directly in Polars.
    Compare it to what the `mo.ui.dataframe` builder generated above.
    """)
    return


@app.cell
def _(demand_cached, pl):
    demand_agg = demand_cached.group_by("product_family").agg(
        pl.mean("order_quantity").alias("order_quantity_mean")
    ).sort("product_family")
    demand_agg
    return (demand_agg,)


@app.cell
def _(alt, demand_agg, mo):
    bar = (
        alt.Chart(demand_agg)
        .mark_bar()
        .encode(
            x=alt.X("product_family:N", title="Product Family", sort="-y"),
            y=alt.Y("order_quantity_mean:Q", title="Mean Quantity"),
            color=alt.Color("product_family:N", legend=None),
            tooltip=["product_family:N", alt.Tooltip("order_quantity_mean:Q", format=".1f")],
        )
        .properties(title="Mean Order Quantity by Product Family", width=500, height=300)
    )
    mo.ui.altair_chart(bar)
    return


# ── 9. ABOUT ──────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    | Feature | What it does |
    |---|---|
    | `pl.LazyFrame` (direct) | Table + Query Plan tabs, zero materialisation cost |
    | `demand.collect()` | Full DataFrame UI with column stats and sorting |
    | `mo.ui.table` | Select rows reactively; use `.value` downstream |
    | `mo.ui.dataframe` | Visual query builder with generated Polars code |

    ## References

    - [Marimo: Dataframe Transformation Guide](https://docs.marimo.io/guides/working_with_data/dataframes/)
    - [Polars: Lazy API Overview](https://docs.pola.rs/user-guide/lazy/)
    - [Polars: Query Plan Explained](https://docs.pola.rs/user-guide/lazy/query-plan/)
    """)
    return


if __name__ == "__main__":
    app.run()
