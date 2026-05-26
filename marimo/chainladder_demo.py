import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium", app_title="Chainladder Development Tutorial")


@app.cell
def _():
    import marimo as mo
    import chainladder as cl

    return cl, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chainladder: Selecting Development Factors

    This notebook walks through the key ideas in the chainladder development tutorial —
    from fitting development patterns through to applying them across multidimensional triangles.

    All datasets are built-in samples from the `chainladder` package. No external data required.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample Data

    We use two built-in datasets throughout this notebook:

    | Dataset | Description |
    |---|---|
    | `genins` | General insurance — 10×10 cumulative paid loss triangle |
    | `clrd` | CAS Loss Reserve Database — multiple lines of business |
    """)
    return


@app.cell
def _(cl):
    genins = cl.load_sample("genins")
    genins.to_frame(origin_as_datetime=False)
    return (genins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Estimator Basics

    `Development` follows the scikit-learn API:

    1. **Instantiate** — set assumptions (or use defaults)
    2. **Fit** — pass a triangle via `.fit(triangle)`
    3. **Read fitted attributes** — trailing `_` denotes estimated values

    | Attribute | Meaning |
    |---|---|
    | `ldf_` | Age-to-age link development factors |
    | `cdf_` | Cumulative development factors (tail to ultimate) |
    """)
    return


@app.cell
def _(cl, genins, mo):
    dev_basic = cl.Development().fit(genins)

    mo.vstack([
        mo.md("**Link Development Factors (ldf_):**"),
        dev_basic.ldf_.to_frame(origin_as_datetime=False),
        mo.md("**Cumulative Development Factors (cdf_):**"),
        dev_basic.cdf_.to_frame(origin_as_datetime=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Development Averaging Methods

    Three weighting schemes produce different link development factors:

    | Method | Description |
    |---|---|
    | `volume` | Weighted by data volume (default) |
    | `simple` | Unweighted average of ratios |
    | `regression` | OLS regression through the origin (Y = mX) |

    Select a method below to see how it changes the fitted factors.
    """)
    return


@app.cell
def _(mo):
    average_dropdown = mo.ui.dropdown(
        options=["volume", "simple", "regression"],
        value="volume",
        label="Averaging method",
    )
    average_dropdown
    return (average_dropdown,)


@app.cell
def _(average_dropdown, cl, genins, mo):
    dev_avg = cl.Development(average=average_dropdown.value).fit(genins)

    mo.vstack([
        mo.md(f"**LDF using `average='{average_dropdown.value}'`:**"),
        dev_avg.ldf_.to_frame(origin_as_datetime=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Averaging Period Selection (`n_periods`)

    By default, all available origin periods are used. The `n_periods` parameter
    restricts fitting to the latest **N** origin periods, which can reduce the
    influence of older, less representative data.

    - `-1` uses all periods
    - `N > 0` uses only the latest N periods
    """)
    return


@app.cell
def _(genins, mo):
    age_to_age_styled = (
        genins.age_to_age
        .to_frame(origin_as_datetime=False)
        .style.background_gradient(cmap="viridis", axis=0)
        .format("{:.4f}", na_rep="")
    )
    mo.vstack([
        mo.md("**Age-to-age link ratios (`genins.age_to_age`) — Yellow = higher, Blue = lower:**"),
        mo.Html(age_to_age_styled.to_html()),
    ])
    return


@app.cell
def _(mo):
    n_periods_slider = mo.ui.slider(
        start=-1,
        stop=9,
        step=1,
        value=-1,
        label="n_periods (-1 = all)",
    )
    n_periods_slider
    return (n_periods_slider,)


@app.cell
def _(cl, genins, mo, n_periods_slider):
    dev_n = cl.Development(n_periods=n_periods_slider.value).fit(genins)

    label = "all periods" if n_periods_slider.value == -1 else f"latest {n_periods_slider.value} periods"
    mo.vstack([
        mo.md(f"**LDF fitted on {label}:**"),
        dev_n.ldf_.to_frame(origin_as_datetime=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Discarding Problematic Link Ratios

    Outlier link ratios can distort development factors. Chainladder provides
    several ways to exclude them:

    - `drop_high` / `drop_low` — exclude the highest or lowest ratios (olympic averaging)
    - `drop` — exclude a specific cell by `(origin, age)` coordinate

    Toggle the options below.
    """)
    return


@app.cell
def _(mo):
    drop_high_cb = mo.ui.checkbox(label="Drop highest ratio per column", value=False)
    drop_low_cb = mo.ui.checkbox(label="Drop lowest ratio per column", value=False)
    mo.hstack([drop_high_cb, drop_low_cb])
    return drop_high_cb, drop_low_cb


@app.cell
def _(cl, drop_high_cb, drop_low_cb, genins, mo):
    dev_olympic = cl.Development(
        drop_high=drop_high_cb.value,
        drop_low=drop_low_cb.value,
    ).fit(genins)

    flags = []
    if drop_high_cb.value:
        flags.append("drop_high=True")
    if drop_low_cb.value:
        flags.append("drop_low=True")
    label_olympic = f"`Development({', '.join(flags)})`" if flags else "`Development()` (no drops)"

    mo.vstack([
        mo.md(f"**LDF with {label_olympic}:**"),
        dev_olympic.ldf_.to_frame(origin_as_datetime=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Transformers vs. Predictors

    `Development` is a **transformer** — it modifies the triangle to produce
    development patterns that downstream IBNR models consume.

    - `.fit(triangle)` — estimates parameters from the triangle
    - `.transform(triangle)` — applies the fitted parameters to (possibly different) data
    - `.fit_transform(triangle)` — both steps at once

    A key use case: fit development patterns on an **industry aggregate**, then apply
    them to an **individual company** triangle.
    """)
    return


@app.cell
def _(cl, mo):
    clrd = cl.load_sample("clrd")
    comauto = clrd[clrd["LOB"] == "comauto"]["CumPaidLoss"]
    comauto_industry = comauto.sum()

    industry_dev = cl.Development().fit(comauto_industry)
    transformed = industry_dev.transform(comauto)

    mo.vstack([
        mo.md("**Industry-fitted CDF applied to individual comauto companies:**"),
        transformed.cdf_.to_frame(origin_as_datetime=False),
    ])
    return (clrd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Multidimensional Triangles

    Chainladder can fit development patterns across **multiple triangles simultaneously**
    using a single `Development` call — useful for fitting the same assumptions across
    lines of business.

    Select a line of business below to inspect its cumulative development factors.
    """)
    return


@app.cell
def _(cl, clrd, mo):
    clrd_by_lob = clrd.groupby("LOB").sum()["CumPaidLoss"]
    dev_multi = cl.Development().fit_transform(clrd_by_lob)

    lob_options = sorted(clrd_by_lob.index["LOB"].unique().tolist())
    lob_dropdown = mo.ui.dropdown(
        options=lob_options,
        value=lob_options[0],
        label="Line of business",
    )
    lob_dropdown
    return dev_multi, lob_dropdown


@app.cell
def _(dev_multi, lob_dropdown, mo):
    mo.vstack([
        mo.md(f"**CDF for `{lob_dropdown.value}`:**"),
        dev_multi.cdf_.loc[lob_dropdown.value].to_frame(origin_as_datetime=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    | Concept | Key parameter / method |
    |---|---|
    | Fitted factors | `dev.ldf_`, `dev.cdf_` |
    | Weighting scheme | `Development(average=...)` |
    | Period selection | `Development(n_periods=...)` |
    | Olympic averaging | `Development(drop_high=True, drop_low=True)` |
    | Industry patterns | `.fit(aggregate).transform(individual)` |
    | Multi-triangle | `.fit_transform(clrd.groupby("LOB").sum())` |

    ## References

    - [Chainladder development tutorial](https://chainladder-python.readthedocs.io/stable/getting_started/tutorials/development-tutorial.html)
    - [Chainladder API reference](https://chainladder-python.readthedocs.io/stable/api.html)
    """)
    return


if __name__ == "__main__":
    app.run()
