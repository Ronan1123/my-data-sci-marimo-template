import marimo

__generated_with = "0.23.7"
app = marimo.App(width="full", app_title="Altair Reactive Charts")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    from vega_datasets import data

    return alt, data, mo


@app.cell
def _(mo):
    mo.md("""
    # Reactive Charts with Altair

    This notebook demonstrates **linked interactive charts** using Altair and marimo's
    `mo.ui.altair_chart`. Selections in one chart automatically filter others.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 1 — Brush to filter

    Click and drag on the scatter plot to select a region.
    The bar chart on the right updates instantly to show only the selected cars.
    """)
    return


@app.cell
def _(alt, data, mo):
    cars = data.cars()

    brush = alt.selection_interval()

    scatter = (
        alt.Chart(cars)
        .mark_point()
        .encode(
            x=alt.X("Horsepower:Q", title="Horsepower"),
            y=alt.Y("Miles_per_Gallon:Q", title="Miles per Gallon"),
            color=alt.condition(brush, "Origin:N", alt.value("lightgrey")),
            tooltip=["Name:N", "Origin:N", "Horsepower:Q", "Miles_per_Gallon:Q"],
        )
        .add_params(brush)
        .properties(title="Horsepower vs MPG", width=500, height=350)
    )

    bars = (
        alt.Chart(cars)
        .mark_bar()
        .encode(
            y=alt.Y("Origin:N", title="Origin"),
            color="Origin:N",
            x=alt.X("count(Origin):Q", title="Count"),
        )
        .transform_filter(brush)
        .properties(title="Cars by Origin (filtered)", width=300, height=350)
    )

    chart = mo.ui.altair_chart(scatter | bars)
    chart
    return (chart,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 2 — Select rows to drill down

    The table below shows the cars from your brush selection.
    Click rows in the table to select specific cars for the histograms below.
    """)
    return


@app.cell
def _(chart, mo):
    filtered_table = mo.ui.table(chart.value)
    filtered_table
    return (filtered_table,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 3 — Histograms update from table selection

    Select one or more rows in the table above.
    The MPG and Horsepower distributions for those cars appear below.
    """)
    return


@app.cell
def _(alt, filtered_table, mo):
    mo.stop(not len(filtered_table.value))

    mpg_hist = mo.ui.altair_chart(
        alt.Chart(filtered_table.value)
        .mark_bar()
        .encode(
            alt.X("Miles_per_Gallon:Q", bin=True, title="Miles per Gallon"),
            y=alt.Y("count()", title="Count"),
        )
        .properties(title="MPG Distribution", width=350, height=250)
    )

    hp_hist = mo.ui.altair_chart(
        alt.Chart(filtered_table.value)
        .mark_bar()
        .encode(
            alt.X("Horsepower:Q", bin=True, title="Horsepower"),
            y=alt.Y("count()", title="Count"),
        )
        .properties(title="Horsepower Distribution", width=350, height=250)
    )

    mo.hstack([mpg_hist, hp_hist], justify="space-around", widths="equal")
    return


if __name__ == "__main__":
    app.run()
