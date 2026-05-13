import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="Data Science Analysis")


# ── 1. SETUP ───────────────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Data Science Analysis
        ---
        **Template** · marimo + polars + scikit-learn + chainladder
        """
    )
    return


@app.cell
def _(mo):
    import sys
    import importlib
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt

    # Add project root to path so src.* imports work
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.utils.config import (
        check_versions,
        RANDOM_SEED,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        OUTPUTS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        DIAGNOSTICS_DIR,
        DEBUG,
    )
    return (
        DEBUG,
        DATA_PROCESSED_DIR,
        DATA_RAW_DIR,
        DIAGNOSTICS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        RANDOM_SEED,
        Path,
        check_versions,
        importlib,
        np,
        pl,
        plt,
        sys,
    )


@app.cell
def _(mo):
    debug_toggle = mo.ui.switch(label="Debug mode", value=False)
    debug_toggle
    return (debug_toggle,)


@app.cell
def _(check_versions, debug_toggle, mo):
    versions = check_versions()
    _rows = [{"Package": k, "Version": v} for k, v in versions.items()]
    mo.vstack([
        mo.md("### Environment"),
        mo.callout(mo.md(f"Debug: `{'ON' if debug_toggle.value else 'OFF'}`"), kind="info"),
        mo.table(_rows),
    ])
    return (versions,)


# ── 2. EDA ─────────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Exploratory Data Analysis")
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(filetypes=[".csv", ".parquet"], label="Upload dataset")
    file_upload
    return (file_upload,)


@app.cell
def _(DATA_RAW_DIR, file_upload, mo, pl):
    from src.data.loaders import load_csv, load_parquet, inspect_schema

    _df = None
    _schema_info = None

    if file_upload.value:
        _file = file_upload.value[0]
        _dest = DATA_RAW_DIR / _file.name
        _dest.write_bytes(_file.contents)

        if _file.name.endswith(".csv"):
            _lf = load_csv(_dest)
        else:
            _lf = load_parquet(_dest)

        _df = _lf.collect()
        _schema_info = inspect_schema(_lf)

    dataset = _df
    return dataset, inspect_schema, load_csv, load_parquet


@app.cell
def _(dataset, mo, pl):
    if dataset is not None:
        _summary = dataset.describe()
        mo.vstack([
            mo.md(f"**Shape:** {dataset.shape[0]:,} rows × {dataset.shape[1]} columns"),
            mo.table(dataset.head(10).to_pandas()),
        ])
    else:
        mo.callout(mo.md("Upload a CSV or Parquet file above to begin."), kind="warn")
    return


@app.cell
def _(dataset, mo, pl):
    if dataset is not None:
        _numeric = [c for c, t in dataset.schema.items()
                    if t in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)]
        col_select = mo.ui.multiselect(
            options=_numeric,
            value=_numeric[:3] if len(_numeric) >= 3 else _numeric,
            label="Select columns to plot",
        )
    else:
        col_select = mo.ui.multiselect(options=[], label="Select columns to plot")
    col_select
    return (col_select,)


@app.cell
def _(col_select, dataset, mo, plt):
    if dataset is not None and col_select.value:
        _fig, _axes = plt.subplots(1, len(col_select.value), figsize=(5 * len(col_select.value), 4))
        if len(col_select.value) == 1:
            _axes = [_axes]
        for _ax, _col in zip(_axes, col_select.value):
            _ax.hist(dataset[_col].drop_nulls().to_numpy(), bins=30, edgecolor="white", linewidth=0.5)
            _ax.set_title(_col)
            _ax.set_xlabel("Value")
            _ax.set_ylabel("Count")
        _fig.tight_layout()
        mo.pyplot(_fig)
    return


# ── 3. TRIANGLE ANALYSIS ───────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Triangle Analysis (Chainladder)")
    return


@app.cell
def _(mo):
    _triangle_options = ["raa", "ukmotor", "clrd", "mack"]
    triangle_select = mo.ui.dropdown(
        options=_triangle_options,
        value="raa",
        label="Sample triangle",
    )
    triangle_select
    return (triangle_select,)


@app.cell
def _(mo, triangle_select):
    import chainladder as cl
    from src.models.reserving import triangle_diagnostics

    try:
        _tri = getattr(cl, triangle_select.value.upper(), None)
        if _tri is None:
            _tri = cl.load_sample(triangle_select.value)
        else:
            _tri = _tri()
        sample_triangle = _tri
        mo.callout(
            mo.md(f"Triangle `{triangle_select.value}` loaded — shape: `{sample_triangle.shape}`"),
            kind="success",
        )
    except Exception as _e:
        sample_triangle = None
        mo.callout(mo.md(f"Could not load triangle: {_e}"), kind="danger")
    return cl, sample_triangle, triangle_diagnostics


@app.cell
def _(mo, sample_triangle, triangle_diagnostics):
    if sample_triangle is not None:
        triangle_diagnostics(sample_triangle)
        mo.md("Development factor plots saved to `outputs/diagnostics/`.")
    return


# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Feature Engineering")
    return


@app.cell
def _(dataset, mo, pl):
    from src.features.transformers import cast_columns, drop_nulls_threshold, log_transform

    if dataset is not None:
        _numeric = [c for c, t in dataset.schema.items()
                    if t in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)]
        log_col_select = mo.ui.multiselect(
            options=_numeric,
            value=[],
            label="Apply log1p transform to",
        )
        null_thresh = mo.ui.slider(
            start=0.1, stop=1.0, step=0.05, value=0.5,
            label="Drop columns with null fraction >",
        )
    else:
        log_col_select = mo.ui.multiselect(options=[], label="Apply log1p transform to")
        null_thresh = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.5,
                                   label="Drop columns with null fraction >")

    mo.vstack([null_thresh, log_col_select])
    return cast_columns, drop_nulls_threshold, log_col_select, log_transform, null_thresh


@app.cell
def _(dataset, drop_nulls_threshold, log_col_select, log_transform, mo, null_thresh, pl):
    if dataset is not None:
        _lf = dataset.lazy()
        _lf = drop_nulls_threshold(_lf, threshold=null_thresh.value)
        if log_col_select.value:
            _lf = log_transform(_lf, log_col_select.value)
        transformed_df = _lf.collect()
        mo.vstack([
            mo.md(f"**Transformed shape:** {transformed_df.shape[0]:,} × {transformed_df.shape[1]}"),
            mo.table(transformed_df.head(5).to_pandas()),
        ])
    else:
        transformed_df = None
        mo.callout(mo.md("Load a dataset in the EDA section first."), kind="warn")
    return (transformed_df,)


# ── 5. MODELLING ──────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Modelling")
    return


@app.cell
def _(mo):
    model_choice = mo.ui.dropdown(
        options=["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"],
        value="LinearRegression",
        label="Model",
    )
    target_input = mo.ui.text(value="", placeholder="target column name", label="Target column")
    cv_folds = mo.ui.slider(start=2, stop=10, step=1, value=5, label="CV folds")
    mo.vstack([model_choice, target_input, cv_folds])
    return cv_folds, model_choice, target_input


@app.cell
def _(
    RANDOM_SEED,
    cv_folds,
    mo,
    model_choice,
    target_input,
    transformed_df,
):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import polars as _pl

    from src.models.pipelines import build_pipeline, cross_validate_pipeline, fit_and_evaluate

    _model_map = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_SEED),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }

    pipeline_result = None
    cv_results = None
    eval_metrics = None

    if transformed_df is not None and target_input.value in transformed_df.columns:
        _target = target_input.value
        _feature_cols = [c for c in transformed_df.columns
                         if c != _target
                         and transformed_df[c].dtype in (
                             _pl.Float32, _pl.Float64, _pl.Int32, _pl.Int64,
                             _pl.UInt32, _pl.UInt64
                         )]
        _X = transformed_df.select(_feature_cols).to_numpy()
        _y = transformed_df[_target].to_numpy()

        _X_train, _X_test, _y_train, _y_test = train_test_split(
            _X, _y, test_size=0.2, random_state=RANDOM_SEED
        )

        _pipeline = build_pipeline(StandardScaler(), _model_map[model_choice.value])
        cv_results = cross_validate_pipeline(_pipeline, _X_train, _y_train, cv=cv_folds.value)
        eval_metrics = fit_and_evaluate(_pipeline, _X_train, _y_train, _X_test, _y_test)
        pipeline_result = _pipeline

        mo.vstack([
            mo.md(f"**Model:** `{model_choice.value}`"),
            mo.table([{"Metric": k, "Value": f"{v:.4f}"} for k, v in eval_metrics.items()]),
        ])
    else:
        mo.callout(
            mo.md("Set a valid **target column** name above (must exist in the transformed dataset)."),
            kind="warn",
        )
    return (
        GradientBoostingRegressor,
        LinearRegression,
        RandomForestRegressor,
        StandardScaler,
        build_pipeline,
        cross_validate_pipeline,
        cv_results,
        eval_metrics,
        fit_and_evaluate,
        pipeline_result,
        train_test_split,
    )


# ── 6. RESERVING ──────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Reserving (Chainladder Methods)")
    return


@app.cell
def _(mo):
    reserving_method = mo.ui.dropdown(
        options=["Chainladder", "Mack", "Bootstrap"],
        value="Chainladder",
        label="Reserving method",
    )
    n_sims_slider = mo.ui.slider(
        start=100, stop=5000, step=100, value=1000,
        label="Bootstrap simulations (Bootstrap only)",
    )
    mo.vstack([reserving_method, n_sims_slider])
    return n_sims_slider, reserving_method


@app.cell
def _(mo, n_sims_slider, reserving_method, sample_triangle):
    from src.models.reserving import (
        fit_chainladder,
        fit_mack,
        fit_bootstrap,
        summarise_reserves,
    )

    reserve_summary = None
    fitted_reserving_model = None

    if sample_triangle is not None:
        try:
            if reserving_method.value == "Chainladder":
                fitted_reserving_model = fit_chainladder(sample_triangle)
            elif reserving_method.value == "Mack":
                fitted_reserving_model = fit_mack(sample_triangle)
            else:
                fitted_reserving_model = fit_bootstrap(
                    sample_triangle, n_sims=n_sims_slider.value
                )
            reserve_summary = summarise_reserves(fitted_reserving_model)
            mo.vstack([
                mo.md(f"**Method:** `{reserving_method.value}`"),
                mo.table(reserve_summary.to_pandas()),
            ])
        except Exception as _e:
            mo.callout(mo.md(f"Reserving error: {_e}"), kind="danger")
    else:
        mo.callout(mo.md("Load a triangle in the Triangle Analysis section first."), kind="warn")
    return (
        fit_bootstrap,
        fit_chainladder,
        fit_mack,
        fitted_reserving_model,
        reserve_summary,
        summarise_reserves,
    )


# ── 7. EXPORT ─────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Export Results")
    return


@app.cell
def _(
    FIGURES_DIR,
    MODELS_DIR,
    col_select,
    dataset,
    mo,
    pipeline_result,
    plt,
    reserve_summary,
):
    from src.models.pipelines import save_pipeline

    def _do_export():
        saved = []

        if pipeline_result is not None:
            _path = save_pipeline(pipeline_result, MODELS_DIR / "pipeline.joblib")
            saved.append(f"Pipeline → `{_path}`")

        if reserve_summary is not None:
            _path = FIGURES_DIR.parent / "diagnostics" / "reserve_summary.csv"
            reserve_summary.write_csv(_path)
            saved.append(f"Reserve summary → `{_path}`")

        if dataset is not None and col_select.value:
            _fig, _ax = plt.subplots(figsize=(8, 4))
            for _col in col_select.value:
                _ax.hist(dataset[_col].drop_nulls().to_numpy(), bins=30, alpha=0.6, label=_col)
            _ax.legend()
            _ax.set_title("EDA Distributions")
            _out = FIGURES_DIR / "eda_distributions.png"
            _fig.savefig(_out, dpi=150, bbox_inches="tight")
            plt.close(_fig)
            saved.append(f"EDA figure → `{_out}`")

        return saved

    export_button = mo.ui.button(label="Export to outputs/", on_click=lambda _: _do_export())
    export_button
    return export_button, save_pipeline


@app.cell
def _(export_button, mo):
    if export_button.value:
        _lines = "\n".join(f"- {s}" for s in export_button.value)
        mo.callout(mo.md(f"Exported:\n{_lines}"), kind="success")
    return


if __name__ == "__main__":
    app.run()
