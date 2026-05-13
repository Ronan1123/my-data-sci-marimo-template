# Data Science Template

A hybrid marimo + polars + scikit-learn + chainladder project template for interactive data analysis and actuarial reserving.

---

## Quick Start (Windows)

> **Requires Python 3.11 or later.** Download from [python.org/downloads](https://www.python.org/downloads/) if needed.  
> Tick **"Add Python to PATH"** during installation. No administrator rights required if you use the Windows Store version of Python or choose "Install just for me" in the installer.

Open a terminal (PowerShell or Command Prompt) in the project folder, then run each step:

**1. Create a virtual environment**
```
python -m venv .venv
```

**2. Activate it**
```
.venv\Scripts\activate
```
You should see `(.venv)` appear at the start of your prompt.

**3. Install all packages**
```
pip install -e ".[dev]"
```
This may take a few minutes the first time.

**4. Launch the notebook**
```
marimo edit marimo\analysis.py
```
The notebook will open in your browser.

---

## Architecture

```
my-data-sci-marimo-template/
├── data/
│   ├── raw/            ← drop source files here (git-ignored)
│   └── processed/      ← cleaned outputs from loaders.py (git-ignored)
├── marimo/
│   └── analysis.py     ← interactive notebook (narrative flow)
├── src/
│   ├── data/loaders.py         ← CSV/Parquet/Triangle loading
│   ├── features/transformers.py ← polars transforms + sklearn adaptor
│   ├── models/
│   │   ├── pipelines.py        ← sklearn Pipelines, eval, persistence
│   │   └── reserving.py        ← CL / Mack / Bootstrap workflows
│   └── utils/config.py         ← paths, seeds, version checks
├── outputs/            ← generated artefacts (git-ignored)
│   ├── models/
│   ├── diagnostics/
│   └── figures/
└── tests/
```

**Rule:** marimo cells call `src.*` functions. Logic never lives inside cells.

---

## Running the Notebook

```bash
marimo edit marimo/analysis.py
```

The notebook opens in your browser. Work through the sections top-to-bottom:

| Section | What it does |
|---|---|
| Setup | Version checks, debug toggle |
| EDA | Upload CSV/Parquet, schema inspection, distribution plots |
| Triangle Analysis | Load chainladder sample, plot development factors |
| Feature Engineering | Column selection, null dropping, log transforms |
| Modelling | Choose model, fit pipeline, view metrics |
| Reserving | Choose CL/Mack/Bootstrap, view IBNR table |
| Export | Save pipeline, figures, and reserve summary to `outputs/` |

---

## Running Tests

```bash
pytest
```

All tests use synthetic data or chainladder built-in samples — no external data files required.

---

## Data Workflow

1. Drop raw source files into `data/raw/` (they are git-ignored).
2. Use `load_csv()` / `load_parquet()` in `src/data/loaders.py` — these return polars `LazyFrame`s.
3. After cleaning, call `save_processed(df, "filename.parquet")` to write to `data/processed/`.
4. Load processed data via `load_parquet(DATA_PROCESSED_DIR / "filename.parquet")`.

---

## Git Workflow

Use [conventional commits](https://www.conventionalcommits.org/):

```
feat: add bootstrap reserve confidence intervals
fix: correct null-drop threshold logic
refactor: simplify PolarsTransformer.transform
docs: update README with data workflow
```

**Never commit:**
- `data/raw/` or `data/processed/`
- `outputs/`
- `.env` files
- `*.pkl`, `*.joblib`, `*.parquet`

---

## Adding New Models

1. Add an estimator to the `model_map` in `marimo/analysis.py`.
2. No changes to `src/` required unless the model needs a custom preprocessor.

## Adding New Reserving Methods

Implement a `fit_*` function in `src/models/reserving.py` that returns a fitted chainladder model with an `ibnr_` attribute, then add it to the `reserving_method` dropdown in the notebook.

## Adding New Datasets

Drop a CSV or Parquet into `data/raw/` and use the file upload widget in the EDA section, or point `load_csv()` / `load_parquet()` directly at the file path.
