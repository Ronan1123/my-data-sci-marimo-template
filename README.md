# Data Science Template

A hybrid marimo + polars + scikit-learn + chainladder project template for interactive data analysis and actuarial reserving.

---

## Quick Start (Windows)

1. **Clone or download** the repository to your computer.
2. **Double-click `setup.bat`** in the project folder.
   - It will check your Python version, create a virtual environment, and install all packages.
   - Keep the window open until you see the "Setup complete!" message.
3. Open a terminal in the project folder (`Win + R` → `cmd`, then `cd` to the folder), then run:
   ```
   .venv\Scripts\activate
   marimo edit marimo\analysis.py
   ```
   The notebook will open in your browser.

> **Requires Python 3.11 or later.** Download from [python.org/downloads](https://www.python.org/downloads/) if needed.  
> Tick **"Add Python to PATH"** during the installer.

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

## Manual Setup (alternative)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install the project with dev dependencies
pip install -e ".[dev]"

# 3. Verify versions
python -c "from src.utils.config import check_versions; check_versions()"
```

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
