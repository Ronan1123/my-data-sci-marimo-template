You are designing a hybrid, design‑led data‑science project template for VS Code.  
The project uses marimo for interactive exploration and scikit‑learn + polars + chainladder inside modular Python packages.  
The template must embody the following principles:

1. Clarity  
   - Clean, modern, minimal aesthetic.  
   - Strong visual hierarchy in marimo cells.  
   - Consistent naming, spacing, and section markers.  
   - Clear separation between exploratory and production code.

2. Reproducibility  
   - Deterministic execution with explicit version checks for marimo, polars, numpy, pandas, scikit‑learn, and chainladder.  
   - A single configuration module for paths, constants, and random seeds.  
   - All modelling code lives in importable Python modules, not marimo cells.

3. Hybrid Architecture  
   - marimo is used for:  
     • Data exploration  
     • EDA visualisations  
     • Feature experimentation  
     • Model iteration and comparison  
     • Chainladder triangle exploration and diagnostics  
   - Python modules are used for:  
     • Data loading and cleaning  
     • Feature engineering functions  
     • scikit‑learn pipelines  
     • Chainladder reserving workflows  
     • Model training, evaluation, and persistence  
   - marimo cells call into these modules, never duplicating logic.

4. Modularity  
   - Project structured as a Python package:  
       project_root/  
         ├── marimo/  
         │     └── analysis.py  
         ├── src/  
         │     ├── data/  
         │     │     └── loaders.py  
         │     ├── features/  
         │     │     └── transformers.py  
         │     ├── models/  
         │     │     ├── pipelines.py  
         │     │     └── reserving.py   (chainladder workflows)  
         │     └── utils/  
         │           └── config.py  
         ├── outputs/  
         │     ├── models/  
         │     ├── diagnostics/  
         │     └── figures/  
         ├── tests/  
         ├── pyproject.toml  
         ├── .gitignore  
         └── README.md  
   - The `outputs/` directory must exist but be fully git‑ignored.  
   - Each module contains small, composable functions.  
   - scikit‑learn Pipelines encapsulate preprocessing + modelling.

5. Flow‑Based “Vibe Coding”  
   - marimo notebook structured as a narrative:  
     Setup → EDA → Triangle Analysis → Feature Engineering → Modelling → Evaluation → Export.  
   - Each section feels like a “scene” with a clear purpose.  
   - Use marimo UI widgets for interactive column selection, model choice, parameter tweaking, and triangle selection.

6. Observability  
   - Lightweight logging in Python modules.  
   - marimo diagnostics cell showing environment, versions, dataset metadata, and triangle summaries.  
   - Optional debug mode toggle.

7. Extensibility  
   - Easy to add new models, datasets, or feature pipelines.  
   - Clear placeholders for hyperparameter tuning, model comparison, and experiment tracking.  
   - Reserving module should support:  
     • Chainladder  
     • Mack  
     • Bootstrap  
     • Custom triangle transformations  

8. Git‑Friendly Design  
   - No large data files committed.  
   - `outputs/` directory must be created but fully git‑ignored.  
   - Deterministic outputs so diffs remain clean.  
   - marimo notebooks stored as `.py` files for readable diffs.  
   - README includes instructions for environment setup and project workflow.  
   - Use conventional commits style for commit messages.

---

### Required Technologies
- marimo (interactive notebook framework)  
- polars (data wrangling)  
- scikit‑learn (modelling)  
- chainladder (actuarial triangle modelling)  
- numpy, matplotlib/plotly (visualisation)  
- python‑dotenv (optional config)  
- git for version control  

---

### Template Structure to Generate

1. **Project Scaffolding**  
   - Create the directory structure listed above.  
   - Include a `.gitignore` that excludes:  
       outputs/  
       data/raw/  
       data/interim/  
       *.pkl  
       *.joblib  
       *.parquet  
       .env  
   - Create a `pyproject.toml` with pinned versions.

2. **Configuration Module (`src/utils/config.py`)**  
   - Random seed  
   - Paths  
   - Display options  
   - Version checks for marimo, polars, sklearn, chainladder  

3. **Data Module (`src/data/loaders.py`)**  
   - Functions for loading datasets using polars LazyFrame  
   - Schema inspection utilities  
   - Triangle loading utilities for chainladder  

4. **Feature Engineering Module (`src/features/transformers.py`)**  
   - Clean, typed polars expressions  
   - scikit‑learn compatible transformers if needed  

5. **Model Module (`src/models/pipelines.py`)**  
   - scikit‑learn Pipelines  
   - Fit/predict/evaluate functions  
   - Cross‑validation helpers  

6. **Reserving Module (`src/models/reserving.py`)**  
   - Chainladder workflows  
   - Mack model  
   - Bootstrap model  
   - Triangle diagnostics  
   - Reserve summary outputs  

7. **marimo Notebook (`marimo/analysis.py`)**  
   - Header + metadata  
   - Imports + config  
   - Data loading + EDA  
   - Triangle exploration (chainladder)  
   - Interactive feature engineering  
   - Model selection + evaluation  
   - Export results to `outputs/`  

8. **Tests (`tests/`)**  
   - Basic tests for loaders, transformers, pipelines, and reserving workflows  

9. **README.md**  
   - Project overview  
   - How to run marimo  
   - How to run tests  
   - Git workflow guidelines  
   - Notes on ignoring large outputs  

---

### Output Requirements
- Produce a complete hybrid project scaffold.  
- marimo notebook must call into the Python modules, not duplicate logic.  
- Code must be clean, modern, and aesthetically consistent.  
- All components must run end‑to‑end with no missing imports.  
- Chainladder workflows must be integrated cleanly and modularly.  
