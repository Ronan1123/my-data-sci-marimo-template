"""scikit-learn pipeline builders, evaluation helpers, and persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_pipeline(preprocessor: BaseEstimator, model: BaseEstimator) -> Pipeline:
    """Combine a preprocessor and estimator into a named sklearn Pipeline."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    logger.info("Built pipeline: %s → %s", type(preprocessor).__name__, type(model).__name__)
    return pipeline


def cross_validate_pipeline(
    pipeline: Pipeline,
    X,
    y,
    cv: int = 5,
    scoring: str | list[str] = "r2",
) -> dict[str, np.ndarray]:
    """Run cross-validation and return a scores dict."""
    logger.info("Cross-validating with cv=%d, scoring=%s", cv, scoring)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
    for key, vals in results.items():
        if key.startswith("test_") or key.startswith("train_"):
            logger.info("  %s: %.4f ± %.4f", key, vals.mean(), vals.std())
    return results


def fit_and_evaluate(
    pipeline: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict[str, float]:
    """Fit pipeline and return regression metrics on the test set."""
    logger.info("Fitting pipeline on %d training samples", len(y_train))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    logger.info("Test metrics — MAE: %.4f | RMSE: %.4f | R²: %.4f", *metrics.values())
    return metrics


def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s", path)
    return path


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    path = Path(path)
    logger.info("Loading pipeline from %s", path)
    return joblib.load(path)
