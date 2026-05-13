"""Tests for src/models/pipelines.py"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.models.pipelines import (
    build_pipeline,
    cross_validate_pipeline,
    fit_and_evaluate,
    load_pipeline,
    save_pipeline,
)


@pytest.fixture
def diabetes_data():
    X, y = load_diabetes(return_X_y=True)
    split = int(0.8 * len(y))
    return X[:split], X[split:], y[:split], y[split:]


def test_build_pipeline_returns_pipeline(diabetes_data):
    from sklearn.pipeline import Pipeline

    pipeline = build_pipeline(StandardScaler(), LinearRegression())
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2


def test_fit_and_evaluate_returns_metrics(diabetes_data):
    X_train, X_test, y_train, y_test = diabetes_data
    pipeline = build_pipeline(StandardScaler(), LinearRegression())
    metrics = fit_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert isinstance(metrics["r2"], float)


def test_fit_and_evaluate_r2_reasonable(diabetes_data):
    X_train, X_test, y_train, y_test = diabetes_data
    pipeline = build_pipeline(StandardScaler(), LinearRegression())
    metrics = fit_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
    assert metrics["r2"] > 0.3


def test_cross_validate_pipeline(diabetes_data):
    X_train, _, y_train, _ = diabetes_data
    pipeline = build_pipeline(StandardScaler(), LinearRegression())
    results = cross_validate_pipeline(pipeline, X_train, y_train, cv=3, scoring="r2")
    assert "test_r2" in results
    assert len(results["test_r2"]) == 3


def test_save_and_load_pipeline(tmp_path, diabetes_data):
    X_train, X_test, y_train, y_test = diabetes_data
    pipeline = build_pipeline(StandardScaler(), LinearRegression())
    fit_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

    path = tmp_path / "pipeline.joblib"
    save_pipeline(pipeline, path)
    loaded = load_pipeline(path)

    original_pred = pipeline.predict(X_test)
    loaded_pred = loaded.predict(X_test)
    np.testing.assert_array_almost_equal(original_pred, loaded_pred)
