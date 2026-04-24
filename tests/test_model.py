import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import (
    ModelResult,
    fit_model,
    fit_iv_model,
    predict_ridership,
    elasticity_impact,
    build_equation_str,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 60
    bus   = np.random.randint(1, 10, n).astype(float)
    win   = np.random.randint(0, 5,  n).astype(float)
    dist  = np.random.uniform(10, 200, n)          # instrument candidate
    entry = np.exp(
        1.5 + 0.3 * np.log(bus + 1) + 0.2 * np.log(win + 1)
        + np.random.normal(0, 0.15, n)
    )
    return pd.DataFrame({"bus_stop_count": bus, "win_count": win,
                         "bus_stop_min_dist": dist, "entry": entry})


# ── OLS ────────────────────────────────────────────────────────────────────────

def test_fit_model_coef_df_columns(sample_df):
    _, coef_df = fit_model(sample_df, ["bus_stop_count", "win_count"])
    assert list(coef_df.columns) == [
        "variable", "coef", "se", "t", "p", "ci_lo", "ci_hi", "significant"
    ]


def test_fit_model_coef_df_rows(sample_df):
    _, coef_df = fit_model(sample_df, ["bus_stop_count", "win_count"])
    assert len(coef_df) == 3  # const + 2 features


def test_fit_model_variable_names(sample_df):
    _, coef_df = fit_model(sample_df, ["bus_stop_count", "win_count"])
    assert "const" in coef_df["variable"].values
    assert "log_bus_stop_count" in coef_df["variable"].values
    assert "log_win_count" in coef_df["variable"].values


def test_fit_model_returns_model_result(sample_df):
    result, _ = fit_model(sample_df, ["bus_stop_count", "win_count"])
    assert isinstance(result, ModelResult)
    assert result.model_type == "OLS"
    assert result.rsquared > 0
    assert result.nobs == 60


# ── IV/2SLS ────────────────────────────────────────────────────────────────────

def test_fit_iv_model_returns_model_result(sample_df):
    result, coef_df = fit_iv_model(
        df=sample_df,
        exog_features=["win_count"],
        endog_features=["bus_stop_count"],
        instruments={"bus_stop_count": ["bus_stop_min_dist"]},
    )
    assert isinstance(result, ModelResult)
    assert result.model_type == "IV/2SLS"
    assert result.nobs == 60


def test_fit_iv_model_coef_df_columns(sample_df):
    _, coef_df = fit_iv_model(
        df=sample_df,
        exog_features=["win_count"],
        endog_features=["bus_stop_count"],
        instruments={"bus_stop_count": ["bus_stop_min_dist"]},
    )
    assert list(coef_df.columns) == [
        "variable", "coef", "se", "t", "p", "ci_lo", "ci_hi", "significant"
    ]


def test_fit_iv_model_same_variable_names_as_ols(sample_df):
    _, ols_coef = fit_model(sample_df, ["bus_stop_count", "win_count"])
    _, iv_coef  = fit_iv_model(
        df=sample_df,
        exog_features=["win_count"],
        endog_features=["bus_stop_count"],
        instruments={"bus_stop_count": ["bus_stop_min_dist"]},
    )
    assert set(ols_coef["variable"]) == set(iv_coef["variable"])


# ── Shared helpers ─────────────────────────────────────────────────────────────

def test_predict_ridership_positive(sample_df):
    _, coef_df = fit_model(sample_df, ["bus_stop_count", "win_count"])
    pred = predict_ridership(coef_df, {"bus_stop_count": 5.0, "win_count": 2.0})
    assert isinstance(pred, float)
    assert pred > 0


def test_predict_ridership_higher_with_more_bus(sample_df):
    _, coef_df = fit_model(sample_df, ["bus_stop_count", "win_count"])
    low  = predict_ridership(coef_df, {"bus_stop_count": 1.0, "win_count": 1.0})
    high = predict_ridership(coef_df, {"bus_stop_count": 10.0, "win_count": 1.0})
    assert high > low


def test_elasticity_impact(sample_df):
    assert elasticity_impact(0.3, 10.0)  == pytest.approx(3.0)
    assert elasticity_impact(-0.1, 20.0) == pytest.approx(-2.0)
    assert elasticity_impact(0.0, 50.0)  == pytest.approx(0.0)


def test_build_equation_str(sample_df):
    _, coef_df = fit_model(sample_df, ["bus_stop_count", "win_count"])
    eq = build_equation_str(coef_df, log_offset=1.0)
    assert eq.startswith("log(ridership)")
    assert "log(bus_stop_count + 1.0)" in eq
    assert "log(win_count + 1.0)" in eq
