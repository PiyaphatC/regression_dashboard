# Elasticity Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard that fits a live log-log OLS or IV/2SLS ridership elasticity model, displays results across three tabs, and lets users run what-if simulations per station.

**Architecture:** Two-file Python app — `model.py` holds pure model functions (OLS + IV, testable without Streamlit), `app.py` holds all Streamlit UI. Data loaded once from the existing CSV and cached. The model refits in-memory on any sidebar change. No results written to disk.

**Tech Stack:** Python 3.10+, Streamlit ≥ 1.32, statsmodels ≥ 0.14, linearmodels ≥ 6.0, pandas, numpy, plotly, pytest

---

## File Map

| File | Created/Modified | Responsibility |
|---|---|---|
| `requirements.txt` | Create | Python dependencies |
| `model.py` | Create | `ModelResult` namedtuple, `fit_model` (OLS), `fit_iv_model` (IV/2SLS), `build_equation_str`, `predict_ridership`, `elasticity_impact` |
| `app.py` | Create | Streamlit UI: sidebar with IV section, three tabs, wires to model.py |
| `tests/test_model.py` | Create | Unit tests for all model.py functions |

---

## Task 1: Project setup

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt in the project root**

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
statsmodels>=0.14.0
linearmodels>=6.0.0
plotly>=5.19.0
openpyxl>=3.1.0
pytest>=8.0.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without error.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add requirements for elasticity dashboard"
```

---

## Task 2: Core model functions (OLS + IV)

**Files:**
- Create: `model.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_model.py`

- [ ] **Step 1: Create tests/\_\_init\_\_.py**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 2: Write failing tests in tests/test_model.py**

```python
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
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 4: Create model.py**

```python
"""
model.py
--------
Pure model functions for the elasticity dashboard.
Supports OLS (statsmodels) and IV/2SLS (linearmodels).
No Streamlit dependency — fully unit-testable.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# Normalised result container — same shape for OLS and IV output.
# fvalue / f_pvalue come from the model's joint significance test.
ModelResult = namedtuple(
    "ModelResult",
    ["rsquared", "rsquared_adj", "nobs", "fvalue", "f_pvalue", "model_type"],
)


def fit_model(
    df: pd.DataFrame,
    features: list[str],
    log_offset: float = 1.0,
    sig_level: float = 0.05,
) -> tuple[ModelResult, pd.DataFrame]:
    """
    Fit a log-log OLS model with HC3 robust SEs.

    Returns
    -------
    ModelResult, coef_df
        coef_df columns: variable, coef, se, t, p, ci_lo, ci_hi, significant
        'variable' values are 'const' or 'log_{feature_name}'
    """
    X = df[features].apply(lambda s: np.log(s + log_offset))
    X.columns = [f"log_{c}" for c in features]
    X = sm.add_constant(X)
    y = np.log(df["entry"] + log_offset)
    y.name = "log_entry"

    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]

    res = sm.OLS(y_clean, X_clean).fit(cov_type="HC3")

    ci = res.conf_int()
    coef_df = pd.DataFrame({
        "variable": res.params.index,
        "coef":     res.params.values,
        "se":       res.bse.values,
        "t":        res.tvalues.values,
        "p":        res.pvalues.values,
        "ci_lo":    ci[0].values,
        "ci_hi":    ci[1].values,
    })
    coef_df["significant"] = coef_df["p"] < sig_level

    model_result = ModelResult(
        rsquared=res.rsquared,
        rsquared_adj=res.rsquared_adj,
        nobs=int(res.nobs),
        fvalue=res.fvalue,
        f_pvalue=res.f_pvalue,
        model_type="OLS",
    )
    return model_result, coef_df.reset_index(drop=True)


def fit_iv_model(
    df: pd.DataFrame,
    exog_features: list[str],
    endog_features: list[str],
    instruments: dict[str, list[str]],
    log_offset: float = 1.0,
    sig_level: float = 0.05,
) -> tuple[ModelResult, pd.DataFrame]:
    """
    Fit a log-log IV/2SLS model via linearmodels.

    Parameters
    ----------
    df : DataFrame with feature columns and 'entry'
    exog_features : feature columns treated as exogenous regressors
    endog_features : feature columns suspected to be endogenous
    instruments : mapping {endog_feature: [instrument_col, ...]}
                  instrument columns are log-transformed with the same offset
    log_offset : added before log(), default 1.0
    sig_level : threshold for the 'significant' flag

    Returns
    -------
    ModelResult, coef_df   (same schema as fit_model)
    """
    all_features = exog_features + endog_features

    # Build log-transformed matrices
    log_df = df[all_features + list({c for cols in instruments.values() for c in cols})].apply(
        lambda s: np.log(s + log_offset)
    )
    log_df.columns = [f"log_{c}" for c in log_df.columns]

    y = np.log(df["entry"] + log_offset)
    y.name = "log_entry"

    exog_cols  = [f"log_{f}" for f in exog_features]
    endog_cols = [f"log_{f}" for f in endog_features]
    instr_cols = [f"log_{c}" for cols in instruments.values() for c in cols]

    data = pd.concat([y, log_df], axis=1).dropna()
    y_clean    = data["log_entry"]
    exog_mat   = sm.add_constant(data[exog_cols]) if exog_cols else pd.DataFrame(
        np.ones((len(data), 1)), index=data.index, columns=["const"]
    )
    endog_mat  = data[endog_cols]
    instr_mat  = data[instr_cols]

    res = IV2SLS(
        dependent=y_clean,
        exog=exog_mat,
        endog=endog_mat,
        instruments=instr_mat,
    ).fit(cov_type="robust")

    ci = res.conf_int
    params     = res.params
    std_errors = res.std_errors
    tstats     = res.tstats
    pvalues    = res.pvalues

    coef_df = pd.DataFrame({
        "variable": params.index,
        "coef":     params.values,
        "se":       std_errors.values,
        "t":        tstats.values,
        "p":        pvalues.values,
        "ci_lo":    ci["lower"].values,
        "ci_hi":    ci["upper"].values,
    })
    coef_df["significant"] = coef_df["p"] < sig_level

    f_stat = res.f_statistic
    model_result = ModelResult(
        rsquared=float(res.rsquared),
        rsquared_adj=float(res.rsquared_adj) if hasattr(res, "rsquared_adj") else float("nan"),
        nobs=int(res.nobs),
        fvalue=float(f_stat.stat),
        f_pvalue=float(f_stat.pval),
        model_type="IV/2SLS",
    )
    return model_result, coef_df.reset_index(drop=True)


def build_equation_str(coef_df: pd.DataFrame, log_offset: float = 1.0) -> str:
    """
    Return a multi-line formatted equation string.

    Example:
        log(ridership) = +8.4230
          + 0.3120 · log(bus_stop_count + 1.0)
          - 0.0720 · log(taxi_count + 1.0)
    """
    intercept = float(coef_df.loc[coef_df["variable"] == "const", "coef"].values[0])
    lines = [f"log(ridership) = {intercept:+.4f}"]
    for _, row in coef_df[coef_df["variable"] != "const"].iterrows():
        sign = "+" if row["coef"] >= 0 else "-"
        feat_raw = row["variable"].replace("log_", "")
        lines.append(f"  {sign} {abs(row['coef']):.4f} · log({feat_raw} + {log_offset})")
    return "\n".join(lines)


def predict_ridership(
    coef_df: pd.DataFrame,
    feature_values: dict,
    log_offset: float = 1.0,
) -> float:
    """
    Compute exp(ŷ) from the coefficient table and a dict of raw feature values.

    Parameters
    ----------
    coef_df : DataFrame returned by fit_model or fit_iv_model
    feature_values : {raw_feature_name: value}, e.g. {"bus_stop_count": 4.0}
    log_offset : must match value used during model fitting

    Returns
    -------
    Predicted ridership (float, already exponentiated)
    """
    log_pred = float(coef_df.loc[coef_df["variable"] == "const", "coef"].values[0])
    for feat, val in feature_values.items():
        log_feat = f"log_{feat}"
        row = coef_df[coef_df["variable"] == log_feat]
        if not row.empty:
            log_pred += float(row["coef"].values[0]) * np.log(float(val) + log_offset)
    return float(np.exp(log_pred))


def elasticity_impact(coef: float, pct_change_x: float) -> float:
    """
    Point elasticity approximation.
    Expected % change in ridership ≈ coef × pct_change_x.
    Both input and output are in percentage points (e.g. 10 = 10%).
    """
    return coef * pct_change_x
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add model.py tests/__init__.py tests/test_model.py
git commit -m "feat: OLS and IV/2SLS model functions with unit tests"
```

---

## Task 3: App skeleton — data loading, sidebar with IV section, tab scaffold

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py**

```python
"""
app.py
------
Bangkok BTS/MRT Ridership Elasticity Dashboard
Run: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    ModelResult,
    build_equation_str,
    elasticity_impact,
    fit_iv_model,
    fit_model,
    predict_ridership,
)

DATA_PATH = "Output/combined_station_summary_expanded.csv"

ALL_FEATURES = [
    "bus_stop_count", "bus_stop_min_dist",
    "win_count",
    "songtaew_count",
    "minibus_count",
    "taxi_count", "taxi_min_dist",
    "tuktuk_count",
    "kiss_ride_count",
    "park_ride_car_count",
    "park_ride_moto_count",
    "bike_parking_count",
    "bike_share_count",
    "scooter_share_count",
]

DEFAULT_FEATURES = [
    "bus_stop_count", "bus_stop_min_dist",
    "win_count",
    "taxi_count",
    "kiss_ride_count",
    "bike_parking_count",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df.dropna(subset=["entry"])


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """All numeric columns except 'entry' — instrument candidates."""
    return [
        c for c in df.select_dtypes(include="number").columns
        if c != "entry"
    ]


def build_sidebar(df: pd.DataFrame) -> tuple:
    """
    Render sidebar controls.

    Returns
    -------
    selected_features : list[str]
    log_offset        : float
    sig_level         : float
    endog_features    : list[str]   (subset of selected_features)
    instruments       : dict[str, list[str]]  {endog_feat: [instrument_cols]}
    """
    st.sidebar.title("⚙️ Model Controls")

    # ── Features ──────────────────────────────────────────────────────────
    st.sidebar.subheader("Features")
    available = [f for f in ALL_FEATURES if f in df.columns]
    selected_features = [
        feat for feat in available
        if st.sidebar.checkbox(feat, value=(feat in DEFAULT_FEATURES), key=f"feat_{feat}")
    ]

    st.sidebar.subheader("Parameters")
    sig_level  = float(st.sidebar.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.20, step=0.01))
    log_offset = float(st.sidebar.number_input("Log offset", value=1.0, min_value=0.0, step=0.5))

    # ── Instrumental Variables ─────────────────────────────────────────────
    endog_features: list[str] = []
    instruments: dict[str, list[str]] = {}

    with st.sidebar.expander("🔧 Instrumental Variables", expanded=False):
        st.caption(
            "Mark regressors you suspect are endogenous, then choose "
            "instruments for each from the same CSV."
        )
        if not selected_features:
            st.info("Select features above first.")
        else:
            endog_features = st.multiselect(
                "Endogenous regressors",
                options=selected_features,
                default=[],
                key="endog_select",
            )
            all_numeric = get_numeric_columns(df)
            for endog_feat in endog_features:
                instrument_candidates = [c for c in all_numeric if c != endog_feat]
                chosen = st.multiselect(
                    f"Instruments for {endog_feat}",
                    options=instrument_candidates,
                    default=[],
                    key=f"instr_{endog_feat}",
                )
                if chosen:
                    instruments[endog_feat] = chosen

    return selected_features, log_offset, sig_level, endog_features, instruments


def run_model(
    df: pd.DataFrame,
    selected_features: list[str],
    endog_features: list[str],
    instruments: dict[str, list[str]],
    log_offset: float,
    sig_level: float,
) -> tuple[ModelResult, pd.DataFrame]:
    """Choose OLS or IV based on sidebar state and refit."""
    use_iv = bool(
        endog_features
        and all(endog in instruments and instruments[endog] for endog in endog_features)
    )
    if use_iv:
        exog_features = [f for f in selected_features if f not in endog_features]
        return fit_iv_model(df, exog_features, endog_features, instruments, log_offset, sig_level)
    return fit_model(df, selected_features, log_offset, sig_level)


def render_model_results(model_result: ModelResult, coef_df: pd.DataFrame,
                         df: pd.DataFrame, selected_features: list[str],
                         log_offset: float) -> None:
    pass  # Task 4 and 5


def render_station_explorer(df: pd.DataFrame, coef_df: pd.DataFrame,
                             selected_features: list[str], log_offset: float) -> None:
    pass  # Task 6


def render_whatif(df: pd.DataFrame, coef_df: pd.DataFrame,
                  selected_features: list[str], log_offset: float) -> None:
    pass  # Task 7


def main() -> None:
    st.set_page_config(page_title="Bangkok Ridership Elasticity", layout="wide", page_icon="🚉")
    df = load_data()
    selected_features, log_offset, sig_level, endog_features, instruments = build_sidebar(df)

    st.title("🚉 Bangkok Ridership Elasticity Dashboard")
    st.caption(f"{len(df)} stations · sources: {', '.join(df['source'].unique())}")

    if not selected_features:
        st.warning("Select at least one feature in the sidebar.")
        return

    model_result, coef_df = run_model(
        df, selected_features, endog_features, instruments, log_offset, sig_level
    )

    # Model type badge in main area
    badge_color = "#38bdf8" if model_result.model_type == "OLS" else "#a78bfa"
    st.markdown(
        f"<span style='background:{badge_color};color:#0f172a;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:700'>{model_result.model_type}</span>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📊 Model Results", "🗺️ Station Explorer", "🔮 What-if Simulator"])
    with tab1:
        render_model_results(model_result, coef_df, df, selected_features, log_offset)
    with tab2:
        render_station_explorer(df, coef_df, selected_features, log_offset)
    with tab3:
        render_whatif(df, coef_df, selected_features, log_offset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test skeleton**

```bash
streamlit run app.py
```

Expected: Title, sidebar checkboxes, "OLS" badge visible, three empty tabs, IV expander in sidebar. Selecting an endogenous variable shows instrument multiselects. Badge turns purple when IV/2SLS is active. No terminal errors.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: app skeleton with sidebar IV controls and tab scaffold"
```

---

## Task 4: Tab 1 — Equation banner and model stats

**Files:**
- Modify: `app.py` — replace `render_model_results` stub (equation + stats only)

- [ ] **Step 1: Replace the render_model_results stub**

Replace:
```python
def render_model_results(model_result: ModelResult, coef_df: pd.DataFrame,
                         df: pd.DataFrame, selected_features: list[str],
                         log_offset: float) -> None:
    pass  # Task 4 and 5
```

With:
```python
def render_model_results(model_result: ModelResult, coef_df: pd.DataFrame,
                         df: pd.DataFrame, selected_features: list[str],
                         log_offset: float) -> None:
    # ── Equation banner ───────────────────────────────────────────────────
    st.subheader("Fitted Model Equation")
    intercept = float(coef_df.loc[coef_df["variable"] == "const", "coef"].values[0])
    html_parts = [
        f"<div style='line-height:2'><span style='color:#f1f5f9;font-size:14px'>"
        f"log(ridership) = <span style='color:#fbbf24'>{intercept:+.4f}</span></span></div>"
    ]
    for _, row in coef_df[coef_df["variable"] != "const"].iterrows():
        sign = "+" if row["coef"] >= 0 else "−"
        feat_raw = row["variable"].replace("log_", "")
        coef_color = (
            "#4ade80" if row["significant"] and row["coef"] > 0
            else ("#f87171" if row["significant"] and row["coef"] < 0 else "#94a3b8")
        )
        html_parts.append(
            f"<div style='line-height:2;padding-left:16px'>"
            f"<span style='color:#64748b'>{sign} </span>"
            f"<span style='color:{coef_color}'>{abs(row['coef']):.4f}</span>"
            f"<span style='color:#94a3b8'> · log({feat_raw} + {log_offset})</span>"
            f"</div>"
        )
    legend = (
        "<div style='margin-top:10px;font-size:11px'>"
        "<span style='color:#4ade80'>■ significant positive</span>&nbsp;&nbsp;"
        "<span style='color:#f87171'>■ significant negative</span>&nbsp;&nbsp;"
        "<span style='color:#94a3b8'>■ not significant (p ≥ α)</span></div>"
    )
    st.markdown(
        "<div style='background:#1e293b;padding:16px 20px;border-radius:8px;"
        "font-family:monospace;border:1px solid #334155'>"
        + "".join(html_parts) + legend + "</div>",
        unsafe_allow_html=True,
    )

    # ── Model stats ───────────────────────────────────────────────────────
    st.subheader("Model Statistics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²",           f"{model_result.rsquared:.4f}")
    c2.metric("Adj R²",       f"{model_result.rsquared_adj:.4f}")
    c3.metric("Observations", model_result.nobs)
    c4.metric("F-statistic",  f"{model_result.fvalue:.2f}")
    c5.metric("F p-value",    f"{model_result.f_pvalue:.4f}")
```

- [ ] **Step 2: Smoke-test**

```bash
streamlit run app.py
```

Expected: Tab 1 shows color-coded equation banner and five metric cards. Switch to IV mode (sidebar) — stats update, badge turns purple. No errors.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Tab 1 equation banner and model stats"
```

---

## Task 5: Tab 1 — Coefficient chart and station table

**Files:**
- Modify: `app.py` — append chart and table to `render_model_results`

- [ ] **Step 1: Append to the end of the render_model_results function body**

```python
    # ── Coefficient chart ─────────────────────────────────────────────────
    st.subheader("Elasticity Coefficients (95% CI)")
    plot_df = coef_df[coef_df["variable"] != "const"].copy()
    plot_df = plot_df.sort_values("coef", key=abs, ascending=True)
    bar_colors = [
        "#4ade80" if (r["significant"] and r["coef"] > 0)
        else ("#f87171" if (r["significant"] and r["coef"] < 0) else "#94a3b8")
        for _, r in plot_df.iterrows()
    ]
    fig_coef = go.Figure(go.Bar(
        y=plot_df["variable"],
        x=plot_df["coef"],
        orientation="h",
        marker_color=bar_colors,
        error_x=dict(
            type="data", symmetric=False,
            array=(plot_df["ci_hi"] - plot_df["coef"]).tolist(),
            arrayminus=(plot_df["coef"] - plot_df["ci_lo"]).tolist(),
            color="#475569", thickness=2,
        ),
        hovertemplate="<b>%{y}</b><br>Elasticity: %{x:.4f}<extra></extra>",
    ))
    fig_coef.add_vline(x=0, line_color="#475569", line_width=1)
    fig_coef.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
        xaxis_title="Elasticity estimate",
        height=max(300, len(plot_df) * 38),
        margin=dict(l=0, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    # ── Station table ─────────────────────────────────────────────────────
    st.subheader("Stations — Actual vs Predicted")
    station_df = df[["station_name", "source", "entry"]].copy()
    station_df["predicted"] = df.apply(
        lambda row: predict_ridership(coef_df, {f: row[f] for f in selected_features}, log_offset),
        axis=1,
    )
    station_df["residual"]    = station_df["entry"] - station_df["predicted"]
    station_df["pct_error_%"] = (station_df["residual"] / station_df["entry"] * 100).round(1)
    station_df[["entry", "predicted", "residual"]] = (
        station_df[["entry", "predicted", "residual"]].round(0).astype(int)
    )
    station_df = station_df.rename(columns={
        "station_name": "Station", "source": "Source",
        "entry": "Actual",        "predicted": "Predicted",
        "residual": "Residual",
    })
    st.dataframe(station_df, use_container_width=True, height=420)
```

- [ ] **Step 2: Smoke-test**

```bash
streamlit run app.py
```

Expected: Tab 1 shows all four sections. Switch OLS ↔ IV — equation, chart, and table all update. No errors.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Tab 1 coefficient chart and station table"
```

---

## Task 6: Tab 2 — Station Explorer

**Files:**
- Modify: `app.py` — replace `render_station_explorer` stub

- [ ] **Step 1: Replace render_station_explorer stub**

Replace:
```python
def render_station_explorer(df: pd.DataFrame, coef_df: pd.DataFrame,
                             selected_features: list[str], log_offset: float) -> None:
    pass  # Task 6
```

With:
```python
def render_station_explorer(df: pd.DataFrame, coef_df: pd.DataFrame,
                             selected_features: list[str], log_offset: float) -> None:
    st.subheader("Station Explorer")
    selected_station = st.selectbox("Select a station", df["station_name"].tolist(), key="explorer_station")
    row = df[df["station_name"] == selected_station].iloc[0]
    feat_data = {f: float(row[f]) for f in selected_features if f in row.index}

    col_info, col_pred = st.columns(2)
    with col_info:
        st.markdown("**Station Info**")
        st.write(f"Source: `{row['source']}`")
        if "line_code" in row.index:
            st.write(f"Line code: `{row['line_code']}`")
        st.markdown("**Feature Values**")
        st.dataframe(
            pd.DataFrame({"Feature": list(feat_data.keys()), "Value": list(feat_data.values())}),
            use_container_width=True, hide_index=True,
        )

    with col_pred:
        actual    = float(row["entry"])
        predicted = predict_ridership(coef_df, feat_data, log_offset)
        residual  = actual - predicted
        pct_err   = residual / actual * 100
        st.markdown("**Prediction**")
        m1, m2 = st.columns(2)
        m1.metric("Actual ridership",    f"{actual:,.0f}")
        m2.metric("Predicted ridership", f"{predicted:,.0f}")
        m3, m4 = st.columns(2)
        m3.metric("Residual", f"{residual:+,.0f}")
        m4.metric("% Error",  f"{pct_err:+.1f}%")

    st.subheader("Elasticity Contribution per Feature")
    st.caption("α_i · log(x_i + offset) for this station.")
    contrib_rows = []
    for feat, val in feat_data.items():
        log_feat = f"log_{feat}"
        coef_row = coef_df[coef_df["variable"] == log_feat]
        if coef_row.empty:
            continue
        coef_val = float(coef_row["coef"].values[0])
        contrib_rows.append({
            "Feature":      feat,
            "Contribution": coef_val * float(np.log(val + log_offset)),
            "Coef":         coef_val,
        })
    contrib_df = pd.DataFrame(contrib_rows).sort_values("Contribution", key=abs, ascending=True)
    fig_contrib = go.Figure(go.Bar(
        y=contrib_df["Feature"],
        x=contrib_df["Contribution"],
        orientation="h",
        marker_color=["#4ade80" if v >= 0 else "#f87171" for v in contrib_df["Contribution"]],
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
    ))
    fig_contrib.add_vline(x=0, line_color="#475569", line_width=1)
    fig_contrib.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
        xaxis_title="α_i · log(x_i + offset)",
        height=max(250, len(contrib_df) * 38),
        margin=dict(l=0, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_contrib, use_container_width=True)
```

- [ ] **Step 2: Smoke-test**

```bash
streamlit run app.py
```

Expected: Tab 2 shows station dropdown, feature table, prediction metrics, contribution chart. Switching stations updates all sections. No errors.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Tab 2 station explorer"
```

---

## Task 7: Tab 3 — What-if Simulator

**Files:**
- Modify: `app.py` — replace `render_whatif` stub

- [ ] **Step 1: Replace render_whatif stub**

Replace:
```python
def render_whatif(df: pd.DataFrame, coef_df: pd.DataFrame,
                  selected_features: list[str], log_offset: float) -> None:
    pass  # Task 7
```

With:
```python
def render_whatif(df: pd.DataFrame, coef_df: pd.DataFrame,
                  selected_features: list[str], log_offset: float) -> None:
    st.subheader("What-if Simulator")
    st.caption("Choose a base station, adjust sliders, see how predicted ridership changes.")

    base_station = st.selectbox("Base station", df["station_name"].tolist(), key="whatif_station")
    base_row = df[df["station_name"] == base_station].iloc[0]
    base_feat_vals = {f: float(base_row[f]) for f in selected_features if f in base_row.index}
    base_pred = predict_ridership(coef_df, base_feat_vals, log_offset)

    st.markdown("---")
    st.markdown("**Adjust feature values:**")
    new_vals: dict[str, float] = {}
    slider_cols = st.columns(min(3, len(selected_features)))
    for i, feat in enumerate(selected_features):
        base_val = base_feat_vals.get(feat, 0.0)
        max_val  = max(float(df[feat].max()) * 1.5, base_val + 1)
        step     = 1.0 if feat.endswith("_count") else 10.0
        new_vals[feat] = slider_cols[i % len(slider_cols)].slider(
            feat, min_value=0.0, max_value=float(max_val),
            value=float(base_val), step=step, key=f"slider_{feat}",
        )

    new_pred   = predict_ridership(coef_df, new_vals, log_offset)
    abs_change = new_pred - base_pred
    pct_change = abs_change / base_pred * 100 if base_pred > 0 else 0.0

    st.markdown("---")
    st.subheader("Prediction Output")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Base predicted",  f"{base_pred:,.0f}")
    o2.metric("New predicted",   f"{new_pred:,.0f}")
    o3.metric("Change",          f"{abs_change:+,.0f}")
    o4.metric("% Change",        f"{pct_change:+.1f}%")

    st.subheader("Elasticity Impact per Feature")
    st.caption("α_i × Δ%X_i — point elasticity approximation per feature.")
    impact_rows = []
    for feat in selected_features:
        base_val = base_feat_vals.get(feat, 0.0)
        new_val  = new_vals.get(feat, 0.0)
        coef_row = coef_df[coef_df["variable"] == f"log_{feat}"]
        if coef_row.empty:
            continue
        coef_val = float(coef_row["coef"].values[0])
        pct_x    = (new_val - base_val) / (base_val + 1e-9) * 100
        impact_rows.append({
            "Feature":                feat,
            "Base value":             base_val,
            "New value":              new_val,
            "Δ% feature":             round(pct_x, 1),
            "Elasticity (α)":         round(coef_val, 4),
            "Expected Δ% ridership":  round(elasticity_impact(coef_val, pct_x), 2),
        })
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)
```

- [ ] **Step 2: Smoke-test**

```bash
streamlit run app.py
```

Expected: Tab 3 shows station dropdown, feature sliders, four metric cards, impact table. Moving a slider instantly updates metrics and table. No errors.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Tab 3 what-if simulator"
```

---

## Task 8: Final smoke-test and GitHub prep

**Files:**
- No new files

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All 11 tests PASS.

- [ ] **Step 2: Full app walkthrough**

```bash
streamlit run app.py
```

Verify manually:
- Tab 1: toggle a feature off → equation, chart, and table all update
- Tab 1: change α → colour coding updates
- Sidebar IV expander: mark `bus_stop_count` as endogenous, select `bus_stop_min_dist` as instrument → badge turns purple (IV/2SLS), equation and stats update
- Remove endogenous selection → reverts to OLS badge
- Tab 2: switch stations → all panels update
- Tab 3: move a slider → metrics and impact table update instantly

- [ ] **Step 3: Add .gitignore entries**

```bash
echo ".superpowers/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".streamlit/" >> .gitignore
```

- [ ] **Step 4: Final commit**

```bash
git add .gitignore
git commit -m "chore: gitignore for streamlit and superpowers artifacts"
```

**Deployment reminder (manual, done by user):**
1. Push repo to GitHub
2. Go to share.streamlit.io → New app → select repo → set main file to `app.py`
3. Deploy → share the generated URL
