# Elasticity Dashboard — Design Spec
**Date:** 2026-04-24 (updated with IV support)
**Project:** Bangkok BTS/MRT Station-Level Ridership Elasticity

---

## Overview

A Streamlit web dashboard that lets researchers explore ridership elasticity estimates per station, inspect the fitted model, and run what-if scenarios by adjusting station-level feature inputs. Supports both plain OLS and Two-Stage Least Squares (IV/2SLS) estimation — switchable from the sidebar.

Deployed for free on **Streamlit Community Cloud** (shareable URL; no installation for colleagues).

---

## Data & Model

- **Input data:** `Output/combined_station_summary_expanded.csv` (192 stations, 41 columns)
- **Dependent variable:** `entry` (daily ridership)
- **Default model:** log-log OLS with HC3-robust standard errors
- **IV model:** log-log IV/2SLS (Two-Stage Least Squares) via `linearmodels.iv.IV2SLS`, activated when the user designates at least one endogenous regressor and selects instruments for it
- **Log transform:** `log(x + offset)` where offset defaults to `1.0`
- **Features & instruments:** all numeric columns from the same CSV (`_count`, `_min_dist`, `_mean_dist` per mode)

The model is refit in-memory whenever the user changes sidebar controls. No results are written to disk from the app.

---

## File Structure

```
app.py                                          ← single Streamlit entry point
model.py                                        ← pure model functions (OLS + IV, testable)
requirements.txt                                ← streamlit, pandas, statsmodels, linearmodels, numpy, plotly
Output/combined_station_summary_expanded.csv    ← data (existing)
tests/test_model.py                             ← unit tests for model.py
```

---

## Sidebar Controls

Visible on every tab. Changing any control triggers an immediate model refit.

### Section 1 — Features
| Control | Type | Default |
|---|---|---|
| Feature checkboxes | One checkbox per available metric column | Key mode counts selected by default |
| Significance level (α) | Number input | `0.05` |
| Log offset | Number input | `1.0` |

### Section 2 — Instrumental Variables (expander, collapsed by default)

| Control | Type | Behaviour |
|---|---|---|
| Endogenous regressors | Multiselect from **selected features** | Features the user suspects are endogenous |
| Instruments per endogenous var | Multiselect (one per endogenous feature) from **all available columns minus the endogenous one** | Columns to use as instruments in the first stage |

**Model switching logic:**
- If no endogenous regressors are designated → fit OLS (HC3)
- If ≥ 1 endogenous regressor with ≥ 1 instrument each → fit IV/2SLS (robust SEs)
- A badge in the top bar shows the active model type: `OLS` or `IV/2SLS`

---

## Tab 1 — Model Results

### 1. Equation Banner
Full fitted equation rendered at the top of the tab:

```
log(ridership) = 8.423
               + 0.312 · log(bus_stop_count + 1)
               + 0.184 · log(win_count + 1)
               − 0.072 · log(taxi_count + 1)
               + ...
```

Color coding:
- **Green** — significant positive (p < α)
- **Red** — significant negative (p < α)
- **Grey** — not significant

Updates live when features, IV settings, or parameters change.

### 2. Model Stats Row
R², Adj R², n (observations), F-statistic, F p-value, **Model type badge (OLS / IV·2SLS)** — displayed as metric cards.

### 3. Coefficient Chart
Horizontal bar chart (Plotly) of elasticity estimates with 95% CI error bars. Bars colored by significance (green / red / grey). Variables sorted by absolute elasticity value.

### 4. Station Table
All stations in a sortable, searchable `st.dataframe`:

| Column | Description |
|---|---|
| Station | Name + line code |
| Source | A21 / AEC / CUTI |
| Actual ridership | `entry` value |
| Predicted ridership | `exp(ŷ)` |
| Residual | Actual − Predicted |
| % Error | (Actual − Predicted) / Actual × 100 |

---

## Tab 2 — Station Explorer

- **Station selector** — searchable dropdown by station name or line code
- **Feature breakdown table** — raw feature values for the selected station
- **Prediction card** — predicted ridership, actual ridership, residual, % error
- **Elasticity contribution chart** — horizontal bar chart showing each feature's additive contribution to `log(predicted ridership)` for that station: `α_i × log(x_i + offset)`

Tabs 2 and 3 consume `coef_df` only — they work identically regardless of whether OLS or IV was used to produce the coefficients.

---

## Tab 3 — What-if Simulator

- **Base station selector** — choose any station; its real feature values pre-fill all sliders
- **Feature sliders** — one per active feature; integer step for count features, float for distance features
- **Prediction output:**
  - Predicted ridership: `exp(α₀ + Σ αᵢ · log(xᵢ + offset))`
  - Change from base station: absolute and % difference
- **Elasticity impact table** — for each feature:
  - Base value → new value
  - % change in feature
  - Expected % change in ridership (`αᵢ × % change in xᵢ`, point elasticity approximation)

---

## Core Functions (`model.py`)

| Function | Signature | Purpose |
|---|---|---|
| `fit_model` | `(df, features, log_offset, sig_level) → (ModelResult, coef_df)` | OLS with HC3 SEs |
| `fit_iv_model` | `(df, exog_features, endog_features, instruments, log_offset, sig_level) → (ModelResult, coef_df)` | IV/2SLS via linearmodels |
| `build_equation_str` | `(coef_df, log_offset) → str` | Multi-line formatted equation |
| `predict_ridership` | `(coef_df, feature_values, log_offset) → float` | Scalar prediction from coef table |
| `elasticity_impact` | `(coef, pct_change_x) → float` | Point elasticity approximation |

**`ModelResult` namedtuple** (normalises statsmodels vs linearmodels output):
```python
ModelResult = namedtuple("ModelResult", ["rsquared", "rsquared_adj", "nobs", "fvalue", "f_pvalue", "model_type"])
# model_type: "OLS" or "IV/2SLS"
```

Both `fit_model` and `fit_iv_model` return `(ModelResult, coef_df)` with an identical `coef_df` schema:
`[variable, coef, se, t, p, ci_lo, ci_hi, significant]`

---

## Deployment

1. Place `app.py`, `model.py`, and `requirements.txt` in the project root
2. Push to GitHub (private repo is fine)
3. Connect to [Streamlit Community Cloud](https://streamlit.io/cloud) — free tier
4. Set the main file path to `app.py`
5. Share the auto-generated URL with colleagues

No secrets or environment variables required.

---

## Out of Scope

- Writing results to disk from the app
- User authentication
- Multi-user session isolation (Streamlit handles this automatically)
- Deploying the CSV via a database — flat file is sufficient for 192 stations
- External instrument data (instruments come from the same CSV only)
