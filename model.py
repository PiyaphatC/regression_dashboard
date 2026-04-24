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
    features: list,
    log_offset: float = 1.0,
    sig_level: float = 0.05,
) -> tuple:
    """
    Fit a log-log OLS model with HC3 robust SEs.

    Returns
    -------
    ModelResult, coef_df
        coef_df columns: variable, coef, se, t, p, ci_lo, ci_hi, significant
        'variable' values are 'const' or 'log_{feature_name}'
    """
    if not features:
        raise ValueError("features must be a non-empty list")
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
        rsquared=float(res.rsquared),
        rsquared_adj=float(res.rsquared_adj),
        nobs=int(res.nobs),
        fvalue=float(res.fvalue),
        f_pvalue=float(res.f_pvalue),
        model_type="OLS",
    )
    return model_result, coef_df.reset_index(drop=True)


def fit_iv_model(
    df: pd.DataFrame,
    exog_features: list,
    endog_features: list,
    instruments: dict,
    log_offset: float = 1.0,
    sig_level: float = 0.05,
) -> tuple:
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
    instr_cols_raw = list({c for cols in instruments.values() for c in cols})

    # Build log-transformed matrices
    log_df = df[all_features + instr_cols_raw].apply(
        lambda s: np.log(s + log_offset)
    )
    log_df.columns = [f"log_{c}" for c in log_df.columns]

    y = np.log(df["entry"] + log_offset)
    y.name = "log_entry"

    exog_cols  = [f"log_{f}" for f in exog_features]
    endog_cols = [f"log_{f}" for f in endog_features]
    instr_cols = [f"log_{c}" for c in instr_cols_raw]

    data = pd.concat([y, log_df], axis=1).dropna()
    y_clean = data["log_entry"]

    if exog_cols:
        exog_mat = sm.add_constant(data[exog_cols])
    else:
        exog_mat = pd.DataFrame(
            np.ones((len(data), 1)), index=data.index, columns=["const"]
        )

    endog_mat = data[endog_cols]
    instr_mat = data[instr_cols]

    res = IV2SLS(
        dependent=y_clean,
        exog=exog_mat,
        endog=endog_mat,
        instruments=instr_mat,
    ).fit(cov_type="robust")

    # conf_int is a method in linearmodels (not a property) — call with ()
    ci = res.conf_int()
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
    rsq_adj = float(res.rsquared_adj) if hasattr(res, "rsquared_adj") else float("nan")

    model_result = ModelResult(
        rsquared=float(res.rsquared),
        rsquared_adj=rsq_adj,
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
        feat_raw = row["variable"][4:]  # strip exactly the leading 'log_' prefix
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
            safe_val = max(float(val) + log_offset, 1e-9)
            log_pred += float(row["coef"].values[0]) * np.log(safe_val)
    return float(np.exp(log_pred))


def elasticity_impact(coef: float, pct_change_x: float) -> float:
    """
    Point elasticity approximation.
    Expected % change in ridership ≈ coef × pct_change_x.
    Both input and output are in percentage points (e.g. 10 = 10%).
    """
    return coef * pct_change_x
