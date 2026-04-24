"""
02_elasticity_model.py
----------------------
Estimates ridership elasticity with respect to station-level accessibility
features using log-log OLS regression (pooled cross-section).

Interpretation  : A coefficient β on log(X) means a 1% increase in X is
                  associated with a β% change in ridership — i.e. β is the
                  direct elasticity estimate.

Inputs  : Output/combined_station_summary.csv   (from 01_combine_sources.py)
          [ridership CSV to be merged]           (set RIDERSHIP_FILE below)
Output  : Output/elasticity_results.csv
          Output/elasticity_results.txt          (human-readable summary)

Usage   : python 02_elasticity_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration  ← edit this block to customise the model
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

# --- Input files ---
STATION_FILE   = os.path.join(OUTPUT_DIR, "combined_station_summary.csv")

# Set to None if ridership is already a column in STATION_FILE
RIDERSHIP_FILE = None          # e.g. "ridership_data.csv"
RIDERSHIP_JOIN_KEY = "station" # column used to merge ridership → station data
RIDERSHIP_COL  = "ridership"   # dependent variable column name

# --- Feature selection ---
# For each transport mode, choose which metric(s) to use as regressors.
# Options per mode: "{mode}_count", "{mode}_min_dist", "{mode}_mean_dist"
# Set to an empty list to exclude a mode entirely.
FEATURE_CONFIG: dict[str, list[str]] = {
    "bus_stop":      ["bus_stop_count",      "bus_stop_min_dist"],
    "win":           ["win_count"],
    "songtaew":      ["songtaew_count"],
    "minibus":       ["minibus_count"],
    "taxi":          ["taxi_count",          "taxi_min_dist"],
    "tuktuk":        ["tuktuk_count"],
    "kiss_ride":     ["kiss_ride_count"],
    "park_ride_car": ["park_ride_car_count"],
    "park_ride_moto":["park_ride_moto_count"],
    "bike_parking":  ["bike_parking_count"],
    "bike_share":    ["bike_share_count"],
    "scooter_share": ["scooter_share_count"],
    # --- add future POI variables here, e.g.:
    # "retail_poi":  ["retail_poi_count"],
}

# --- Model controls ---
CONTROL_COLS: list[str] = []
# e.g. ["district_dummy", "year"] — add after POI merge

# --- Log-transform options ---
# Adds a small constant before log() to avoid log(0).
LOG_OFFSET = 1.0

# --- Output ---
SIGNIFICANCE_LEVEL = 0.05   # used to flag significant coefficients in summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_transform(series: pd.Series, offset: float = LOG_OFFSET) -> pd.Series:
    """Apply log(x + offset) transformation."""
    return np.log(series + offset)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten FEATURE_CONFIG into a matrix of log-transformed regressors.
    Columns in FEATURE_CONFIG that are missing from df are skipped with a warning.
    """
    selected = []
    for mode, cols in FEATURE_CONFIG.items():
        for col in cols:
            if col not in df.columns:
                print(f"  [WARN] Feature '{col}' not found in data — skipping.")
                continue
            selected.append(col)

    X = df[selected].copy()
    X = X.apply(log_transform)
    X.columns = [f"log_{c}" for c in X.columns]
    return X


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Return variance inflation factors to diagnose multicollinearity."""
    vif_data = pd.DataFrame({
        "feature": X.columns,
        "VIF": [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def format_results(result, sig_level: float = SIGNIFICANCE_LEVEL) -> pd.DataFrame:
    """Extract coefficient table from OLS result into a tidy DataFrame."""
    summary = pd.DataFrame({
        "variable":    result.params.index,
        "elasticity":  result.params.values,
        "std_error":   result.bse.values,
        "t_stat":      result.tvalues.values,
        "p_value":     result.pvalues.values,
        "ci_lower":    result.conf_int()[0].values,
        "ci_upper":    result.conf_int()[1].values,
    })
    summary["significant"] = summary["p_value"] < sig_level
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load station data ---
    print(f"\nLoading station data: {STATION_FILE}")
    if not os.path.exists(STATION_FILE):
        raise FileNotFoundError(
            f"Station file not found: {STATION_FILE}\n"
            "Run 01_combine_sources.py first."
        )
    df = pd.read_csv(STATION_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    print(f"  {len(df)} stations, {len(df.columns)} columns loaded.")

    # --- 2. Merge ridership (if external) ---
    if RIDERSHIP_FILE is not None:
        ridership_path = os.path.join(BASE_DIR, RIDERSHIP_FILE)
        print(f"\nMerging ridership data: {ridership_path}")
        ridership = pd.read_csv(ridership_path, encoding="utf-8-sig")
        ridership.columns = ridership.columns.str.replace("\ufeff", "", regex=False).str.strip()
        df = df.merge(ridership[[RIDERSHIP_JOIN_KEY, RIDERSHIP_COL]],
                      on=RIDERSHIP_JOIN_KEY, how="inner")
        print(f"  {len(df)} stations retained after merge.")
    else:
        if RIDERSHIP_COL not in df.columns:
            raise ValueError(
                f"Dependent variable '{RIDERSHIP_COL}' not found in data.\n"
                "Either set RIDERSHIP_FILE to an external file or ensure the "
                "column exists in combined_station_summary.csv."
            )

    # --- 3. Drop rows with missing dependent variable ---
    before = len(df)
    df = df.dropna(subset=[RIDERSHIP_COL])
    if len(df) < before:
        print(f"  [INFO] Dropped {before - len(df)} rows with missing '{RIDERSHIP_COL}'.")

    # --- 4. Build feature matrix ---
    print("\nBuilding log-transformed feature matrix...")
    X = build_feature_matrix(df)

    # Add controls (not log-transformed by default)
    for col in CONTROL_COLS:
        if col in df.columns:
            X[col] = df[col].values
        else:
            print(f"  [WARN] Control '{col}' not found — skipping.")

    X = sm.add_constant(X)

    # --- 5. Dependent variable ---
    y = log_transform(df[RIDERSHIP_COL])
    y.name = f"log_{RIDERSHIP_COL}"

    # --- 6. Align and drop remaining NaNs ---
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]

    print(f"  Final sample: {len(data)} observations, {X_clean.shape[1]} regressors.")

    # --- 7. VIF diagnostic ---
    X_no_const = X_clean.drop(columns=["const"], errors="ignore")
    vif = compute_vif(X_no_const)
    high_vif = vif[vif["VIF"] > 10]
    if not high_vif.empty:
        print(
            f"\n  [WARNING] High multicollinearity detected (VIF > 10):\n"
            f"{high_vif.to_string(index=False)}"
        )

    # --- 8. Fit OLS ---
    print("\nFitting log-log OLS model...")
    model  = sm.OLS(y_clean, X_clean)
    result = model.fit(cov_type="HC3")   # heteroskedasticity-robust SEs

    # --- 9. Format and save results ---
    coef_table = format_results(result)

    coef_path = os.path.join(OUTPUT_DIR, "elasticity_results.csv")
    coef_table.to_csv(coef_path, index=False, encoding="utf-8-sig")

    # Human-readable summary
    summary_path = os.path.join(OUTPUT_DIR, "elasticity_results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())
        f.write("\n\n--- VIF Diagnostics ---\n")
        f.write(vif.to_string(index=False))

    # --- 10. Console summary ---
    print(f"\n{'='*60}")
    print(f"  Dependent variable : log({RIDERSHIP_COL})")
    print(f"  Observations       : {int(result.nobs)}")
    print(f"  R²                 : {result.rsquared:.4f}")
    print(f"  Adj. R²            : {result.rsquared_adj:.4f}")
    print(f"  F-statistic (p)    : {result.fvalue:.2f}  ({result.f_pvalue:.4f})")
    print(f"{'='*60}")

    sig = coef_table[coef_table["significant"] & (coef_table["variable"] != "const")]
    if not sig.empty:
        print(f"\n  Significant elasticities (p < {SIGNIFICANCE_LEVEL}):\n")
        print(
            sig[["variable", "elasticity", "p_value", "ci_lower", "ci_upper"]]
            .to_string(index=False, float_format="{:.4f}".format)
        )
    else:
        print("\n  No regressors significant at the chosen threshold.")

    print(f"\n  Results saved to:\n    {coef_path}\n    {summary_path}")


if __name__ == "__main__":
    main()
