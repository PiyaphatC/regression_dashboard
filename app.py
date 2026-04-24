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
