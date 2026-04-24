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
    # ── Equation banner ───────────────────────────────────────────────────
    st.subheader("Fitted Model Equation")
    const_row = coef_df[coef_df["variable"] == "const"]
    if const_row.empty:
        st.error("Internal error: model missing constant term.")
        return
    intercept = float(const_row["coef"].values[0])
    html_parts = [
        f"<div style='line-height:2'><span style='color:#f1f5f9;font-size:14px'>"
        f"log(ridership) = <span style='color:#fbbf24'>{intercept:+.4f}</span></span></div>"
    ]
    for _, row in coef_df[coef_df["variable"] != "const"].iterrows():
        sign = "+" if row["coef"] >= 0 else "−"
        feat_raw = row["variable"][4:]  # strip leading 'log_'
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
    adj_r2 = model_result.rsquared_adj
    c2.metric("Adj R²", f"{adj_r2:.4f}" if not np.isnan(adj_r2) else "—")
    c3.metric("Observations", model_result.nobs)
    c4.metric("F-statistic",  f"{model_result.fvalue:.2f}")
    c5.metric("F p-value",    f"{model_result.f_pvalue:.4f}")

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
            "Contribution": coef_val * float(np.log(max(val + log_offset, 1e-9))),
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


def render_whatif(df: pd.DataFrame, coef_df: pd.DataFrame,
                  selected_features: list[str], log_offset: float) -> None:
    st.subheader("What-if Simulator")
    st.caption("Choose a base station, adjust sliders, see how predicted ridership changes.")

    base_station = st.selectbox("Base station", df["station_name"].tolist(), key="whatif_station")
    base_row = df[df["station_name"] == base_station].iloc[0]
    base_feat_vals = {f: float(base_row[f]) for f in selected_features if f in base_row.index}
    base_pred = predict_ridership(coef_df, base_feat_vals, log_offset)

    # Reset sliders to the new station's original values when the station changes
    if st.session_state.get("_whatif_prev_station") != base_station:
        for feat in selected_features:
            slider_key = f"slider_{feat}"
            if slider_key in st.session_state:
                del st.session_state[slider_key]
        st.session_state["_whatif_prev_station"] = base_station

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
            "Feature":               feat,
            "Base value":            base_val,
            "New value":             new_val,
            "Δ% feature":            round(pct_x, 1),
            "Elasticity (α)":        round(coef_val, 4),
            "Expected Δ% ridership": round(elasticity_impact(coef_val, pct_x), 2),
        })
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)


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
