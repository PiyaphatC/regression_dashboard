"""
app.py
------
Bangkok BTS/MRT Ridership Elasticity Dashboard
Run: streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    ModelResult,
    build_equation_str,
    elasticity_impact,
    fit_iv_model,
    fit_linear_model,
    fit_model,
    fit_semi_log_model,
    predict_ridership,
)

DATA_PATH = "Output/combined_station_summary_expanded_rev14.csv"
RADII_PATH = "Output/features_by_radius.csv"

NON_FEATURE_COLS = {"entry", "source", "line_code", "station_name", "station", "display_name",
                    "line_color", "station_type", "station_typology", "radius_m",
                    "n_zones_in_buffer"}

# Mapping from line_code prefix → (line colour, system type)
_LINE_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "BL": ("🔵 Blue",    "MRT"),
    "PP": ("🟣 Purple",  "MRT"),
    "N":  ("🟢 Green",   "BTS"),
    "E":  ("🟢 Green",   "BTS"),
    "S":  ("🟢 Green",   "BTS"),
    "W":  ("🟢 Green",   "BTS"),
    "CEN":("🟢 Green",   "BTS"),
    "G":  ("🟡 Gold",    "BTS"),
    "A":  ("🔴 Airport", "ARL"),
    "PK": ("🩷 Pink",    "Monorail"),
    "YL": ("🟡 Yellow",  "Monorail"),
    "RN": ("🔴 Red",     "SRT"),
    "RW": ("🔴 Red",     "SRT"),
    "MT": ("🟡 Yellow",  "Monorail"),
}


def _classify_line(code: str) -> tuple[str, str]:
    """Return (line_color, station_type) from a line_code like 'BL01'."""
    for prefix in sorted(_LINE_PREFIX_MAP, key=len, reverse=True):  # longest first
        if code.startswith(prefix):
            return _LINE_PREFIX_MAP[prefix]
    return ("Other", "Other")

DEFAULT_FEATURES = [
    # Sidewalk quality — surface (length with poor rating)
    "sidewalk_length_surface_neg1",
    # Sidewalk quality — shade (length with poor rating)
    "sidewalk_length_shade_neg1",
    # Sidewalk quality — obstacle-free (length with poor rating)
    "sidewalk_length_obstacle_neg1",
    # Last-mile feeder modes
    "win_count",
    "bike_share_count",
    "taxi_count",
    # Bus connectivity
    "bus_stop_mean_dist",
]


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df = df.dropna(subset=["entry"])
    classified = df["line_code"].apply(_classify_line)
    df["line_color"]   = classified.apply(lambda x: x[0])
    df["station_type"] = classified.apply(lambda x: x[1])
    df["display_name"] = df["line_code"] + " — " + df["station_name"]
    return df


@st.cache_data
def load_radii_data(path: str = RADII_PATH) -> pd.DataFrame:
    """Load the pre-computed multi-radius feature CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df = df.dropna(subset=["entry"])
    classified = df["line_code"].apply(_classify_line)
    df["line_color"]   = classified.apply(lambda x: x[0])
    df["station_type"] = classified.apply(lambda x: x[1])
    df["display_name"] = df["line_code"] + " — " + df["station_name"]
    return df


def get_radii_list(df_radii: pd.DataFrame) -> list[int]:
    return sorted(df_radii["radius_m"].unique().astype(int).tolist())


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """All numeric columns except 'entry' — instrument candidates."""
    return [
        c for c in df.select_dtypes(include="number").columns
        if c not in ("entry", "radius_m")
    ]


def _forward_stepwise(df: pd.DataFrame, candidates: list[str],
                      model_spec: str, log_offset: float, sig_level: float,
                      mode: str = "best_r2",
                      max_features: int = 15) -> tuple[list[str], float, int]:
    """Forward stepwise feature selection.

    mode="best_r2"    : maximise Adj R²
    mode="max_signif" : maximise number of significant variables (p < α),
                        ties broken by Adj R²

    Returns (best_features, best_adj_r2, n_significant).
    """
    selected: list[str] = []
    best_adj_r2 = -np.inf
    best_n_sig = 0

    for _ in range(min(max_features, len(candidates))):
        improved = False
        best_feat = None
        for feat in candidates:
            if feat in selected:
                continue
            trial = selected + [feat]
            try:
                mr, cdf = run_model(df, trial, [], {}, log_offset, sig_level, model_spec)
            except Exception:
                continue
            n_sig = int(cdf[(cdf["variable"] != "const") & cdf["significant"]].shape[0])
            adj_r2 = mr.rsquared_adj

            if mode == "best_r2":
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_n_sig = n_sig
                    best_feat = feat
                    improved = True
            else:  # max_signif
                if (n_sig > best_n_sig) or (n_sig == best_n_sig and adj_r2 > best_adj_r2):
                    best_n_sig = n_sig
                    best_adj_r2 = adj_r2
                    best_feat = feat
                    improved = True

        if not improved or best_feat is None:
            break
        selected.append(best_feat)
    return selected, best_adj_r2, best_n_sig


def build_sidebar(df: pd.DataFrame, radii_list: list[int] | None = None) -> tuple:
    """
    Render sidebar controls.

    Returns
    -------
    selected_stations : list[str]
    selected_features : list[str]
    log_offset        : float
    sig_level         : float
    model_spec        : str  ("log-log" or "linear")
    endog_features    : list[str]   (subset of selected_features)
    instruments       : dict[str, list[str]]  {endog_feat: [instrument_cols]}
    buffer_radius     : int          (selected buffer radius in metres)
    """
    st.sidebar.title("⚙️ Model Controls")

    # ── Model selection ────────────────────────────────────────────────────
    st.sidebar.subheader("Model")
    model_spec = st.sidebar.radio(
        "Specification",
        options=["log-log", "semi-log", "linear"],
        index=0,
        horizontal=True,
        key="model_spec",
    )

    # ── Buffer radius ─────────────────────────────────────────────────────
    buffer_radius = 500  # default
    if radii_list:
        st.sidebar.subheader("Buffer Radius")
        default_idx = radii_list.index(500) if 500 in radii_list else 0
        buffer_radius = st.sidebar.select_slider(
            "Radius (m)",
            options=radii_list,
            value=radii_list[default_idx],
            key="buffer_radius",
            help="Buffer radius around each station for POI/walkability/zone aggregation",
        )
        st.sidebar.caption(f"Buffer: {buffer_radius} m")

    # ── Station filter ────────────────────────────────────────────────────
    st.sidebar.subheader("Station Filter")
    all_types      = sorted(df["station_type"].unique())
    all_colors     = sorted(df["line_color"].unique())
    all_typologies = sorted(df["station_typology"].dropna().unique())

    sel_types = st.sidebar.multiselect(
        "System type", options=all_types, default=all_types, key="filter_type",
    )
    sel_colors = st.sidebar.multiselect(
        "Line colour", options=all_colors, default=all_colors, key="filter_color",
    )
    sel_typologies = st.sidebar.multiselect(
        "Station typology", options=all_typologies, default=all_typologies, key="filter_typology",
    )

    filtered_stations = df[
        df["station_type"].isin(sel_types)
        & df["line_color"].isin(sel_colors)
        & df["station_typology"].isin(sel_typologies)
    ]["display_name"].tolist()

    sel_stations = st.sidebar.multiselect(
        "Stations",
        options=filtered_stations,
        default=filtered_stations,
        key="filter_stations",
    )
    st.sidebar.caption(f"{len(sel_stations)} / {len(df)} stations selected")

    # ── Features ──────────────────────────────────────────────────────────
    st.sidebar.subheader("Features")
    available = [
        c for c in df.select_dtypes(include="number").columns
        if c not in NON_FEATURE_COLS
    ]
    selected_features = [
        feat for feat in available
        if st.sidebar.checkbox(feat, value=(feat in DEFAULT_FEATURES), key=f"feat_{feat}")
    ]

    opt_mode = st.sidebar.radio(
        "Optimization goal",
        options=["Best R²", "Most significant vars"],
        index=0, horizontal=True, key="opt_mode",
    )
    st.sidebar.button(
        "🎯 Optimize Variables", key="btn_optimize",
        help="Forward stepwise: find the best feature combination",
        on_click=lambda: st.session_state.update({"_run_optimize": True}),
    )

    st.sidebar.subheader("Parameters")
    sig_level  = float(st.sidebar.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.20, step=0.01))
    log_offset = float(st.sidebar.number_input(
        "Log offset",
        value=1.0, min_value=0.0, step=0.5,
        disabled=(model_spec == "linear"),
        help="Only used in log-log specification",
    ))

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

    return sel_stations, selected_features, log_offset, sig_level, model_spec, endog_features, instruments, buffer_radius


def run_model(
    df: pd.DataFrame,
    selected_features: list[str],
    endog_features: list[str],
    instruments: dict[str, list[str]],
    log_offset: float,
    sig_level: float,
    model_spec: str = "log-log",
) -> tuple[ModelResult, pd.DataFrame]:
    """Choose OLS/IV and log-log/semi-log/linear based on sidebar state and refit."""
    if model_spec == "linear":
        return fit_linear_model(df, selected_features, sig_level)
    if model_spec == "semi-log":
        return fit_semi_log_model(df, selected_features, log_offset, sig_level)
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
    model_spec = model_result.model_spec

    # ── Equation banner ───────────────────────────────────────────────────
    st.subheader("Fitted Model Equation")
    const_row = coef_df[coef_df["variable"] == "const"]
    if const_row.empty:
        st.error("Internal error: model missing constant term.")
        return
    intercept = float(const_row["coef"].values[0])

    if model_spec == "linear":
        lhs = "ridership"
        def term_str(row):
            return f"<span style='color:#94a3b8'> · {row['variable']}</span>"
    elif model_spec == "semi-log":
        lhs = "ridership"
        def term_str(row):
            feat_raw = row["variable"][4:]  # strip leading 'log_'
            return f"<span style='color:#94a3b8'> · log({feat_raw} + {log_offset})</span>"
    else:  # log-log
        lhs = "log(ridership)"
        def term_str(row):
            feat_raw = row["variable"][4:]  # strip leading 'log_'
            return f"<span style='color:#94a3b8'> · log({feat_raw} + {log_offset})</span>"

    html_parts = [
        f"<div style='line-height:2'><span style='color:#f1f5f9;font-size:14px'>"
        f"{lhs} = <span style='color:#fbbf24'>{intercept:+.4f}</span></span></div>"
    ]
    for _, row in coef_df[coef_df["variable"] != "const"].iterrows():
        sign = "+" if row["coef"] >= 0 else "−"
        coef_color = (
            "#4ade80" if row["significant"] and row["coef"] > 0
            else ("#f87171" if row["significant"] and row["coef"] < 0 else "#94a3b8")
        )
        html_parts.append(
            f"<div style='line-height:2;padding-left:16px'>"
            f"<span style='color:#64748b'>{sign} </span>"
            f"<span style='color:{coef_color}'>{abs(row['coef']):.4f}</span>"
            + term_str(row) +
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

    # ── Coefficient table ─────────────────────────────────────────────────
    st.subheader("Coefficient Summary")
    tbl = coef_df[coef_df["variable"] != "const"].copy()
    tbl = tbl[["variable", "coef", "se", "t", "p", "ci_lo", "ci_hi", "significant"]].copy()
    tbl.columns = ["Variable", "Coef", "Std Err", "t-stat", "p-value", "CI low (95%)", "CI high (95%)", "Significant"]

    def _style_pval(val):
        if val < 0.01:
            return "color: #4ade80; font-weight: bold"
        if val < 0.05:
            return "color: #86efac"
        if val < 0.1:
            return "color: #fbbf24"
        return "color: #94a3b8"

    styled = (
        tbl.style
        .format({
            "Coef": "{:.4f}", "Std Err": "{:.4f}", "t-stat": "{:.3f}",
            "p-value": "{:.4f}", "CI low (95%)": "{:.4f}", "CI high (95%)": "{:.4f}",
        })
        .map(_style_pval, subset=["p-value"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        f"R² = {model_result.rsquared:.4f} | Adj R² = {model_result.rsquared_adj:.4f} | "
        f"n = {model_result.nobs} | F = {model_result.fvalue:.2f} | F p-value = {model_result.f_pvalue:.4f}"
    )

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
    station_df = df[["display_name", "source", "entry"]].copy()
    station_df["predicted"] = df.apply(
        lambda row: predict_ridership(coef_df, {f: row[f] for f in selected_features}, log_offset, model_spec),
        axis=1,
    )
    station_df["residual"]    = station_df["entry"] - station_df["predicted"]
    station_df["pct_error_%"] = (station_df["residual"] / station_df["entry"] * 100).round(1)
    station_df[["entry", "predicted", "residual"]] = (
        station_df[["entry", "predicted", "residual"]].round(0).astype("Int64")
    )
    station_df = station_df.rename(columns={
        "display_name": "Station", "source": "Source",
        "entry": "Actual",        "predicted": "Predicted",
        "residual": "Residual",
    })
    st.dataframe(station_df, use_container_width=True, height=420)


def render_station_explorer(df: pd.DataFrame, coef_df: pd.DataFrame,
                             selected_features: list[str], log_offset: float,
                             model_spec: str = "log-log") -> None:
    st.subheader("Station Explorer")
    selected_station = st.selectbox("Select a station", df["display_name"].tolist(), key="explorer_station")
    row = df[df["display_name"] == selected_station].iloc[0]
    feat_data = {f: float(row[f]) for f in selected_features if f in row.index}

    col_info, col_pred = st.columns(2)
    with col_info:
        st.markdown("**Station Info**")
        if "line_code" in row.index:
            st.write(f"Line code: `{row['line_code']}`")
        st.markdown("**Feature Values**")
        st.dataframe(
            pd.DataFrame({"Feature": list(feat_data.keys()), "Value": list(feat_data.values())}),
            use_container_width=True, hide_index=True,
        )

    with col_pred:
        actual    = float(row["entry"])
        predicted = predict_ridership(coef_df, feat_data, log_offset, model_spec)
        residual  = actual - predicted
        pct_err   = residual / actual * 100
        st.markdown("**Prediction**")
        m1, m2 = st.columns(2)
        m1.metric("Actual ridership",    f"{actual:,.0f}")
        m2.metric("Predicted ridership", f"{predicted:,.0f}")
        m3, m4 = st.columns(2)
        m3.metric("Residual", f"{residual:+,.0f}")
        m4.metric("% Error",  f"{pct_err:+.1f}%")

    if model_spec == "linear":
        contrib_caption = "β_i · x_i for this station."
        contrib_xaxis  = "β_i · x_i"
    else:  # log-log or semi-log
        contrib_caption = "β_i · log(x_i + offset) for this station."
        contrib_xaxis  = "β_i · log(x_i + offset)"

    st.subheader("Elasticity Contribution per Feature")
    st.caption(contrib_caption)
    contrib_rows = []
    for feat, val in feat_data.items():
        var_name = feat if model_spec == "linear" else f"log_{feat}"
        coef_row = coef_df[coef_df["variable"] == var_name]
        if coef_row.empty:
            continue
        coef_val = float(coef_row["coef"].values[0])
        contribution = (
            coef_val * float(val)
            if model_spec == "linear"
            else coef_val * float(np.log(max(val + log_offset, 1e-9)))
        )
        contrib_rows.append({"Feature": feat, "Contribution": contribution, "Coef": coef_val})
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
        xaxis_title=contrib_xaxis,
        height=max(250, len(contrib_df) * 38),
        margin=dict(l=0, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_contrib, use_container_width=True)


def render_whatif(df: pd.DataFrame, coef_df: pd.DataFrame,
                  selected_features: list[str], log_offset: float,
                  model_spec: str = "log-log") -> None:
    st.subheader("What-if Simulator")
    st.caption("Choose a base station, adjust sliders, see how predicted ridership changes.")

    base_station = st.selectbox("Base station", df["display_name"].tolist(), key="whatif_station")
    base_row = df[df["display_name"] == base_station].iloc[0]
    base_feat_vals = {f: float(base_row[f]) for f in selected_features if f in base_row.index}
    base_pred = predict_ridership(coef_df, base_feat_vals, log_offset, model_spec)

    # Reset sliders to the new station's original values when the station changes
    if st.session_state.get("_whatif_prev_station") != base_station:
        for feat in selected_features:
            st.session_state[f"slider_{feat}"] = base_feat_vals.get(feat, 0.0)
        st.session_state["_whatif_prev_station"] = base_station

    st.markdown("---")
    st.markdown("**Adjust feature values:**")
    if st.button("↺ Reset to base values", key="whatif_reset"):
        for feat in selected_features:
            st.session_state[f"slider_{feat}"] = base_feat_vals.get(feat, 0.0)

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

    new_pred   = predict_ridership(coef_df, new_vals, log_offset, model_spec)
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
    if model_spec == "log-log":
        st.caption("α_i × Δ%X_i — point elasticity approximation per feature.")
    elif model_spec == "semi-log":
        st.caption("Point elasticity (β_i / ŷ) × Δ%X_i — evaluated at base station values.")
    else:
        st.caption("Point elasticity (β_i × x_i / ŷ) × Δ%X_i — evaluated at base station values.")
    impact_rows = []
    for feat in selected_features:
        base_val = base_feat_vals.get(feat, 0.0)
        new_val  = new_vals.get(feat, 0.0)
        var_name = feat if model_spec == "linear" else f"log_{feat}"
        coef_row = coef_df[coef_df["variable"] == var_name]
        if coef_row.empty:
            continue
        coef_val = float(coef_row["coef"].values[0])
        if model_spec == "log-log":
            elasticity = coef_val                                        # α already is elasticity
        elif model_spec == "semi-log":
            elasticity = coef_val / (base_pred + 1e-9)                  # β / ŷ
        else:  # linear
            elasticity = coef_val * base_val / (base_pred + 1e-9)       # β · x / ŷ
        pct_x = (new_val - base_val) / (base_val + 1e-9) * 100
        impact_rows.append({
            "Feature":               feat,
            "Base value":            base_val,
            "New value":             new_val,
            "Δ% feature":            round(pct_x, 1),
            "Elasticity":            round(elasticity, 4),
            "Expected Δ% ridership": round(elasticity_impact(elasticity, pct_x), 2),
        })
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)


def render_buffer_sensitivity(
    df_radii: pd.DataFrame,
    selected_features: list[str],
    sel_stations: list[str],
    log_offset: float,
    sig_level: float,
    model_spec: str,
    endog_features: list[str],
    instruments: dict[str, list[str]],
) -> None:
    """Tab 4: show how coefficients change across buffer radii."""
    st.subheader("Sensitivity to Buffer Radius")
    st.caption(
        "Each point shows the elasticity coefficient estimated at that buffer radius. "
        "Shaded bands are 95% confidence intervals. Solid dots = significant (p < α)."
    )

    radii = sorted(df_radii["radius_m"].unique().astype(int))
    rows = []
    for r in radii:
        df_r = df_radii[
            (df_radii["radius_m"] == r) & df_radii["display_name"].isin(sel_stations)
        ].copy()
        if df_r.empty or len(df_r) < 3:
            continue
        try:
            mr, cdf = run_model(df_r, selected_features, endog_features, instruments,
                                log_offset, sig_level, model_spec)
            for _, row in cdf[cdf["variable"] != "const"].iterrows():
                rows.append({
                    "radius_m": r,
                    "variable": row["variable"],
                    "coef": row["coef"],
                    "ci_lo": row["ci_lo"],
                    "ci_hi": row["ci_hi"],
                    "p": row["p"],
                    "significant": row["significant"],
                })
            rows.append({
                "radius_m": r,
                "variable": "__r2__",
                "coef": mr.rsquared,
                "ci_lo": mr.rsquared_adj,
                "ci_hi": mr.nobs,
                "p": 0,
                "significant": True,
            })
        except Exception:
            continue

    if not rows:
        st.warning("Could not fit models across radii with the current feature set.")
        return

    sens_df = pd.DataFrame(rows)

    # ── R² trend ──────────────────────────────────────────────────────────
    r2_df = sens_df[sens_df["variable"] == "__r2__"].copy()
    sens_df = sens_df[sens_df["variable"] != "__r2__"]

    if not r2_df.empty:
        st.subheader("Model Fit across Radii")
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Scatter(
            x=r2_df["radius_m"], y=r2_df["coef"],
            mode="lines+markers", name="R²",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=8),
        ))
        fig_r2.add_trace(go.Scatter(
            x=r2_df["radius_m"], y=r2_df["ci_lo"],
            mode="lines+markers", name="Adj R²",
            line=dict(color="#a78bfa", width=2, dash="dash"),
            marker=dict(size=6),
        ))
        fig_r2.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
            xaxis_title="Buffer Radius (m)", yaxis_title="R²",
            height=350, margin=dict(l=0, r=20, t=10, b=40),
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # ── Per-feature coefficient sensitivity ───────────────────────────────
    st.subheader("Coefficient Sensitivity per Feature")
    features_in_data = sorted(sens_df["variable"].unique())

    # Let user pick which features to display (default: all)
    show_features = st.multiselect(
        "Features to display",
        options=features_in_data,
        default=features_in_data,
        key="sens_features",
    )
    plot_df = sens_df[sens_df["variable"].isin(show_features)]

    if plot_df.empty:
        st.info("Select at least one feature above.")
        return

    colors = [
        "#38bdf8", "#4ade80", "#f87171", "#fbbf24", "#a78bfa",
        "#fb923c", "#2dd4bf", "#e879f9", "#94a3b8", "#34d399",
        "#f472b6", "#818cf8", "#facc15", "#22d3ee", "#c084fc",
    ]

    def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
        """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure()
    for i, feat in enumerate(show_features):
        fdf = plot_df[plot_df["variable"] == feat].sort_values("radius_m")
        color = colors[i % len(colors)]
        # CI band
        fig.add_trace(go.Scatter(
            x=list(fdf["radius_m"]) + list(fdf["radius_m"][::-1]),
            y=list(fdf["ci_hi"]) + list(fdf["ci_lo"][::-1]),
            fill="toself", fillcolor=_hex_to_rgba(color, 0.1),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # Line
        fig.add_trace(go.Scatter(
            x=fdf["radius_m"], y=fdf["coef"],
            mode="lines+markers", name=feat,
            line=dict(color=color, width=2),
            marker=dict(
                size=[10 if s else 6 for s in fdf["significant"]],
                symbol=["circle" if s else "circle-open" for s in fdf["significant"]],
                color=color,
            ),
            hovertemplate=f"<b>{feat}</b><br>Radius: %{{x}}m<br>Coef: %{{y:.4f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#475569", line_width=1)
    fig.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
        xaxis_title="Buffer Radius (m)", yaxis_title="Elasticity Coefficient",
        height=500, margin=dict(l=0, r=20, t=10, b=40),
        legend=dict(x=1.02, y=1, bordercolor="#334155", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────
    st.subheader("Coefficient Table by Radius")
    pivot = sens_df.pivot_table(index="variable", columns="radius_m", values="coef")
    pivot.columns = [f"{int(c)}m" for c in pivot.columns]
    st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Bangkok Ridership Elasticity", layout="wide", page_icon="🚉")

    # Load multi-radius data if available, else fall back to single-radius rev14
    has_radii = os.path.exists(RADII_PATH)
    if has_radii:
        df_radii = load_radii_data()
        radii_list = get_radii_list(df_radii)
    else:
        df_radii = None
        radii_list = None

    df_all = load_data()

    # ── Pre-sidebar optimization pass ─────────────────────────────────────
    # Must update checkbox session_state BEFORE build_sidebar renders them.
    if st.session_state.pop("_run_optimize", False):
        # Reconstruct the data slice using sidebar state from the previous run
        _df_src = df_all
        if has_radii and df_radii is not None:
            _buf_r = st.session_state.get("buffer_radius", 500)
            _df_src = df_radii[df_radii["radius_m"] == _buf_r].copy()
            _df_src = _df_src.dropna(subset=["entry"])
            _df_src["line_color"]   = _df_src["line_code"].apply(lambda c: _classify_line(c)[0])
            _df_src["station_type"] = _df_src["line_code"].apply(lambda c: _classify_line(c)[1])
            _df_src["display_name"] = _df_src["line_code"] + " — " + _df_src["station_name"]

        # Use previous station selection from session state
        _prev_stations = st.session_state.get("filter_stations")
        if _prev_stations:
            _df_src = _df_src[_df_src["display_name"].isin(_prev_stations)]

        if not _df_src.empty:
            available_feats = [
                c for c in _df_src.select_dtypes(include="number").columns
                if c not in NON_FEATURE_COLS
            ]
            _ms = st.session_state.get("model_spec", "log-log")
            _lo = float(st.session_state.get("log_offset", 1.0))
            _sl = float(st.session_state.get("sig_level", 0.05))
            _om = "max_signif" if st.session_state.get("opt_mode") == "Most significant vars" else "best_r2"
            opt_feats, opt_adj_r2, opt_n_sig = _forward_stepwise(
                _df_src, available_feats, _ms, _lo, _sl, mode=_om,
            )
            for feat in available_feats:
                st.session_state[f"feat_{feat}"] = feat in opt_feats
            st.session_state["_opt_result"] = (opt_feats, opt_adj_r2, opt_n_sig, _om)

    sel_stations, selected_features, log_offset, sig_level, model_spec, endog_features, instruments, buffer_radius = build_sidebar(
        df_all, radii_list
    )

    # Use radius-specific data when available
    if has_radii and df_radii is not None:
        df_all = df_radii[df_radii["radius_m"] == buffer_radius].copy()

    st.title("🚉 Bangkok Ridership Elasticity Dashboard")

    # Show optimization result banner
    opt_result = st.session_state.pop("_opt_result", None)
    if opt_result:
        opt_feats, opt_adj_r2, opt_n_sig, opt_mode_used = opt_result
        mode_label = "Best R²" if opt_mode_used == "best_r2" else "Most significant vars"
        st.success(
            f"**Optimized ({mode_label}):** {len(opt_feats)} variables selected · "
            f"Adj R² = {opt_adj_r2:.4f} · "
            f"{opt_n_sig} significant (p < α) · "
            f"Features: {', '.join(opt_feats)}"
        )

    if not selected_features:
        st.warning("Select at least one feature in the sidebar.")
        return

    df = df_all[df_all["display_name"].isin(sel_stations)].copy()
    if df.empty:
        st.warning("No stations selected. Adjust the station filter in the sidebar.")
        return

    model_result, coef_df = run_model(
        df, selected_features, endog_features, instruments, log_offset, sig_level, model_spec
    )

    # Model type / spec badges
    badge_color = "#38bdf8" if model_result.model_type == "OLS" else "#a78bfa"
    spec_color  = "#34d399" if model_spec == "log-log" else ("#a78bfa" if model_spec == "semi-log" else "#fb923c")
    radius_badge = (
        f"&nbsp;<span style='background:#fbbf24;color:#0f172a;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:700'>{buffer_radius}m buffer</span>"
        if has_radii else ""
    )
    st.markdown(
        f"<span style='background:{badge_color};color:#0f172a;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:700'>{model_result.model_type}</span>"
        f"&nbsp;"
        f"<span style='background:{spec_color};color:#0f172a;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:700'>{model_spec}</span>"
        + radius_badge,
        unsafe_allow_html=True,
    )

    tabs = ["Model Results", "Station Explorer", "What-if Simulator"]
    if has_radii:
        tabs.append("Buffer Sensitivity")
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        render_model_results(model_result, coef_df, df, selected_features, log_offset)
    with tab_objs[1]:
        render_station_explorer(df, coef_df, selected_features, log_offset, model_spec)
    with tab_objs[2]:
        render_whatif(df, coef_df, selected_features, log_offset, model_spec)
    if has_radii and len(tab_objs) > 3:
        with tab_objs[3]:
            render_buffer_sensitivity(
                df_radii, selected_features, sel_stations,
                log_offset, sig_level, model_spec,
                endog_features, instruments,
            )


if __name__ == "__main__":
    main()
