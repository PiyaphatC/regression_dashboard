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
    fit_iv_model,
    fit_linear_model,
    fit_model,
    fit_semi_log_model,
    predict_ridership,
)

DATA_PATH = "Output/combined_station_summary_expanded_rev16.csv"
RADII_PATH = "Output/features_by_radius.csv"
SW_DEV_PATH = "Output/sw_dev.xlsx"

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
    # Last-mile feeder counts
    "bus_stop_count",
    "win_count",
    "taxi_count",
    "park_ride_car_count",
    "bike_parking_count",
    "bike_share_count",
    # Last-mile feeder distances
    "bike_parking_mean_dist",
    "bike_share_mean_dist",
    "bus_stop_mean_dist",
    "taxi_mean_dist",
    "win_mean_dist",
    # Sidewalk
    "sw_total_length",
    "sidewalk_length_surface_neg1",
    "sidewalk_length_shade_neg1",
    "sidewalk_length_obstacle_neg1",
    # Socio-economic
    "POP25",
    "PRIM25",
    "STU25",
]


@st.cache_data(ttl=300)
def load_sw_dev(path: str = SW_DEV_PATH) -> dict[str, float]:
    """Load proposed sidewalk additions: {line_code: delta_meters}."""
    try:
        sw = pd.read_excel(path)
        sw["delta_sw"] = sw["ความยาวทางเท้าที่เสนอเพิ่ม"].str.extract(r"([+-]?\d+)").astype(float)
        return dict(zip(sw["รหัสสถานี"], sw["delta_sw"]))
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    df = df.dropna(subset=["entry"])
    classified = df["line_code"].apply(_classify_line)
    df["line_color"]   = classified.apply(lambda x: x[0])
    df["station_type"] = classified.apply(lambda x: x[1])
    df["display_name"] = df["line_code"] + " — " + df["station_name"]
    return df


@st.cache_data(ttl=300)
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


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Expected coefficient signs for "correct signs" mode.
# +1 = positive expected, -1 = negative expected, None = no constraint.
def _expected_sign(feat: str) -> int | None:
    f = feat.lower()
    # Distance → negative (farther = less ridership)
    if f.endswith("_min_dist") or f.endswith("_mean_dist"):
        return -1
    # Count → positive (more feeders/POIs = more ridership)
    if f.endswith("_count"):
        return 1
    # Sidewalk quality pct: good → positive, bad → negative
    if f.endswith("_pct_1"):
        return 1
    if f.endswith("_pct_neg1"):
        return -1
    # Sidewalk quality length: good → positive, bad → negative
    if "length_surface_1" in f or "length_shade_1" in f or "length_obstacle_1" in f:
        return 1
    if "length_surface_neg1" in f or "length_shade_neg1" in f or "length_obstacle_neg1" in f:
        return -1
    # General walkability → positive
    if f in ("sidewalk_length", "sidewalk_length_1", "sidewalk_length_2",
             "sw_total_length", "sw_width_mean",
             "road_length_gt4m", "road_length_gt4m_atleast1sw",
             "road_width_mean", "road_width_min", "road_width_max",
             "n_segments"):
        return 1
    # Socio-economic → positive
    if any(f.startswith(p) for p in ("pop", "prim", "sec", "ter", "stu", "tour", "comar", "tpl")):
        return 1
    return None


def _compute_vif(df: pd.DataFrame, features: list[str],
                 log_offset: float, model_spec: str) -> float:
    """Return the max VIF across features. Returns inf on failure."""
    try:
        if model_spec == "linear":
            X = df[features].copy()
        else:
            X = df[features].apply(lambda s: np.log(s + log_offset))
        X = X.dropna()
        if X.shape[0] < X.shape[1] + 1:
            return float("inf")
        vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return max(vifs)
    except Exception:
        return float("inf")


def _forward_stepwise(df: pd.DataFrame, candidates: list[str],
                      model_spec: str, log_offset: float, sig_level: float,
                      mode: str = "best_r2",
                      max_features: int = 15) -> tuple[list[str], float, int]:
    """Forward stepwise feature selection.

    Modes:
      best_r2       — maximise Adj R²
      max_signif    — maximise number of significant variables, ties → Adj R²
      all_signif    — largest subset where ALL variables are significant (p < α)
      no_vif        — maximise Adj R² with VIF < 10 constraint
      correct_signs — maximise Adj R² with correct coefficient sign constraint
      parsimonious  — fewest variables reaching ≥ 95% of best achievable Adj R²

    Returns (best_features, best_adj_r2, n_significant).
    """
    # For parsimonious mode: first find the ceiling Adj R²
    if mode == "parsimonious":
        ceiling_feats, ceiling_r2, _ = _forward_stepwise(
            df, candidates, model_spec, log_offset, sig_level, mode="best_r2",
            max_features=max_features,
        )
        target_r2 = ceiling_r2 * 0.95
        # Now find smallest subset that meets the target
        # Try sizes 1, 2, 3, ... using forward stepwise with early stop
        best_parsi = None
        for size_limit in range(1, len(ceiling_feats) + 1):
            sel, adj_r2, n_sig = _forward_stepwise(
                df, candidates, model_spec, log_offset, sig_level,
                mode="best_r2", max_features=size_limit,
            )
            if adj_r2 >= target_r2:
                best_parsi = (sel, adj_r2, n_sig)
                break
        if best_parsi:
            return best_parsi
        return ceiling_feats, ceiling_r2, len(ceiling_feats)

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
            non_const = cdf[cdf["variable"] != "const"]
            n_sig = int(non_const["significant"].sum())
            adj_r2 = mr.rsquared_adj

            # ── Mode-specific acceptance logic ────────────────────────
            if mode == "all_signif":
                if n_sig < len(trial):
                    continue  # not all significant → skip
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_n_sig = n_sig
                    best_feat = feat
                    improved = True

            elif mode == "no_vif":
                max_vif = _compute_vif(df, trial, log_offset, model_spec)
                if max_vif > 10:
                    continue  # VIF too high → skip
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_n_sig = n_sig
                    best_feat = feat
                    improved = True

            elif mode == "correct_signs":
                sign_ok = True
                for _, row in non_const.iterrows():
                    raw_feat = row["variable"].removeprefix("log_")
                    expected = _expected_sign(raw_feat)
                    if expected is not None and np.sign(row["coef"]) != expected:
                        sign_ok = False
                        break
                if not sign_ok:
                    continue
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_n_sig = n_sig
                    best_feat = feat
                    improved = True

            elif mode == "max_signif":
                if (n_sig > best_n_sig) or (n_sig == best_n_sig and adj_r2 > best_adj_r2):
                    best_n_sig = n_sig
                    best_adj_r2 = adj_r2
                    best_feat = feat
                    improved = True

            else:  # best_r2
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_n_sig = n_sig
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

    _EXCLUDE_DEFAULT = {"RN01 — กรุงเทพอภิวัฒน์", "RW01 — กรุงเทพอภิวัฒน์"}
    filtered_stations = df[
        df["station_type"].isin(sel_types)
        & df["line_color"].isin(sel_colors)
        & df["station_typology"].isin(sel_typologies)
    ]["display_name"].tolist()
    default_stations = [s for s in filtered_stations if s not in _EXCLUDE_DEFAULT]

    sel_stations = st.sidebar.multiselect(
        "Stations",
        options=filtered_stations,
        default=default_stations,
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

    opt_mode = st.sidebar.selectbox(
        "Optimization goal",
        options=[
            "Best R²",
            "Most significant vars",
            "All vars significant",
            "No multicollinearity (VIF<10)",
            "Correct signs only",
            "Parsimonious (fewest vars)",
        ],
        index=0, key="opt_mode",
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
    _THAI_LABELS = {
        "const": "ค่าคงที่",
        "bus_stop_count": "จำนวนป้ายรถประจำทาง",
        "win_count": "จำนวนวินมอเตอร์ไซค์รับจ้าง",
        "taxi_count": "จำนวนจุดรถแท็กซี่",
        "park_ride_car_count": "จำนวนที่จอดแล้วจร (รถยนต์)",
        "bike_parking_count": "จำนวนที่จอดจักรยาน",
        "bike_share_count": "จำนวนจุดจักรยานสาธารณะ",
        "bike_parking_mean_dist": "ระยะทางเฉลี่ยถึงที่จอดจักรยาน",
        "bike_share_mean_dist": "ระยะทางเฉลี่ยถึงจุดจักรยานสาธารณะ",
        "bus_stop_mean_dist": "ระยะทางเฉลี่ยถึงป้ายรถประจำทาง",
        "taxi_mean_dist": "ระยะทางเฉลี่ยถึงจุดรถแท็กซี่",
        "win_mean_dist": "ระยะทางเฉลี่ยถึงวินมอเตอร์ไซค์รับจ้าง",
        "sw_total_length": "ความยาวรวมของทางเท้า",
        "sidewalk_length_surface_neg1": "ความยาวทางเท้าที่พื้นผิวไม่ดี",
        "sidewalk_length_shade_neg1": "ความยาวทางเท้าที่ไม่มีร่มเงา",
        "sidewalk_length_obstacle_neg1": "ความยาวทางเท้าที่มีสิ่งกีดขวาง",
        "POP25": "จำนวนประชากร (ปี 2025)",
        "PRIM25": "การสร้างการเดินทางจากงานหลัก (Primary job, 2025)",
        "STU25": "จำนวนนักเรียน/นักศึกษา (ปี 2025)",
    }
    const_tbl = coef_df[coef_df["variable"] == "const"].copy()
    feat_tbl = coef_df[coef_df["variable"] != "const"].copy()
    tbl = pd.concat([const_tbl, feat_tbl], ignore_index=True)
    tbl = tbl[["variable", "coef", "se", "t", "p", "ci_lo", "ci_hi", "significant"]].copy()
    tbl.insert(1, "thai", tbl["variable"].map(_THAI_LABELS).fillna(""))
    tbl.columns = ["Variable", "ตัวแปร (ภาษาไทย)", "Coef", "Std Err", "t-stat", "p-value", "CI low (95%)", "CI high (95%)", "Significant"]

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
        st.caption("Δ%y = (exp(β × Δlog(x+c)) − 1) × 100 per feature.")
    elif model_spec == "semi-log":
        st.caption("Δ%y = β × Δlog(x+c) / y × 100 per feature.")
    else:
        st.caption("Δ%y = β × Δx / y × 100 per feature.")
    impact_rows = []
    for feat in selected_features:
        base_val = base_feat_vals.get(feat, 0.0)
        new_val  = new_vals.get(feat, 0.0)
        var_name = feat if model_spec == "linear" else f"log_{feat}"
        coef_row = coef_df[coef_df["variable"] == var_name]
        if coef_row.empty:
            continue
        coef_val = float(coef_row["coef"].values[0])
        delta_pct_y = _approx_delta_pct_y(coef_val, base_val, new_val, base_pred, log_offset, model_spec)
        impact_rows.append({
            "Feature":               feat,
            "Base value":            base_val,
            "New value":             new_val,
            "β":                     round(coef_val, 4),
            "Δ% ridership":          round(delta_pct_y, 2),
        })
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)


# ── Walkability variable helpers for policy tabs ─────────────────────────

# Mapping: dimension → list of possible neg1 variable names
_WALK_NEG1_VARS: dict[str, list[str]] = {
    "Surface":  ["surface_pct_neg1", "sidewalk_length_surface_neg1"],
    "Shade":    ["shade_pct_neg1", "sidewalk_length_shade_neg1"],
    "Obstacle": ["obs_pct_neg1", "sidewalk_length_obstacle_neg1"],
}

# Corresponding _0 variables (receive the shifted amount when neg1 → 0)
_WALK_0_VARS: dict[str, list[str]] = {
    "Surface":  ["surface_pct_0", "sidewalk_length_surface_0"],
    "Shade":    ["shade_pct_0", "sidewalk_length_shade_0"],
    "Obstacle": ["obs_pct_0", "sidewalk_length_obstacle_0"],
}


def _find_walk_vars(selected_features: list[str]) -> dict[str, list[str]]:
    """Return {dimension: [matched neg1 features]} for selected walkability vars."""
    found: dict[str, list[str]] = {}
    for dim, candidates in _WALK_NEG1_VARS.items():
        matched = [v for v in candidates if v in selected_features]
        if matched:
            found[dim] = matched
    return found


def _find_walk0_vars(selected_features: list[str]) -> dict[str, list[str]]:
    """Return {dimension: [matched _0 features]} for selected walkability vars."""
    found: dict[str, list[str]] = {}
    for dim, candidates in _WALK_0_VARS.items():
        matched = [v for v in candidates if v in selected_features]
        if matched:
            found[dim] = matched
    return found


def _build_scenario_vals(
    base_vals: dict[str, float],
    neg1_vars: list[str],
    zero_vars: list[str],
    df_row: pd.Series,
) -> dict[str, float]:
    """Build a copy of base_vals with neg1 vars set to 0 and _0 vars increased.

    Each neg1 var is paired with its corresponding _0 var by matching the
    dimension prefix (e.g. sidewalk_length_surface_neg1 → sidewalk_length_surface_0).
    """
    new = dict(base_vals)
    for v in neg1_vars:
        old_amount = new.get(v, 0.0)
        new[v] = 0.0
        # Find the matching _0 var by replacing the suffix
        corresponding_zero = v.replace("_neg1", "_0")
        if corresponding_zero in zero_vars and corresponding_zero in new:
            new[corresponding_zero] = new[corresponding_zero] + old_amount
    return new


def _delta_log_y(
    coef_val: float, base_val: float, new_val: float,
    log_offset: float,
) -> float:
    """Return Δlog(y) = β × Δlog(x+c) for a single feature change (log-log only).

    These values are additive in log space and can be summed across features
    before converting to Δ%y via (exp(Σ Δlog(y)) − 1) × 100.
    """
    delta_log_x = np.log(new_val + log_offset) - np.log(base_val + log_offset)
    return coef_val * delta_log_x


def _approx_delta_pct_y(
    coef_val: float, base_val: float, new_val: float,
    base_pred: float, log_offset: float, model_spec: str,
) -> float:
    """Return Δ%y for a single feature change.

    Formulas per model specification:
      log-log : log(y) = α + β·log(x+c)  →  Δ%y = (exp(β × Δlog(x+c)) − 1) × 100
      semi-log: y = α + β·log(x+c)       →  Δ%y = β × Δlog(x+c) / y × 100
      linear  : y = α + β·x              →  Δ%y = β × Δx / y × 100
    where Δlog(x+c) = ln(x_new + c) − ln(x_old + c)
    """
    if model_spec == "log-log":
        return (np.exp(_delta_log_y(coef_val, base_val, new_val, log_offset)) - 1) * 100
    elif model_spec == "semi-log":
        delta_log_x = np.log(new_val + log_offset) - np.log(base_val + log_offset)
        return coef_val * delta_log_x / (base_pred + 1e-9) * 100
    else:  # linear
        delta_x = new_val - base_val
        return coef_val * delta_x / (base_pred + 1e-9) * 100


def _policy_sort_widget(key_suffix: str) -> str:
    """Render a sort-by selectbox and return the chosen column."""
    return st.selectbox(
        "Sort / group by",
        ["Line Color", "Station Typology"],
        key=f"policy_sort_{key_suffix}",
    )


def _policy_download(result_df: pd.DataFrame, key_suffix: str) -> None:
    """Render a CSV download button."""
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"policy_recommendation_{key_suffix}.csv",
        mime="text/csv",
        key=f"policy_dl_{key_suffix}",
    )


# ── Tab 4: Policy Recommendation (Exact / What-if) ──────────────────────

def _show_exact_calculation(
    station_row: pd.Series, coef_df: pd.DataFrame,
    selected_features: list[str], walk_neg1: dict, walk_zero: dict,
    log_offset: float, model_spec: str,
    sw_dev_map: dict[str, float] | None = None,
) -> None:
    """Show detailed exact calculation for a selected station (Surface, Shade, Proposed SW only)."""
    c = log_offset
    base_vals = {f: float(station_row[f]) for f in selected_features if f in station_row.index}
    base_pred = predict_ridership(coef_df, base_vals, log_offset, model_spec)
    st.markdown(f"**Base predicted ridership = {base_pred:,.1f}**")

    surface_neg1_list = walk_neg1.get("Surface", [])
    shade_neg1_list = walk_neg1.get("Shade", [])
    surface_zero_list = walk_zero.get("Surface", [])
    shade_zero_list = walk_zero.get("Shade", [])

    dim_labels = {"Surface": "Upgrade Surface", "Shade": "Upgrade Shade"}
    for dim, label in dim_labels.items():
        neg1_list = walk_neg1.get(dim, [])
        zero_list = walk_zero.get(dim, [])
        if not neg1_list:
            continue
        feat = neg1_list[0]
        x_old = base_vals.get(feat, 0.0)
        var_name = feat if model_spec == "linear" else f"log_{feat}"
        cr = coef_df[coef_df["variable"] == var_name]
        if cr.empty:
            continue
        beta = float(cr["coef"].values[0])
        scenario = _build_scenario_vals(base_vals, neg1_list, zero_list, station_row)
        new_pred = predict_ridership(coef_df, scenario, log_offset, model_spec)
        chg = new_pred - base_pred

        if model_spec == "log-log":
            log_old = np.log(x_old + c)
            delta_log = beta * (0 - log_old)
            st.markdown(
                f"**{label}:** `{feat}` = {x_old:.1f} → 0  \n"
                f"β = {beta:+.4f}  \n"
                f"log({x_old:.1f}+{c}) = {log_old:.4f} → log(0+{c}) = 0  \n"
                f"Δlog(y) = {beta:.4f} × (0 − {log_old:.4f}) = {delta_log:+.4f}  \n"
                f"y\_new = {new_pred:,.0f}, **Δ riders = {chg:+,.0f}**"
            )
        elif model_spec == "semi-log":
            log_old = np.log(x_old + c)
            delta_y = beta * (0 - log_old)
            st.markdown(
                f"**{label}:** `{feat}` = {x_old:.1f} → 0  \n"
                f"β = {beta:+.4f}  \n"
                f"Δy = {beta:.4f} × (log(0+{c}) − log({x_old:.1f}+{c})) = {delta_y:+.1f}  \n"
                f"y\_new = {new_pred:,.0f}, **Δ riders = {chg:+,.0f}**"
            )
        else:
            delta_x = 0 - x_old
            st.markdown(
                f"**{label}:** `{feat}` = {x_old:.1f} → 0  \n"
                f"β = {beta:+.4f}  \n"
                f"Δy = β × Δx = {beta:.4f} × {delta_x:.1f} = {chg:+.1f}  \n"
                f"y\_new = {new_pred:,.0f}, **Δ riders = {chg:+,.0f}**"
            )

    # Add Proposed SW
    has_sw_dev = sw_dev_map and "sw_total_length" in selected_features
    line_code = station_row.get("line_code", "")
    delta_sw = sw_dev_map.get(line_code, 0.0) if has_sw_dev else 0.0
    if has_sw_dev and delta_sw > 0:
        sw_base = base_vals.get("sw_total_length", 0.0)
        sw_sc = dict(base_vals)
        sw_sc["sw_total_length"] = sw_base + delta_sw
        sw_pred = predict_ridership(coef_df, sw_sc, log_offset, model_spec)
        sw_chg = sw_pred - base_pred
        st.markdown(
            f"**Add Proposed SW:** `sw_total_length` = {sw_base:.1f} → {sw_base + delta_sw:.1f} (+{delta_sw:.0f} m)  \n"
            f"y\_new = {sw_pred:,.0f}, **Δ riders = {sw_chg:+,.0f}**"
        )

    # All Upgrades Combined
    combined_neg1 = surface_neg1_list + shade_neg1_list
    combined_zero = surface_zero_list + shade_zero_list
    combined_sc = _build_scenario_vals(base_vals, combined_neg1, combined_zero, station_row)
    if has_sw_dev and delta_sw > 0:
        combined_sc["sw_total_length"] = combined_sc.get(
            "sw_total_length", base_vals.get("sw_total_length", 0.0)
        ) + delta_sw
    combined_pred = predict_ridership(coef_df, combined_sc, log_offset, model_spec)
    combined_chg = combined_pred - base_pred
    if model_spec == "log-log":
        parts = []
        total_dl = 0.0
        for feat in combined_neg1:
            x_old = base_vals.get(feat, 0.0)
            var_name = f"log_{feat}"
            cr = coef_df[coef_df["variable"] == var_name]
            if cr.empty or x_old == 0:
                continue
            beta = float(cr["coef"].values[0])
            dl = beta * (0 - np.log(x_old + c))
            total_dl += dl
            short = feat.replace("sidewalk_length_", "").replace("_neg1", "")
            parts.append(f"{short}: Δlog = {dl:+.4f}")
        if has_sw_dev and delta_sw > 0:
            sw_base = base_vals.get("sw_total_length", 0.0)
            var_name = "log_sw_total_length"
            cr = coef_df[coef_df["variable"] == var_name]
            if not cr.empty:
                beta = float(cr["coef"].values[0])
                dl = beta * (np.log(sw_base + delta_sw + c) - np.log(sw_base + c))
                total_dl += dl
                parts.append(f"proposed_sw: Δlog = {dl:+.4f}")
        st.markdown(
            f"**All Upgrades Combined:** surface + shade neg1 → 0, plus proposed SW  \n"
            + "  \n".join(parts) + "  \n"
            f"Σ Δlog(y) = {total_dl:+.4f}  \n"
            f"y\_new = {base_pred:,.0f} × exp({total_dl:.4f}) = {combined_pred:,.0f}  \n"
            f"**Δ riders = {combined_chg:+,.0f}**"
        )
    else:
        st.markdown(f"**All Upgrades Combined:** surface + shade neg1 → 0, plus proposed SW → **Δ riders = {combined_chg:+,.0f}**")


def render_policy_exact(
    df: pd.DataFrame, coef_df: pd.DataFrame,
    selected_features: list[str], log_offset: float,
    model_spec: str = "log-log",
    sw_dev_map: dict[str, float] | None = None,
) -> None:
    st.subheader("Policy Recommendation — Exact Prediction")
    st.caption(
        "Scenario: upgrade poor-quality (−1) surface & shade sidewalk to neutral (0), "
        "plus proposed sidewalk development. Combined effect applies all changes simultaneously."
    )

    walk_neg1 = _find_walk_vars(selected_features)
    walk_zero = _find_walk0_vars(selected_features)
    has_sw_dev = sw_dev_map and "sw_total_length" in selected_features

    # Surface / Shade neg1 variable names (if in the model)
    surface_neg1_list = walk_neg1.get("Surface", [])
    shade_neg1_list = walk_neg1.get("Shade", [])
    surface_zero_list = walk_zero.get("Surface", [])
    shade_zero_list = walk_zero.get("Shade", [])

    if not surface_neg1_list and not shade_neg1_list and not has_sw_dev:
        st.warning(
            "No surface/shade neg1 variables or SW development data available. "
            "Please add sidewalk variables to see policy impacts."
        )
        return

    sort_col_label = _policy_sort_widget("exact")

    rows = []
    for _, station_row in df.iterrows():
        base_vals = {f: float(station_row[f]) for f in selected_features if f in station_row.index}
        base_pred = predict_ridership(coef_df, base_vals, log_offset, model_spec)
        actual = float(station_row["entry"])
        line_code = station_row.get("line_code", "")

        # Raw neg1 metre values
        surface_m = float(station_row.get("sidewalk_length_surface_neg1", 0.0))
        shade_m = float(station_row.get("sidewalk_length_shade_neg1", 0.0))
        delta_sw = sw_dev_map.get(line_code, 0.0) if has_sw_dev else 0.0

        record: dict = {
            "Station": station_row.get("station_name", ""),
            "Station ID": line_code,
            "Line Color": station_row.get("line_color", ""),
            "Observed Ridership": round(actual),
            "Predicted Ridership": round(base_pred),
            "Surface −1 (m)": round(surface_m),
            "Shade −1 (m)": round(shade_m),
            "Proposed SW (m)": round(delta_sw),
        }

        # Upgrade Surface: set surface_neg1 → 0
        if surface_neg1_list:
            sc = _build_scenario_vals(base_vals, surface_neg1_list, surface_zero_list, station_row)
            chg = predict_ridership(coef_df, sc, log_offset, model_spec) - base_pred
            record["Upgrade Surface (riders)"] = round(chg)
        else:
            record["Upgrade Surface (riders)"] = 0

        # Upgrade Shade: set shade_neg1 → 0
        if shade_neg1_list:
            sc = _build_scenario_vals(base_vals, shade_neg1_list, shade_zero_list, station_row)
            chg = predict_ridership(coef_df, sc, log_offset, model_spec) - base_pred
            record["Upgrade Shade (riders)"] = round(chg)
        else:
            record["Upgrade Shade (riders)"] = 0

        # Add Proposed SW: add proposed sidewalk length
        sw_chg = 0
        if has_sw_dev and delta_sw > 0:
            sw_sc = dict(base_vals)
            sw_sc["sw_total_length"] = base_vals.get("sw_total_length", 0.0) + delta_sw
            sw_chg = round(predict_ridership(coef_df, sw_sc, log_offset, model_spec) - base_pred)
        record["Add Proposed SW (riders)"] = sw_chg

        # Combined: surface_neg1→0 + shade_neg1→0 + proposed SW, all at once
        combined_neg1 = surface_neg1_list + shade_neg1_list
        combined_zero = surface_zero_list + shade_zero_list
        combined_sc = _build_scenario_vals(base_vals, combined_neg1, combined_zero, station_row)
        if has_sw_dev and delta_sw > 0:
            combined_sc["sw_total_length"] = combined_sc.get(
                "sw_total_length", base_vals.get("sw_total_length", 0.0)
            ) + delta_sw
        combined_chg = predict_ridership(coef_df, combined_sc, log_offset, model_spec) - base_pred
        record["All Upgrades Combined (riders)"] = round(combined_chg)

        rows.append(record)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(sort_col_label)

    st.dataframe(result_df, use_container_width=True, hide_index=True, height=500)

    _policy_download(result_df, "exact")

    # Calculation detail
    st.markdown("---")
    station_names = sorted(df["display_name"].tolist())
    sel = st.selectbox("Show calculation for station:", station_names, key="exact_calc_station")
    sel_row = df[df["display_name"] == sel].iloc[0]
    _show_exact_calculation(sel_row, coef_df, selected_features, walk_neg1, walk_zero, log_offset, model_spec, sw_dev_map)


# ── Tab 5: Policy Recommendation (Elasticity Approximation) ─────────────

def _elasticity_pct(coef_df: pd.DataFrame, feat: str, base_val: float,
                    new_val: float, model_spec: str) -> float:
    """Simple elasticity: Δ%y = β × Δ%x.  Works for any model spec."""
    vn = feat if model_spec == "linear" else f"log_{feat}"
    cr = coef_df[coef_df["variable"] == vn]
    if cr.empty or base_val == 0:
        return 0.0
    beta = float(cr["coef"].values[0])
    pct_x = (new_val - base_val) / base_val * 100
    return beta * pct_x


def render_policy_approx(
    df: pd.DataFrame, coef_df: pd.DataFrame,
    selected_features: list[str], log_offset: float,
    model_spec: str = "log-log",
    sw_dev_map: dict[str, float] | None = None,
) -> None:
    st.subheader("Policy Recommendation — Elasticity")
    st.caption(
        "Elasticity approximation: **Δ%y = β × Δ%x**. "
        "A 1% change in x leads to a β% change in ridership. "
        "Rider changes are applied to **observed ridership**."
    )

    walk_neg1 = _find_walk_vars(selected_features)
    has_sw_dev = sw_dev_map and "sw_total_length" in selected_features

    surface_neg1_list = walk_neg1.get("Surface", [])
    shade_neg1_list = walk_neg1.get("Shade", [])

    if not surface_neg1_list and not shade_neg1_list and not has_sw_dev:
        st.warning(
            "No surface/shade neg1 variables or SW development data available. "
            "Please add sidewalk variables to see policy impacts."
        )
        return

    sort_col_label = _policy_sort_widget("approx")

    rows = []
    for _, station_row in df.iterrows():
        base_vals = {f: float(station_row[f]) for f in selected_features if f in station_row.index}
        actual = float(station_row["entry"])
        line_code = station_row.get("line_code", "")

        surface_m = float(station_row.get("sidewalk_length_surface_neg1", 0.0))
        shade_m = float(station_row.get("sidewalk_length_shade_neg1", 0.0))
        delta_sw = sw_dev_map.get(line_code, 0.0) if has_sw_dev else 0.0

        record: dict = {
            "Station": station_row.get("station_name", ""),
            "Station ID": line_code,
            "Line Color": station_row.get("line_color", ""),
            "Observed Ridership": round(actual),
            "Predicted Ridership": round(predict_ridership(coef_df, base_vals, log_offset, model_spec)),
            "Surface −1 (m)": round(surface_m),
            "Shade −1 (m)": round(shade_m),
            "Proposed SW (m)": round(delta_sw),
        }

        # Upgrade Surface: β × Δ%x × actual
        surface_pct = 0.0
        for feat in surface_neg1_list:
            surface_pct += _elasticity_pct(coef_df, feat, base_vals.get(feat, 0.0), 0.0, model_spec)
        surface_chg = actual * surface_pct / 100
        record["Upgrade Surface (riders)"] = round(surface_chg)

        # Upgrade Shade: β × Δ%x × actual
        shade_pct = 0.0
        for feat in shade_neg1_list:
            shade_pct += _elasticity_pct(coef_df, feat, base_vals.get(feat, 0.0), 0.0, model_spec)
        shade_chg = actual * shade_pct / 100
        record["Upgrade Shade (riders)"] = round(shade_chg)

        # Add Proposed SW: β × Δ%x × actual
        sw_pct = 0.0
        sw_chg = 0.0
        if has_sw_dev and delta_sw > 0:
            sw_base = base_vals.get("sw_total_length", 0.0)
            sw_pct = _elasticity_pct(coef_df, "sw_total_length", sw_base, sw_base + delta_sw, model_spec)
            sw_chg = actual * sw_pct / 100
        record["Add Proposed SW (riders)"] = round(sw_chg)

        # Combined: sum of Δ%y (additive in elasticity approximation)
        combined_pct = surface_pct + shade_pct + sw_pct
        combined_chg = actual * combined_pct / 100
        record["All Upgrades Combined (riders)"] = round(combined_chg)

        rows.append(record)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(sort_col_label)

    st.dataframe(result_df, use_container_width=True, hide_index=True, height=500)

    _policy_download(result_df, "approx")

    # Calculation detail
    st.markdown("---")
    station_names = sorted(df["display_name"].tolist())
    sel = st.selectbox("Show calculation for station:", station_names, key="approx_calc_station")
    sel_row = df[df["display_name"] == sel].iloc[0]
    _show_elasticity_calculation(sel_row, coef_df, selected_features, walk_neg1, model_spec, sw_dev_map)


def _show_elasticity_calculation(
    station_row: pd.Series, coef_df: pd.DataFrame,
    selected_features: list[str], walk_neg1: dict,
    model_spec: str, sw_dev_map: dict[str, float] | None = None,
) -> None:
    """Show detailed elasticity (β × Δ%x) calculation for a selected station."""
    base_vals = {f: float(station_row[f]) for f in selected_features if f in station_row.index}
    actual = float(station_row["entry"])
    st.markdown(f"**Base observed ridership = {actual:,.0f}**")

    surface_neg1_list = walk_neg1.get("Surface", [])
    shade_neg1_list = walk_neg1.get("Shade", [])

    dim_labels = {"Surface": ("Upgrade Surface", surface_neg1_list),
                  "Shade": ("Upgrade Shade", shade_neg1_list)}
    total_pct = 0.0
    for dim, (label, neg1_list) in dim_labels.items():
        for feat in neg1_list:
            x_old = base_vals.get(feat, 0.0)
            vn = feat if model_spec == "linear" else f"log_{feat}"
            cr = coef_df[coef_df["variable"] == vn]
            if cr.empty:
                continue
            beta = float(cr["coef"].values[0])
            if x_old == 0:
                st.markdown(f"**{label}:** `{feat}` = 0 → 0 (no change)")
                continue
            pct_x = (0.0 - x_old) / x_old * 100
            pct_y = beta * pct_x
            riders = actual * pct_y / 100
            total_pct += pct_y
            st.markdown(
                f"**{label}:** `{feat}` = {x_old:.1f} → 0  \n"
                f"β = {beta:+.4f}  \n"
                f"Δ%x = (0 − {x_old:.1f}) / {x_old:.1f} × 100 = {pct_x:+.2f}%  \n"
                f"Δ%y = β × Δ%x = {beta:.4f} × ({pct_x:+.2f}%) = {pct_y:+.4f}%  \n"
                f"**Δ riders = {actual:,.0f} × {pct_y:.4f}% = {riders:+,.0f}**"
            )

    # Add Proposed SW
    has_sw_dev = sw_dev_map and "sw_total_length" in selected_features
    line_code = station_row.get("line_code", "")
    delta_sw = sw_dev_map.get(line_code, 0.0) if has_sw_dev else 0.0
    if has_sw_dev and delta_sw > 0:
        sw_base = base_vals.get("sw_total_length", 0.0)
        vn = "sw_total_length" if model_spec == "linear" else "log_sw_total_length"
        cr = coef_df[coef_df["variable"] == vn]
        if not cr.empty and sw_base > 0:
            beta = float(cr["coef"].values[0])
            pct_x = delta_sw / sw_base * 100
            pct_y = beta * pct_x
            riders = actual * pct_y / 100
            total_pct += pct_y
            st.markdown(
                f"**Add Proposed SW:** `sw_total_length` = {sw_base:.1f} → {sw_base + delta_sw:.1f} (+{delta_sw:.0f} m)  \n"
                f"β = {beta:+.4f}  \n"
                f"Δ%x = {delta_sw:.1f} / {sw_base:.1f} × 100 = {pct_x:+.2f}%  \n"
                f"Δ%y = β × Δ%x = {beta:.4f} × ({pct_x:+.2f}%) = {pct_y:+.4f}%  \n"
                f"**Δ riders = {actual:,.0f} × {pct_y:.4f}% = {riders:+,.0f}**"
            )

    # All Upgrades Combined
    combined_riders = actual * total_pct / 100
    st.markdown(
        f"**All Upgrades Combined:** sum of individual Δ%y  \n"
        f"Σ Δ%y = {total_pct:+.4f}%  \n"
        f"**Δ riders = {actual:,.0f} × {total_pct:.4f}% = {combined_riders:+,.0f}**"
    )


# ── Tab 6: Policy Comparison Plot ─────────────────────────────────────────

def render_policy_plot(
    df: pd.DataFrame, coef_df: pd.DataFrame,
    selected_features: list[str], log_offset: float,
    model_spec: str = "log-log",
    sw_dev_map: dict[str, float] | None = None,
) -> None:
    st.subheader("Policy Comparison — Exact vs Elasticity")
    st.caption(
        "All stations on the x-axis, sorted by observed ridership. "
        "Select a scenario to see the predicted ridership change from both methods."
    )

    walk_neg1 = _find_walk_vars(selected_features)
    walk_zero = _find_walk0_vars(selected_features)

    if not walk_neg1:
        st.warning(
            "No walkability neg1 variables are selected in the model. "
            "Please add surface/shade/obstacle neg1 variables to see policy impacts."
        )
        return

    surface_neg1_list = walk_neg1.get("Surface", [])
    shade_neg1_list = walk_neg1.get("Shade", [])
    surface_zero_list = walk_zero.get("Surface", [])
    shade_zero_list = walk_zero.get("Shade", [])

    has_sw_dev = sw_dev_map and "sw_total_length" in selected_features

    available_scenarios = []
    if surface_neg1_list:
        available_scenarios.append("Upgrade Surface")
    if shade_neg1_list:
        available_scenarios.append("Upgrade Shade")
    if has_sw_dev:
        available_scenarios.append("Add Proposed SW")
    available_scenarios.append("All Upgrades Combined")

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        scenario = st.selectbox(
            "Scenario", available_scenarios, key="policy_plot_scenario",
        )
    with c2:
        unit = st.radio(
            "Display unit", ["% change", "Rider change"],
            horizontal=True, key="policy_plot_unit",
        )
    use_pct = unit == "% change"

    # ── Compute exact & elasticity for all stations ───────────────────
    records: list[dict] = []

    for _, row in df.iterrows():
        base_vals = {f: float(row[f]) for f in selected_features if f in row.index}
        base_pred = predict_ridership(coef_df, base_vals, log_offset, model_spec)
        actual = float(row["entry"])
        station_label = row["display_name"]

        line_code = row.get("line_code", "")
        delta_sw = sw_dev_map.get(line_code, 0.0) if has_sw_dev else 0.0

        if scenario == "Upgrade Surface":
            # Exact: re-run equation with surface_neg1 → 0
            sc = _build_scenario_vals(base_vals, surface_neg1_list, surface_zero_list, row)
            exact_chg = predict_ridership(coef_df, sc, log_offset, model_spec) - base_pred
            # Elasticity: β × Δ%x × actual
            elast_pct = sum(
                _elasticity_pct(coef_df, f, base_vals.get(f, 0.0), 0.0, model_spec)
                for f in surface_neg1_list
            )
            approx_chg = actual * elast_pct / 100

        elif scenario == "Upgrade Shade":
            sc = _build_scenario_vals(base_vals, shade_neg1_list, shade_zero_list, row)
            exact_chg = predict_ridership(coef_df, sc, log_offset, model_spec) - base_pred
            elast_pct = sum(
                _elasticity_pct(coef_df, f, base_vals.get(f, 0.0), 0.0, model_spec)
                for f in shade_neg1_list
            )
            approx_chg = actual * elast_pct / 100

        elif scenario == "Add Proposed SW":
            if delta_sw > 0:
                sw_sc = dict(base_vals)
                sw_sc["sw_total_length"] = base_vals.get("sw_total_length", 0.0) + delta_sw
                exact_chg = predict_ridership(coef_df, sw_sc, log_offset, model_spec) - base_pred
                sw_base = base_vals.get("sw_total_length", 0.0)
                elast_pct = _elasticity_pct(coef_df, "sw_total_length", sw_base, sw_base + delta_sw, model_spec)
                approx_chg = actual * elast_pct / 100
            else:
                exact_chg = 0.0
                approx_chg = 0.0

        else:  # All Upgrades Combined
            # Exact: surface_neg1→0 + shade_neg1→0 + proposed SW, all at once
            combined_neg1 = surface_neg1_list + shade_neg1_list
            combined_zero = surface_zero_list + shade_zero_list
            combined_sc = _build_scenario_vals(base_vals, combined_neg1, combined_zero, row)
            if has_sw_dev and delta_sw > 0:
                combined_sc["sw_total_length"] = combined_sc.get(
                    "sw_total_length", base_vals.get("sw_total_length", 0.0)
                ) + delta_sw
            exact_chg = predict_ridership(coef_df, combined_sc, log_offset, model_spec) - base_pred
            # Elasticity: sum of individual Δ%y
            elast_pct = 0.0
            for f in surface_neg1_list:
                elast_pct += _elasticity_pct(coef_df, f, base_vals.get(f, 0.0), 0.0, model_spec)
            for f in shade_neg1_list:
                elast_pct += _elasticity_pct(coef_df, f, base_vals.get(f, 0.0), 0.0, model_spec)
            if has_sw_dev and delta_sw > 0:
                sw_base = base_vals.get("sw_total_length", 0.0)
                elast_pct += _elasticity_pct(coef_df, "sw_total_length", sw_base, sw_base + delta_sw, model_spec)
            approx_chg = actual * elast_pct / 100

        exact_pct = exact_chg / base_pred * 100 if base_pred > 0 else 0.0
        approx_pct_total = approx_chg / actual * 100 if actual > 0 else 0.0

        records.append({
            "Station": station_label,
            "Observed Ridership": actual,
            "Exact (%)": exact_pct,
            "Elasticity (%)": approx_pct_total,
            "Exact (riders)": exact_chg,
            "Elasticity (riders)": approx_chg,
        })

    result_df = pd.DataFrame(records).sort_values("Observed Ridership", ascending=False)
    stations_sorted = result_df["Station"].tolist()

    if use_pct:
        y_exact, y_approx = result_df["Exact (%)"], result_df["Elasticity (%)"]
        y_label = "Ridership change (%)"
    else:
        y_exact, y_approx = result_df["Exact (riders)"], result_df["Elasticity (riders)"]
        y_label = "Ridership change (riders)"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stations_sorted, y=y_exact, name="Exact",
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        x=stations_sorted, y=y_approx, name="Elasticity",
        marker_color="#EF553B",
    ))
    fig.update_layout(
        title=f"{scenario} — All Stations",
        xaxis_title="Station",
        yaxis_title=y_label,
        barmode="group",
        height=600,
        xaxis=dict(
            categoryorder="array", categoryarray=stations_sorted,
            tickangle=-90, tickfont=dict(size=8),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


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

    # Load multi-radius data if available, else fall back to single-radius rev16
    has_radii = os.path.exists(RADII_PATH)
    if has_radii:
        df_radii = load_radii_data()
        radii_list = get_radii_list(df_radii)
    else:
        df_radii = None
        radii_list = None

    df_all = load_data()
    sw_dev_map = load_sw_dev()

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
            _opt_label = st.session_state.get("opt_mode", "Best R²")
            _mode_map = {
                "Best R²": "best_r2",
                "Most significant vars": "max_signif",
                "All vars significant": "all_signif",
                "No multicollinearity (VIF<10)": "no_vif",
                "Correct signs only": "correct_signs",
                "Parsimonious (fewest vars)": "parsimonious",
            }
            _om = _mode_map.get(_opt_label, "best_r2")
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
        _mode_labels = {
            "best_r2": "Best R²",
            "max_signif": "Most significant vars",
            "all_signif": "All vars significant",
            "no_vif": "No multicollinearity",
            "correct_signs": "Correct signs",
            "parsimonious": "Parsimonious",
        }
        mode_label = _mode_labels.get(opt_mode_used, opt_mode_used)
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

    tabs = ["Model Results", "Station Explorer", "What-if Simulator",
            "Policy (Exact)", "Policy (Elasticity Approx.)",
            "Policy Plot"]
    if has_radii:
        tabs.append("Buffer Sensitivity")
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        render_model_results(model_result, coef_df, df, selected_features, log_offset)
    with tab_objs[1]:
        render_station_explorer(df, coef_df, selected_features, log_offset, model_spec)
    with tab_objs[2]:
        render_whatif(df, coef_df, selected_features, log_offset, model_spec)
    with tab_objs[3]:
        render_policy_exact(df, coef_df, selected_features, log_offset, model_spec, sw_dev_map=sw_dev_map)
    with tab_objs[4]:
        render_policy_approx(df, coef_df, selected_features, log_offset, model_spec, sw_dev_map=sw_dev_map)
    with tab_objs[5]:
        render_policy_plot(df, coef_df, selected_features, log_offset, model_spec, sw_dev_map=sw_dev_map)
    if has_radii and len(tab_objs) > 6:
        with tab_objs[6]:
            render_buffer_sensitivity(
                df_radii, selected_features, sel_stations,
                log_offset, sig_level, model_spec,
                endog_features, instruments,
            )


if __name__ == "__main__":
    main()
