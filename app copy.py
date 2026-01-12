from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from tools.apt_r import apt_via_r


from tools.io import read_excel, list_sheets
from tools.transforms import prepare_series, transform_series
from tools.stationarity import adf_test, kpss_test, zivot_andrews_test
from tools.bai_perron_r import bai_perron_via_r
from tools.engle_granger import engle_granger_test
from tools.johansen import johansen_test
from tools.vecm import vecm_fit, vecm_irf_fevd
from tools.ardl_r import ardl_via_r
from tools.nardl_r import nardl_via_r

# NEW: APT + GARCH (via R wrappers you added)
from tools.apt_r import apt_via_r
from tools.garch_r import ugarch_via_r, dcc_mgarch_via_r


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Time Series Bot", layout="wide")
st.title("📈 Time Series Bot")


# -----------------------------
# Upload & Sheet
# -----------------------------
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file → choose columns → run tests/models.")
    st.stop()

sheets = list_sheets(uploaded)
sheet = st.selectbox("Select sheet", sheets, index=0)

df = read_excel(uploaded, sheet_name=sheet)
df.columns = df.columns.astype(str).str.strip()  # robust column names
cols = df.columns.tolist()

st.caption("Preview")
st.dataframe(df.head(50), use_container_width=True)


# -----------------------------
# Helper functions
# -----------------------------
def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def get_series(df_work: pd.DataFrame, col: str, date_col: str | None) -> pd.Series:
    """
    Read + per-series transform based on session_state.
    """
    s0 = prepare_series(df_work, col, date_col=date_col)
    use_log = bool(st.session_state.get(f"log__{col}", False))
    diff_k = int(st.session_state.get(f"diff__{col}", 0))
    return transform_series(s0, use_log=use_log, diff_k=diff_k)


def ensure_no_diff_for_levels(cols_: list[str], context: str) -> None:
    """
    Many models here are typically specified in LEVELS:
    - Cointegration tests (Engle-Granger, Johansen)
    - VECM
    - ARDL / NARDL
    - APT (asymmetric ECM / threshold cointegration)

    Log(levels) is fine; differencing changes the model meaning.
    """
    bad = [c for c in cols_ if int(st.session_state.get(f"diff__{c}", 0)) != 0]
    if bad:
        st.error(
            f"For **{context}**, these variables must NOT be differenced (diff_k must be 0): "
            + ", ".join(bad)
        )
        st.stop()


def make_long_df(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    wide = pd.concat(series_map, axis=1).dropna(how="any")
    long = wide.reset_index().melt(
        id_vars=wide.index.name or "index",
        var_name="variable",
        value_name="value",
    )
    if "index" not in long.columns:
        long = long.rename(columns={long.columns[0]: "index"})
    return long


def build_levels_df(df_work: pd.DataFrame, cols_: list[str], date_col: str | None) -> pd.DataFrame:
    """
    Build aligned dataframe for selected variables, using per-variable transforms.
    (Upstream we block differencing for models that require LEVELS.)
    """
    ser_list = [get_series(df_work, c, date_col=date_col).rename(c) for c in cols_]
    out = pd.concat(ser_list, axis=1).dropna()
    return out


def display_vecm_irf_fevd_all(rf: dict) -> None:
    """
    Show ALL IRFs and FEVDs (no dropdowns) to avoid Streamlit rerun issues.
    Expected rf keys: columns, irfs, fevd
    irfs shape: (h+1, k, k)  -> [t, response_i, shock_j]
    fevd shape: (h, k, k)    -> [t, response_i, shock_j]
    """
    cols_v = rf["columns"]
    irfs = np.asarray(rf["irfs"], dtype=float)
    fevd = np.asarray(rf["fevd"], dtype=float)

    k = len(cols_v)

    st.subheader("Impulse Response Functions (IRF) — all combinations")
    for shock_j in range(k):
        shock_name = cols_v[shock_j]
        st.markdown(f"### Shock (impulse): **{shock_name}**")
        for resp_i in range(k):
            resp_name = cols_v[resp_i]
            y = irfs[:, resp_i, shock_j]
            df_irf = pd.DataFrame({"horizon": np.arange(len(y)), "irf": y})
            fig = px.line(df_irf, x="horizon", y="irf", title=f"IRF: Response {resp_name} ← Shock in {shock_name}")
            fig.update_layout(xaxis_title="Horizon (h)", yaxis_title="Response")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Error Variance Decomposition (FEVD) — all responses")
    h = fevd.shape[0]
    horizons = np.arange(1, h + 1)

    for resp_i in range(k):
        resp_name = cols_v[resp_i]
        mat = fevd[:, resp_i, :]  # (h, k)
        df_f = pd.DataFrame(mat, columns=cols_v)
        df_f["horizon"] = horizons
        long = df_f.melt(id_vars="horizon", var_name="shock", value_name="share")
        fig = px.line(long, x="horizon", y="share", color="shock", title=f"FEVD: Shock shares in Var({resp_name})")
        fig.update_layout(xaxis_title="Horizon (h)", yaxis_title="Share")
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 0) Column selection
# -----------------------------
st.subheader("0) Select columns")

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    date_col_choice = st.selectbox("Date column (optional)", ["(none)"] + cols, index=0)
    date_col = None if date_col_choice == "(none)" else date_col_choice

with c2:
    value_cols = [c for c in cols if c != date_col] if date_col else cols
    if not value_cols:
        st.error("No value columns available.")
        st.stop()

    y_col = st.selectbox("y (dependent) — for single-series tests & Engle-Granger", value_cols, index=0)

    x_options = [c for c in value_cols if c != y_col] or value_cols
    x_col = st.selectbox("x (regressor) — for Engle-Granger", x_options, index=0)

with c3:
    st.write("")


# -----------------------------
# 0.5) Sample / period filter
# -----------------------------
st.subheader("0.5) Sample filter")

df_work = df.copy()

if date_col is not None:
    dt = _safe_to_datetime(df_work[date_col])
    df_work = df_work.assign(__dt__=dt).dropna(subset=["__dt__"]).sort_values("__dt__")
    min_dt = df_work["__dt__"].min()
    max_dt = df_work["__dt__"].max()

    if pd.isna(min_dt) or pd.isna(max_dt):
        st.warning("Date column could not be parsed reliably. Using full sample.")
        df_work = df.copy()
    else:
        d1, d2 = st.date_input(
            "Select date range (inclusive)",
            value=(min_dt.date(), max_dt.date()),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
        )
        start_dt = pd.Timestamp(d1)
        end_dt = pd.Timestamp(d2) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df_work = df_work[(df_work["__dt__"] >= start_dt) & (df_work["__dt__"] <= end_dt)].drop(columns=["__dt__"])

else:
    n = len(df_work)
    if n >= 2:
        i1, i2 = st.slider("Select observation range (inclusive)", 0, n - 1, (0, n - 1))
        df_work = df_work.iloc[i1: i2 + 1].copy()

st.caption(f"Filtered rows: {len(df_work)}")


# -----------------------------
# 1) Select tool / model
# -----------------------------
st.subheader("1) Select tool / model")

tool = st.selectbox(
    "Tool / model",
    [
        "ADF (Augmented Dickey-Fuller stationarity test)",
        "KPSS (Kwiatkowski–Phillips–Schmidt–Shin stationarity test)",
        "Zivot-Andrews (unit root test with 1 structural break)",
        "Bai-Perron (multiple structural breaks, via R)",
        "Engle-Granger (cointegration, 2 variables)",
        "Johansen (cointegration, multiple variables)",
        "VECM (fit + IRF/FEVD)",
        "ARDL (AutoRegressive Distributed Lag, via R)",
        "NARDL (Nonlinear ARDL, via R)",
        "APT (Asymmetric / Threshold ECM & tests, via R; 2 variables)",
        "GARCH (univariate, via R)",
        "MGARCH (DCC, via R)",
    ],
)


# -----------------------------
# 1.1) Multi selections (Johansen/VECM + exogs)
# -----------------------------
df_j = None
df_ex = None
df_ex_c = None

j_cols: list[str] = []
exog_cols: list[str] = []
exog_coint_cols: list[str] = []

if tool.startswith("Johansen") or tool.startswith("VECM"):
    st.markdown("**Multivariate selection (LEVELS)**")

    j_cols = st.multiselect(
        "Endogenous variables (Johansen/VECM) — choose at least 2",
        value_cols,
        default=value_cols[:2] if len(value_cols) >= 2 else value_cols,
        key="j_cols_multi",
    )

    exog_cols = st.multiselect(
        "Exogenous variables (short-run exog, optional)",
        [c for c in value_cols if c not in j_cols],
        default=[],
        key="exog_cols_multi",
    )

    exog_coint_cols = st.multiselect(
        "Exogenous variables in cointegration (long-run exog_coint, optional)",
        [c for c in value_cols if c not in j_cols and c not in exog_cols],
        default=[],
        key="exog_coint_cols_multi",
    )


# -----------------------------
# 1.2) ARDL / NARDL selections
# -----------------------------
ardl_y = None
ardl_x: list[str] = []
ardl_z: list[str] = []

nardl_y = None
nardl_x = None
nardl_z: list[str] = []

if tool.startswith("ARDL") or tool.startswith("NARDL"):
    st.markdown("**ARDL / NARDL selection (LEVELS)**")

    ardl_y = st.selectbox(
        "Dependent variable (y)",
        value_cols,
        index=value_cols.index(y_col) if y_col in value_cols else 0,
        key="ardl_y_select",
    )

    pool_x = [c for c in value_cols if c != ardl_y] or value_cols

    if tool.startswith("ARDL"):
        ardl_x = st.multiselect(
            "Regressors (x) — at least 1",
            pool_x,
            default=pool_x[:1],
            key="ardl_x_multi",
        )
        ardl_z = st.multiselect(
            "Additional controls (z) — optional",
            [c for c in pool_x if c not in ardl_x],
            default=[],
            key="ardl_z_multi",
        )

    if tool.startswith("NARDL"):
        nardl_y = ardl_y
        nardl_x = st.selectbox(
            "Main regressor (x) (will be decomposed into x⁺ and x⁻)",
            pool_x,
            index=0,
            key="nardl_x_select",
        )
        nardl_z = st.multiselect(
            "Additional controls (z) — optional",
            [c for c in pool_x if c != nardl_x],
            default=[],
            key="nardl_z_multi",
        )


# -----------------------------
# 1.3) APT selections (2 variables)
# -----------------------------
apt_y = None
apt_x = None
if tool.startswith("APT"):
    st.markdown("**APT selection (LEVELS; exactly 2 variables)**")
    apt_y = st.selectbox(
        "APT dependent variable (y)",
        value_cols,
        index=value_cols.index(y_col) if y_col in value_cols else 0,
        key="apt_y_select",
    )
    apt_x_pool = [c for c in value_cols if c != apt_y] or value_cols
    apt_x = st.selectbox(
        "APT regressor / price (x)",
        apt_x_pool,
        index=0,
        key="apt_x_select",
    )


# -----------------------------
# 1.4) MGARCH selection
# -----------------------------
mgarch_cols: list[str] = []
if tool.startswith("MGARCH"):
    st.markdown("**MGARCH selection (2+ variables; typically returns / transformed series are OK)**")
    mgarch_cols = st.multiselect(
        "Series for DCC-MGARCH (2+)",
        value_cols,
        default=value_cols[:2] if len(value_cols) >= 2 else value_cols,
        key="mgarch_cols_select",
    )


# -----------------------------
# 2) Transformations per variable
# -----------------------------
st.subheader("2) Transformations (per variable)")

plot_cols = st.multiselect(
    "Variables to plot (multiple allowed)",
    value_cols,
    default=[y_col],
    key="plot_cols_multi",
)

needed_cols = set(plot_cols)
needed_cols.add(y_col)
needed_cols.add(x_col)
needed_cols.update(j_cols)
needed_cols.update(exog_cols)
needed_cols.update(exog_coint_cols)
if ardl_y is not None:
    needed_cols.add(ardl_y)
needed_cols.update(ardl_x)
needed_cols.update(ardl_z)
if nardl_y is not None:
    needed_cols.add(nardl_y)
if nardl_x is not None:
    needed_cols.add(nardl_x)
needed_cols.update(nardl_z)
if apt_y is not None:
    needed_cols.add(apt_y)
if apt_x is not None:
    needed_cols.add(apt_x)
needed_cols.update(mgarch_cols)

needed_cols = [c for c in value_cols if c in needed_cols]  # stable order

with st.expander("Per-variable settings (log / difference)", expanded=True):
    st.caption(
        "Notes:\n"
        "- Log(Levels) is typically fine.\n"
        "- Cointegration / VECM / ARDL / NARDL / APT are typically specified in LEVELS → do NOT difference those variables.\n"
        "- GARCH/MGARCH often use returns; differencing is allowed there.\n"
    )
    for c in needed_cols:
        cc1, cc2, cc3 = st.columns([2, 1, 1])
        with cc1:
            st.write(f"**{c}**")
        with cc2:
            st.checkbox("log", key=f"log__{c}", value=bool(st.session_state.get(f"log__{c}", False)))
        with cc3:
            st.number_input(
                "diff_k",
                min_value=0,
                max_value=10,
                step=1,
                key=f"diff__{c}",
                value=int(st.session_state.get(f"diff__{c}", 0)),
            )


# -----------------------------
# 3) Plot
# -----------------------------
st.subheader("3) Plot")

if not plot_cols:
    st.info("Select at least one variable for the plot.")
else:
    try:
        series_map = {c: get_series(df_work, c, date_col=date_col).rename(c) for c in plot_cols}
        long = make_long_df(series_map)
        if date_col is None:
            long["index"] = long["index"].astype(str)

        fig = px.line(long, x="index", y="value", color="variable", title="Selected series (after transformations)")
        fig.update_layout(xaxis_title="Time / index", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Plot error: {e}")


# -----------------------------
# 4) Run tool / model
# -----------------------------
st.subheader("4) Run tool / model")

with st.form("params_form", border=True):
    st.write("Parameters")

    # ADF
    if tool.startswith("ADF"):
        regression_ui = st.selectbox(
            "Deterministic terms",
            [
                "c (Constant)",
                "ct (Constant + linear trend)",
                "n (No constant / no trend)",
            ],
            index=0,
        )
        autolag_ui = st.selectbox(
            "Lag selection",
            ["AIC", "BIC", "t-stat", "None (use maxlag)"],
            index=0,
        )
        maxlag = st.number_input("maxlag (0 = auto)", min_value=0, max_value=200, value=0, step=1)
        use_maxlag = st.checkbox("Force maxlag", value=False)

    # KPSS
    elif tool.startswith("KPSS"):
        regression_ui = st.selectbox(
            "Stationarity type under H0",
            [
                "c (Level-stationary)",
                "ct (Trend-stationary)",
            ],
            index=0,
        )
        nlags_mode = st.selectbox("Lags", ["auto", "manual"], index=0)
        nlags = 0
        if nlags_mode == "manual":
            nlags = st.number_input("nlags", min_value=1, max_value=200, value=12, step=1)

    # Zivot-Andrews
    elif tool.startswith("Zivot"):
        regression_ui = st.selectbox(
            "Break type",
            [
                "c (Break in intercept)",
                "t (Break in trend)",
                "ct (Break in intercept + trend)",
            ],
            index=2,
        )
        autolag_ui = st.selectbox("Lag selection", ["AIC", "BIC", "t-stat"], index=0)
        maxlag = st.number_input("maxlag (0 = auto)", min_value=0, max_value=200, value=0, step=1)
        use_maxlag = st.checkbox("Force maxlag", value=False)

    # Bai-Perron (R)
    elif tool.startswith("Bai-Perron"):
        bp_model_ui = st.selectbox(
            "Model",
            [
                "level (Level shifts only)",
                "trend (Level + trend shifts)",
            ],
            index=1,
        )
        bp_mode_ui = st.selectbox(
            "Number of breaks",
            [
                "unknown (Model selection)",
                "fixed (User-specified k)",
            ],
            index=0,
        )
        ic_ui = st.selectbox(
            "Information criterion (used when breaks are unknown)",
            [
                "KT (Kejriwal–Perron / KT)",
                "BIC (Bayesian Information Criterion)",
                "LWZ (Liu–Wu–Zidek)",
            ],
            index=0,
        )
        eps1 = st.slider("Trimming (eps1)", 0.05, 0.25, 0.15, 0.01)
        max_breaks = st.number_input("Maximum breaks (m)", min_value=1, max_value=20, value=5, step=1)
        k_fixed = 0
        if bp_mode_ui.startswith("fixed"):
            k_fixed = st.number_input("Fixed number of breaks (k)", min_value=0, max_value=int(max_breaks), value=1, step=1)

    # Engle-Granger
    elif tool.startswith("Engle-Granger"):
        eg_trend_ui = st.selectbox(
            "Deterministic terms in cointegrating regression",
            [
                "c (Constant)",
                "ct (Constant + linear trend)",
                "n (No constant / no trend)",
            ],
            index=0,
        )
        eg_autolag_ui = st.selectbox(
            "Residual ADF lag selection",
            [
                "aic (AIC)",
                "bic (BIC)",
                "t-stat (t-stat)",
                "None (No autolag)",
            ],
            index=0,
        )
        eg_direction_ui = st.selectbox(
            "Regression direction",
            [
                "y ~ x (y on x)",
                "x ~ y (x on y)",
                "both (both directions)",
            ],
            index=0,
        )

    # Johansen
    elif tool.startswith("Johansen"):
        det_ui = st.selectbox(
            "Deterministic terms",
            [
                "-1 (None)",
                "0 (Constant)",
                "1 (Trend)",
            ],
            index=1,
        )
        k_ar_diff = st.number_input(
            "k_ar_diff (number of lagged differences)",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
        )

    # VECM
    elif tool.startswith("VECM"):
        vecm_det_ui = st.selectbox(
            "Deterministic specification (statsmodels)",
            [
                "n (None)",
                "co (Constant outside cointegration relation)",
                "ci (Constant inside cointegration relation)",
                "cto (Trend outside cointegration relation)",
                "cti (Trend inside cointegration relation)",
            ],
            index=2,
        )
        vecm_rank_mode = st.selectbox(
            "Cointegration rank selection",
            [
                "manual (Choose rank)",
                "auto (Johansen trace test at 95%)",
            ],
            index=0,
        )
        vecm_rank = st.number_input("Manual rank (r)", min_value=0, max_value=10, value=1, step=1)
        vecm_k_ar_diff = st.number_input("k_ar_diff", min_value=1, max_value=20, value=2, step=1)

        st.markdown("**IRF / FEVD (optional)**")
        do_irf_fevd = st.checkbox("Compute IRF & FEVD", value=False)
        irf_h = st.number_input("Horizon (h)", min_value=1, max_value=60, value=12, step=1)

    # ARDL (R)
    elif tool.startswith("ARDL"):
        ic_ui = st.selectbox(
            "Lag order selection criterion",
            [
                "aic (AIC)",
                "bic (BIC)",
                "hq (Hannan–Quinn / HQ)",
            ],
            index=0,
        )
        max_p = st.number_input("Max AR lag for y (p max)", min_value=0, max_value=20, value=4, step=1)
        max_q = st.number_input("Max distributed lag for x/z (q max)", min_value=0, max_value=20, value=4, step=1)

        case_ui = st.selectbox(
            "Bounds test case (deterministic structure)",
            [
                "1 (No intercept, no trend)",
                "2 (Restricted intercept, no trend)",
                "3 (Unrestricted intercept, no trend)",
                "4 (Unrestricted intercept, restricted trend)",
                "5 (Unrestricted intercept, unrestricted trend)",
            ],
            index=2,
        )

        do_bounds = st.checkbox("Run bounds test (per selected case)", value=True)

    # NARDL (R)
    elif tool.startswith("NARDL"):
        ic_ui = st.selectbox(
            "Lag order selection criterion",
            [
                "aic (AIC)",
                "bic (BIC)",
                "hq (Hannan–Quinn / HQ)",
            ],
            index=0,
        )
        maxlag = st.number_input("Max lag (maxlag)", min_value=0, max_value=20, value=4, step=1)

        case_ui = st.selectbox(
            "NARDL case (deterministic structure)",
            [
                "1 (No intercept, no trend)",
                "2 (Restricted intercept, no trend)",
                "3 (Unrestricted intercept, no trend)",
                "4 (Unrestricted intercept, restricted trend)",
                "5 (Unrestricted intercept, unrestricted trend)",
            ],
            index=2,
        )
        graph = st.checkbox("Request plots from R (graph=TRUE)", value=False)

    # APT (R)
    elif tool.startswith("APT"):
        apt_mode_ui = st.selectbox(
            "APT function",
            [
                "ciTarFit (Threshold cointegration: fit given lag/threshold)",
                "ciTarLag (Threshold cointegration: select lag path)",
                "ciTarThd (Threshold cointegration: estimate threshold)",
                "ecmSymFit (Symmetric ECM)",
                "ecmAsyFit (Asymmetric ECM)",
                "ecmAsyTest (Asymmetry tests based on ecmAsyFit)",
                "ecmDiag (Diagnostics for ECM model)",
            ],
            index=0,
        )

        st.markdown("**Common parameters**")
        apt_lag = st.number_input("lag (ECM/threshold lag)", min_value=1, max_value=24, value=1, step=1)
        apt_digits = st.number_input("digits (printing)", min_value=1, max_value=8, value=3, step=1)

        st.markdown("**Threshold / TAR parameters**")
        apt_model_ui = st.selectbox(
            "Threshold model type",
            [
                "tar (Threshold Autoregressive)",
                "mtar (Momentum TAR)",
            ],
            index=0,
        )
        apt_thresh = st.number_input("threshold value (thresh)", value=0.0, step=0.1)

        apt_maxlag = st.number_input("maxlag (only for ciTarLag)", min_value=1, max_value=24, value=4, step=1)
        apt_adjust = st.checkbox("adjust (only for ciTarLag)", value=True)

        apt_th_range = st.slider("th_range (only for ciTarThd)", 0.01, 0.50, 0.15, 0.01)

        st.markdown("**Asymmetric ECM parameters**")
        apt_split = st.checkbox("split (decompose x into positive/negative changes)", value=True)
        apt_ecm_model_ui = st.selectbox(
            "ecmAsyFit model form",
            [
                "linear (linear adjustment)",
                "nonlinear (nonlinear adjustment)",
            ],
            index=0,
        )

        apt_diag_which = st.selectbox(
            "ecmDiag: which ECM to diagnose",
            [
                "sym (Symmetric ECM)",
                "asy (Asymmetric ECM)",
            ],
            index=0,
        )

    # GARCH (R) univariate
    elif tool.startswith("GARCH"):
        garch_var_ui = st.selectbox(
            "Variance model",
            [
                "sGARCH (standard GARCH)",
                "eGARCH (exponential GARCH)",
                "gjrGARCH (GJR / threshold GARCH)",
            ],
            index=0,
        )
        garch_p = st.number_input("GARCH p", min_value=1, max_value=5, value=1, step=1)
        garch_q = st.number_input("GARCH q", min_value=1, max_value=5, value=1, step=1)

        arma_p = st.number_input("Mean ARMA p", min_value=0, max_value=10, value=0, step=1)
        arma_q = st.number_input("Mean ARMA q", min_value=0, max_value=10, value=0, step=1)
        include_mean = st.checkbox("Include mean in the mean equation", value=True)

        dist_ui = st.selectbox(
            "Innovation distribution",
            [
                "norm (Normal)",
                "std (Student-t)",
                "ged (GED)",
            ],
            index=0,
        )

    # MGARCH (R) DCC
    elif tool.startswith("MGARCH"):
        mgarch_var_ui = st.selectbox(
            "Univariate variance model (applied to each series)",
            [
                "sGARCH (standard GARCH)",
                "eGARCH (exponential GARCH)",
                "gjrGARCH (GJR / threshold GARCH)",
            ],
            index=0,
        )
        mgarch_p = st.number_input("GARCH p (each series)", min_value=1, max_value=5, value=1, step=1)
        mgarch_q = st.number_input("GARCH q (each series)", min_value=1, max_value=5, value=1, step=1)

        dcc_a = st.number_input("DCC a", min_value=1, max_value=5, value=1, step=1)
        dcc_b = st.number_input("DCC b", min_value=1, max_value=5, value=1, step=1)

        dist_ui = st.selectbox(
            "Multivariate distribution",
            [
                "norm (Normal)",
                "std (Student-t; may require different spec in some setups)",
            ],
            index=0,
        )

    run = st.form_submit_button("Run")


# -----------------------------
# 5) Execute
# -----------------------------
if run:
    try:
        # Build main single series (used by single-series tests and also GARCH)
        s = get_series(df_work, y_col, date_col=date_col)

        if tool.startswith("ADF"):
            reg_map = {
                "c (Constant)": "c",
                "ct (Constant + linear trend)": "ct",
                "n (No constant / no trend)": "n",
            }
            regression = reg_map[regression_ui]

            if autolag_ui == "AIC":
                autolag = "AIC"
            elif autolag_ui == "BIC":
                autolag = "BIC"
            elif autolag_ui == "t-stat":
                autolag = "t-stat"
            else:
                autolag = None

            maxlag_val = int(maxlag) if use_maxlag else None
            res = adf_test(s, regression=regression, autolag=autolag, maxlag=maxlag_val)
            st.success("ADF computed.")
            st.json(res)
            st.info("H0: Unit root (non-stationary). p < 0.05 → evidence for stationarity.")

        elif tool.startswith("KPSS"):
            reg_map = {
                "c (Level-stationary)": "c",
                "ct (Trend-stationary)": "ct",
            }
            regression = reg_map[regression_ui]

            nlags_val = "auto" if nlags_mode == "auto" else int(nlags)
            res = kpss_test(s, regression=regression, nlags=nlags_val)
            st.success("KPSS computed.")
            st.json(res)
            st.info("H0: Stationary. p < 0.05 → evidence against stationarity.")

        elif tool.startswith("Zivot"):
            reg_map = {
                "c (Break in intercept)": "c",
                "t (Break in trend)": "t",
                "ct (Break in intercept + trend)": "ct",
            }
            regression = reg_map[regression_ui]
            maxlag_val = int(maxlag) if use_maxlag else None

            autolag = autolag_ui
            res = zivot_andrews_test(s, regression=regression, autolag=autolag, maxlag=maxlag_val)
            st.success("Zivot-Andrews computed.")
            st.json(res)
            st.info("Unit-root test allowing one structural break. break_index indicates estimated break position.")

        elif tool.startswith("Bai-Perron"):
            bp_model = "level" if bp_model_ui.startswith("level") else "trend"
            bp_mode = "unknown" if bp_mode_ui.startswith("unknown") else "fixed"
            ic_code = ic_ui.split(" ")[0].strip()  # KT/BIC/LWZ

            with st.spinner("Running Bai-Perron via R..."):
                res = bai_perron_via_r(
                    s,
                    model=bp_model,
                    max_breaks=int(max_breaks),
                    ic=ic_code,
                    eps1=float(eps1),
                    mode=bp_mode,
                    k_fixed=int(k_fixed),
                )
            if "error" in res:
                st.error(res["error"])
                if "stderr" in res:
                    st.code(res["stderr"])
                if "stdout" in res:
                    st.code(res["stdout"])
            else:
                st.success("Bai-Perron computed (R).")
                st.json(res)
                if res.get("selected_model_text"):
                    st.subheader("R output")
                    st.text(res["selected_model_text"])

        elif tool.startswith("Engle-Granger"):
            ensure_no_diff_for_levels([y_col, x_col], context="Engle-Granger")

            y = get_series(df_work, y_col, date_col=date_col)
            x = get_series(df_work, x_col, date_col=date_col)

            trend_map = {
                "c (Constant)": "c",
                "ct (Constant + linear trend)": "ct",
                "n (No constant / no trend)": "n",
            }
            eg_trend = trend_map[eg_trend_ui]

            autolag_map = {
                "aic (AIC)": "aic",
                "bic (BIC)": "bic",
                "t-stat (t-stat)": "t-stat",
                "None (No autolag)": None,
            }
            eg_autolag = autolag_map[eg_autolag_ui]

            if eg_direction_ui.startswith("y ~ x"):
                res = engle_granger_test(y, x, trend=eg_trend, autolag=eg_autolag)
                st.success("Engle–Granger computed (y ~ x).")
                st.json(res)

            elif eg_direction_ui.startswith("x ~ y"):
                res = engle_granger_test(x, y, trend=eg_trend, autolag=eg_autolag)
                st.success("Engle–Granger computed (x ~ y).")
                st.json(res)

            else:
                st.subheader("Direction: y ~ x")
                st.json(engle_granger_test(y, x, trend=eg_trend, autolag=eg_autolag))
                st.subheader("Direction: x ~ y")
                st.json(engle_granger_test(x, y, trend=eg_trend, autolag=eg_autolag))

            st.info("H0: No cointegration. p < 0.05 → evidence for cointegration.")

        elif tool.startswith("Johansen"):
            if len(j_cols) < 2:
                st.error("Select at least 2 endogenous variables.")
                st.stop()

            ensure_no_diff_for_levels(j_cols + exog_cols + exog_coint_cols, context="Johansen (cointegration test)")

            df_j = build_levels_df(df_work, j_cols, date_col=date_col)
            det_order = int(det_ui.split(" ")[0].strip())  # -1/0/1

            with st.spinner("Running Johansen..."):
                res = johansen_test(df_j, det_order=det_order, k_ar_diff=int(k_ar_diff))

            if "error" in res:
                st.error(res["error"])
                st.json(res)
            else:
                st.success("Johansen computed.")
                st.json(res)

                k = res["n_vars"]
                ranks = list(range(k))
                trace = res["trace_stat"]
                maxeig = res["max_eig_stat"]
                trace_cv = res["trace_cv_90_95_99"]
                maxeig_cv = res["max_eig_cv_90_95_99"]

                tab = pd.DataFrame(
                    {
                        "r (rank H0)": ranks,
                        "trace": trace,
                        "trace_cv_90": [row[0] for row in trace_cv],
                        "trace_cv_95": [row[1] for row in trace_cv],
                        "trace_cv_99": [row[2] for row in trace_cv],
                        "max_eig": maxeig,
                        "max_eig_cv_90": [row[0] for row in maxeig_cv],
                        "max_eig_cv_95": [row[1] for row in maxeig_cv],
                        "max_eig_cv_99": [row[2] for row in maxeig_cv],
                    }
                )
                st.subheader("Johansen test table")
                st.dataframe(tab, use_container_width=True)
                st.info("Rule of thumb: statistic > 95% critical value ⇒ reject H0(r) ⇒ rank > r.")

        elif tool.startswith("VECM"):
            if len(j_cols) < 2:
                st.error("Select at least 2 endogenous variables.")
                st.stop()

            ensure_no_diff_for_levels(j_cols + exog_cols + exog_coint_cols, context="VECM")

            df_j = build_levels_df(df_work, j_cols, date_col=date_col)
            df_ex = build_levels_df(df_work, exog_cols, date_col=date_col) if exog_cols else None
            df_ex_c = build_levels_df(df_work, exog_coint_cols, date_col=date_col) if exog_coint_cols else None

            parts = [df_j]
            if df_ex is not None:
                parts.append(df_ex)
            if df_ex_c is not None:
                parts.append(df_ex_c)

            all_df = pd.concat(parts, axis=1).dropna()
            df_j = all_df[j_cols]
            df_ex = all_df[exog_cols] if exog_cols else None
            df_ex_c = all_df[exog_coint_cols] if exog_coint_cols else None

            vecm_det = vecm_det_ui.split(" ")[0].strip()  # n/co/ci/cto/cti

            # choose rank
            rank_use = int(vecm_rank)
            if vecm_rank_mode.startswith("auto"):
                jres = johansen_test(df_j, det_order=0, k_ar_diff=int(vecm_k_ar_diff))
                if "error" in jres:
                    st.error("Auto rank selection failed (Johansen).")
                    st.json(jres)
                    st.stop()

                trace = jres["trace_stat"]
                cv95 = [row[1] for row in jres["trace_cv_90_95_99"]]

                rank_use = 0
                for r, (ts, c) in enumerate(zip(trace, cv95)):
                    if ts > c:
                        rank_use = r + 1
                rank_use = min(rank_use, df_j.shape[1] - 1)

            with st.spinner("Fitting VECM..."):
                res = vecm_fit(
                    df_j,
                    rank=rank_use,
                    k_ar_diff=int(vecm_k_ar_diff),
                    deterministic=vecm_det,
                    exog=df_ex,
                    exog_coint=df_ex_c,
                )

            if "error" in res:
                st.error(res["error"])
                if "detail" in res:
                    st.code(res["detail"])
                if "hint" in res:
                    st.info(res["hint"])
                st.stop()

            st.success("VECM fitted.")
            st.write(f"Rank used: **{rank_use}**")

            if res.get("summary"):
                st.subheader("VECM summary")
                st.text(res["summary"])
            else:
                st.json(res)

            if do_irf_fevd:
                with st.spinner("Computing IRF/FEVD..."):
                    rf = vecm_irf_fevd(
                        df_j,
                        rank=rank_use,
                        k_ar_diff=int(vecm_k_ar_diff),
                        deterministic=vecm_det,
                        exog=df_ex,
                        exog_coint=df_ex_c,
                        horizon=int(irf_h),
                    )

                if "error" in rf:
                    st.error(rf["error"])
                    if "detail" in rf:
                        st.code(rf["detail"])
                    if "hint" in rf:
                        st.info(rf["hint"])
                else:
                    display_vecm_irf_fevd_all(rf)

        elif tool.startswith("ARDL"):
            if ardl_y is None or not ardl_x:
                st.error("ARDL: select dependent y and at least one regressor x.")
                st.stop()

            ensure_no_diff_for_levels([ardl_y] + list(ardl_x) + list(ardl_z), context="ARDL")

            df_vars = build_levels_df(df_work, [ardl_y] + list(ardl_x) + list(ardl_z), date_col=date_col)

            ic_code = ic_ui.split(" ")[0].strip().lower()  # aic/bic/hq
            case_num = int(case_ui.split(" ")[0].strip())  # 1..5

            with st.spinner("Running ARDL via R..."):
                res = ardl_via_r(
                    df=df_vars,
                    y=ardl_y,
                    x=list(ardl_x),
                    z=list(ardl_z),
                    ic=ic_code,
                    max_p=int(max_p),
                    max_q=int(max_q),
                    do_bounds=bool(do_bounds),
                    case=int(case_num),
                )

            if not isinstance(res, dict) or res.get("ok") is False:
                st.error("ARDL failed.")
                st.json(res)
            else:
                st.success("ARDL computed (R).")
                if "summary" in res and res["summary"]:
                    st.subheader("Model summary (R)")
                    st.text(res["summary"])
                st.subheader("Full output (JSON)")
                st.json(res)
                if "bounds" in res and res["bounds"]:
                    st.subheader("Bounds test (per selected case)")
                    st.json(res["bounds"])

        elif tool.startswith("NARDL"):
            if nardl_y is None or nardl_x is None:
                st.error("NARDL: select y and x.")
                st.stop()

            ensure_no_diff_for_levels([nardl_y, nardl_x] + list(nardl_z), context="NARDL")

            df_vars = build_levels_df(df_work, [nardl_y, nardl_x] + list(nardl_z), date_col=date_col)

            ic_code = ic_ui.split(" ")[0].strip().lower()
            case_num = int(case_ui.split(" ")[0].strip())

            rhs = " + ".join([nardl_x] + list(nardl_z)) if nardl_z else nardl_x
            formula = f"{nardl_y} ~ {rhs}"

            with st.spinner("Running NARDL via R..."):
                res = nardl_via_r(
                    df=df_vars,
                    formula=formula,
                    ic=ic_code,
                    maxlag=int(maxlag),
                    case=int(case_num),
                    graph=bool(graph),
                )

            if not isinstance(res, dict) or res.get("ok") is False:
                st.error("NARDL failed.")
                st.json(res)
            else:
                st.success("NARDL computed (R).")
                if "summary" in res and res["summary"]:
                    st.subheader("Model summary (R)")
                    st.text(res["summary"])
                st.subheader("Full output (JSON)")
                st.json(res)
                if "bounds" in res and res["bounds"]:
                    st.subheader("Bounds test (per selected case)")
                    st.json(res["bounds"])

        elif tool.startswith("APT"):
            if apt_y is None or apt_x is None:
                st.error("APT: select y and x.")
                st.stop()

            ensure_no_diff_for_levels([apt_y, apt_x], context="APT (asymmetric / threshold ECM)")

            y = get_series(df_work, apt_y, date_col=date_col)
            x = get_series(df_work, apt_x, date_col=date_col)

            mode = apt_mode_ui.split(" ")[0].strip()
            tar_model = apt_model_ui.split(" ")[0].strip()  # tar/mtar
            ecm_model = apt_ecm_model_ui.split(" ")[0].strip()  # linear/nonlinear
            which_diag = apt_diag_which.split(" ")[0].strip()  # sym/asy

            with st.spinner("Running APT via R..."):
                res = apt_via_r(
                    y=y,
                    x=x,
                    mode=mode,
                    model=tar_model,
                    lag=int(apt_lag),
                    thresh=float(apt_thresh),
                    maxlag=int(apt_maxlag),
                    adjust=bool(apt_adjust),
                    th_range=float(apt_th_range),
                    digits=int(apt_digits),
                    split=bool(apt_split),
                    which=which_diag,
                )

            if not isinstance(res, dict) or res.get("ok") is False:
                st.error("APT failed.")
                st.json(res)
            else:
                st.success("APT computed (R).")
                if "printed" in res and res["printed"]:
                    st.subheader("R print output")
                    st.text("\n".join(res["printed"]) if isinstance(res["printed"], list) else str(res["printed"]))
                st.subheader("Full output (JSON)")
                st.json(res)

        elif tool.startswith("GARCH"):
            var_map = {
                "sGARCH (standard GARCH)": "sGARCH",
                "eGARCH (exponential GARCH)": "eGARCH",
                "gjrGARCH (GJR / threshold GARCH)": "gjrGARCH",
            }
            variance_model = var_map[garch_var_ui]
            dist = dist_ui.split(" ")[0].strip()  # norm/std/ged

            with st.spinner("Running univariate GARCH via R..."):
                res = ugarch_via_r(
                    y=s,
                    variance_model=variance_model,
                    garch_p=int(garch_p),
                    garch_q=int(garch_q),
                    arma_p=int(arma_p),
                    arma_q=int(arma_q),
                    include_mean=bool(include_mean),
                    dist=dist,
                )

            if not isinstance(res, dict) or res.get("ok") is False:
                st.error("GARCH failed.")
                st.json(res)
            else:
                st.success("GARCH computed (R).")
                if "summary_text" in res and res["summary_text"]:
                    st.subheader("R output")
                    st.text(res["summary_text"])
                st.subheader("Full output (JSON)")
                st.json(res)

        elif tool.startswith("MGARCH"):
            if len(mgarch_cols) < 2:
                st.error("MGARCH (DCC): select at least 2 series.")
                st.stop()

            df_m = build_levels_df(df_work, mgarch_cols, date_col=date_col)

            var_map = {
                "sGARCH (standard GARCH)": "sGARCH",
                "eGARCH (exponential GARCH)": "eGARCH",
                "gjrGARCH (GJR / threshold GARCH)": "gjrGARCH",
            }
            variance_model = var_map[mgarch_var_ui]
            dist = dist_ui.split(" ")[0].strip()

            with st.spinner("Running DCC-MGARCH via R..."):
                res = dcc_mgarch_via_r(
                    df=df_m,
                    cols=mgarch_cols,
                    variance_model=variance_model,
                    garch_p=int(mgarch_p),
                    garch_q=int(mgarch_q),
                    dcc_order=(int(dcc_a), int(dcc_b)),
                    dist=dist,
                )

            if not isinstance(res, dict) or res.get("ok") is False:
                st.error("MGARCH failed.")
                st.json(res)
            else:
                st.success("MGARCH (DCC) computed (R).")
                if "summary_text" in res and res["summary_text"]:
                    st.subheader("R output")
                    st.text(res["summary_text"])
                st.subheader("Full output (JSON)")
                st.json(res)

    except Exception as e:
        st.error(f"Run failed: {e}")


st.divider()
st.caption("Next steps: Gregory–Hansen (cointegration with break), SVECM identification, forecasting & backtesting.")
