"""Shared look, data and models for the NileMart Streamlit app.

Every page imports this module, so the theme, the model colours and the fitted
models are identical everywhere. Data cleaning, the time split and the candidate
models come straight from the `nilemart` package, so the app always matches
what `python -m nilemart.train` puts into production.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics as M
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from nilemart import FEATURES, GROSS_MARGIN, TARGET
from nilemart.data import SALES_MAX_K, SALES_MIN_K, clean, load_raw, time_split
from nilemart.train import CANDIDATES

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/nilemart_weekly_sales.csv"
VISITORS = "store_visitors_k"
MARGIN = GROSS_MARGIN

# ----------------------------------------------------------------- palette
INK, MUTED, GREEN, LINE, PAPER = "#1d2b24", "#6a776f", "#245c43", "#dce5dc", "#f5f7f2"
GRID, AXIS = "#e4e9e2", "#c3cbc2"
NORMAL_C, HOLIDAY_C = GREEN, "#d98a2b"          # week type: dark green vs amber (large lightness gap)
UNDER_C, OVER_C = "#2a78d6", "#e34948"          # diverging: actual above / below the forecast
TRAIN_C, TEST_C = "#b9cbbf", GREEN
ACTUAL_C = INK

FEATURE_LABELS = {
    "ad_spend_k": "Ad spend",
    "discount_pct": "Discount",
    "is_holiday_period": "Holiday week",
    "competitor_promo": "Competitor promo",
    VISITORS: "Store visitors",
}


@dataclass
class ModelSpec:
    name: str
    algo: str
    inputs: str
    features: tuple
    estimator: object
    color: str
    role: str  # baseline | candidate | leak

    @property
    def usable(self) -> bool:
        return self.role != "leak"


# Notebook order. Each model keeps the same colour on every chart, whatever is filtered.
SPECS = {
    "Model A": ModelSpec("Model A", "Linear regression", "Ad spend only", ("ad_spend_k",),
                         LinearRegression(), "#eda100", "baseline"),
    "Model B": ModelSpec("Model B", "Linear regression", "4 plannable inputs", tuple(FEATURES),
                         CANDIDATES["linear_regression"], "#2a78d6", "candidate"),
    "Model C": ModelSpec("Model C", "Linear regression", "4 inputs + store visitors", (*FEATURES, VISITORS),
                         LinearRegression(), "#8a8f88", "leak"),
    "Decision tree": ModelSpec("Decision tree", "Decision tree", "4 plannable inputs", tuple(FEATURES),
                               CANDIDATES["decision_tree"], "#1baf7a", "candidate"),
    "Random forest": ModelSpec("Random forest", "Random forest", "4 plannable inputs", tuple(FEATURES),
                               CANDIDATES["random_forest"], "#eb6834", "candidate"),
}
ORDER = list(SPECS)
PLANNABLE = [k for k, s in SPECS.items() if s.usable]


def color_of(key: str) -> str:
    return SPECS[key].color if key in SPECS else ACTUAL_C


def color_scale(keys) -> alt.Scale:
    keys = list(keys)
    return alt.Scale(domain=keys, range=[color_of(k) for k in keys])


def dash_scale(keys) -> alt.Scale:
    keys = list(keys)
    return alt.Scale(domain=keys, range=[[5, 3] if k == "Model C" else [1, 0] for k in keys])


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def _read_path(path: str) -> pd.DataFrame:
    return load_raw(path)


@st.cache_data(show_spinner=False)
def _read_bytes(content: bytes) -> pd.DataFrame:
    return load_raw(BytesIO(content))


@st.cache_data(show_spinner=False)
def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    return clean(raw)


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw and cleaned weekly data: the uploaded CSV if there is one, else data/."""
    up = st.session_state.get("upload")
    if up is not None:
        try:
            raw = _read_bytes(up.getvalue())
            return raw, _clean(raw)
        except Exception as e:  # noqa: BLE001 - show any parsing problem, then fall back
            st.sidebar.error(f"Could not use the uploaded file: {e}. Showing the bundled data instead.")
    raw = _read_path(str(DATA_PATH))
    return raw, _clean(raw)


def test_weeks(data: pd.DataFrame) -> int:
    return int(min(st.session_state.get("test_weeks", 30), len(data) // 3))


def scores(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {
        "MAE": float(M.mean_absolute_error(y, p)),
        "RMSE": float(M.root_mean_squared_error(y, p)),
        "R²": float(M.r2_score(y, p)),
        "MAPE": float(np.mean(np.abs(e) / y) * 100),
        "Bias": float(e.mean()),
        "Within 10%": float(np.mean(np.abs(e) / y <= 0.10) * 100),
    }


@st.cache_resource(show_spinner="Fitting the models...")
def fit_models(data: pd.DataFrame, n_test: int):
    """Fit every model on the past weeks and score it on the latest `n_test` weeks."""
    train, test = time_split(data, n_test)
    out = {}
    for key, spec in SPECS.items():
        f = list(spec.features)
        m = clone(spec.estimator).fit(train[f], train[TARGET])
        p_test, p_train = m.predict(test[f]), m.predict(train[f])
        out[key] = {"model": m, "features": f, "pred": p_test, "train_pred": p_train,
                    "test": scores(test[TARGET], p_test), "train": scores(train[TARGET], p_train)}
    return train, test, out


@st.cache_resource(show_spinner="Refitting the models on every week...")
def fit_full(data: pd.DataFrame) -> dict:
    """Plannable models refit on all clean weeks, the way production refits the winner."""
    return {k: clone(SPECS[k].estimator).fit(data[list(SPECS[k].features)], data[TARGET]) for k in PLANNABLE}


@st.cache_data(show_spinner="Running the rolling backtest...")
def backtest(data: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Expanding-window backtest: train on everything before each block, test on the block."""
    fold_weeks = max(5, min(20, len(data) // (n_splits + 2)))
    rows = []
    for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits, test_size=fold_weeks).split(data), 1):
        a, b = data.iloc[tr], data.iloc[te]
        for key, spec in SPECS.items():
            f = list(spec.features)
            p = clone(spec.estimator).fit(a[f], a[TARGET]).predict(b[f])
            rows.append({"fold": i, "weeks": f"{int(b.week.min())}–{int(b.week.max())}", "train_weeks": len(a),
                         "model": key, "MAE": float(M.mean_absolute_error(b[TARGET], p))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Tuning the decision tree...")
def tree_grid(data: pd.DataFrame, n_test: int) -> tuple[pd.DataFrame, dict]:
    """The notebook's GridSearchCV for the tree: time-ordered CV on the training weeks only."""
    train, _ = time_split(data, n_test)
    grid = GridSearchCV(DecisionTreeRegressor(random_state=42),
                        {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [3, 5, 10]},
                        cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_absolute_error")
    grid.fit(train[FEATURES], train[TARGET])
    res = pd.DataFrame(grid.cv_results_["params"])
    res["cv_mae"] = -grid.cv_results_["mean_test_score"]
    return res, grid.best_params_


def importance(R: dict, data: pd.DataFrame, keys) -> pd.DataFrame:
    """Share of each input in a model's predictions.

    Trees: built-in feature importances. Linear models: |coefficient x std of the input|,
    i.e. how much a typical swing in that input moves the forecast, scaled to sum to 1.
    """
    rows = []
    for k in keys:
        m, f = R[k]["model"], R[k]["features"]
        if hasattr(m, "feature_importances_"):
            raw = np.asarray(m.feature_importances_, float)
        else:
            raw = np.abs(m.coef_) * data[f].std().to_numpy()
        share = raw / raw.sum() if raw.sum() else raw
        rows += [{"model": k, "feature": FEATURE_LABELS[c], "share": float(s)} for c, s in zip(f, share)]
    return pd.DataFrame(rows)


def best_plannable(R: dict) -> str:
    return min(PLANNABLE, key=lambda k: R[k]["test"]["MAE"])


def production():
    """(model, meta) saved by `python -m nilemart.train`, or (None, None) if not trained yet."""
    from nilemart.predict import DEFAULT_MODEL, load_model
    if not DEFAULT_MODEL.exists():
        return None, None
    return load_model(str(DEFAULT_MODEL))


# ----------------------------------------------------------------- charts
def style(chart):
    return (chart.configure(font="DM Sans", background="transparent")
            .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                            titleFontWeight="normal", gridColor=GRID, domainColor=AXIS, tickColor=AXIS,
                            labelPadding=6, titlePadding=10)
            .configure_axisX(grid=False)
            .configure_view(stroke=None)
            .configure_legend(orient="top", labelColor=INK, labelFontSize=12, symbolSize=110,
                              columnPadding=18, labelLimit=260)
            .configure_title(font="DM Sans", color=INK, fontSize=14, fontWeight=600, anchor="start", offset=10))


def show(chart, height: int | None = 320):
    if height is not None:
        chart = chart.properties(height=height)
    st.altair_chart(style(chart), width="stretch", theme=None)


def time_lines(wide: pd.DataFrame, series: list[str], y_title="Sales (EGP k)", height=340, extra=None):
    """Lines over weeks with a hover crosshair whose tooltip lists every series."""
    long = wide.melt("week", series, var_name="series", value_name="value")
    x = alt.X("week:Q", title="Week", axis=alt.Axis(format="d", tickMinStep=1), scale=alt.Scale(nice=False, zero=False))
    color = alt.Color("series:N", scale=color_scale(series), sort=series, title=None)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["week"], empty=False, clear="pointerout")
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)), color=color,
        strokeDash=alt.StrokeDash("series:N", scale=dash_scale(series), sort=series, title=None))
    dots = lines.mark_point(filled=True, size=70, stroke="white", strokeWidth=1.5).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    tips = [alt.Tooltip("week:Q", title="Week", format="d")] + [alt.Tooltip(f"{s}:Q", title=s, format=",.1f")
                                                                 for s in series]
    catcher = alt.Chart(wide).mark_rule(strokeWidth=14, opacity=0).encode(x=x, tooltip=tips).add_params(nearest)
    rule = alt.Chart(wide).mark_rule(color=AXIS, strokeWidth=1).encode(x=x).transform_filter(nearest)
    layers = ([extra] if extra is not None else []) + [lines, rule, dots, catcher]
    show(alt.layer(*layers), height)


def band(lo: float, hi: float, color=GREEN, opacity=0.08):
    return alt.Chart(pd.DataFrame({"a": [lo], "b": [hi]})).mark_rect(color=color, opacity=opacity).encode(
        x="a:Q", x2="b:Q")


def vline(x: float, color=MUTED, dash=(4, 4)):
    return alt.Chart(pd.DataFrame({"x": [x]})).mark_rule(color=color, strokeDash=list(dash)).encode(x="x:Q")


def hline(y: float, color=INK, dash=(1, 0), width=1):
    return alt.Chart(pd.DataFrame({"y": [y]})).mark_rule(color=color, strokeDash=list(dash),
                                                          strokeWidth=width).encode(y="y:Q")


def metric_bars(df: pd.DataFrame, metric: str, order: list[str], fmt: str, title: str, height=200):
    """Horizontal bar per model for one metric, value labelled at the end of each bar."""
    base = alt.Chart(df).encode(
        y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=140, ticks=False, domain=False)),
        x=alt.X(f"{metric}:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False),
                scale=alt.Scale(domain=[0, max(df[metric].max() * 1.22, 1e-9)])),
        tooltip=["model:N", alt.Tooltip(f"{metric}:Q", format=fmt)])
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.Color("model:N", scale=color_scale(order), legend=None))
    labels = base.mark_text(align="left", dx=6, color=INK, fontSize=12).encode(text=alt.Text(f"{metric}:Q", format=fmt))
    show((bars + labels).properties(title=title), height)


# ----------------------------------------------------------------- page furniture
def fmt_k(v: float, sign: bool = False) -> str:
    return f"EGP {v:+,.1f}k" if sign else f"EGP {v:,.1f}k"


def eyebrow(text: str):
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)


def hero(kicker: str, title: str, lead: str = ""):
    st.markdown(f'<div class="hero"><p class="eyebrow">{kicker}</p><h1>{title}</h1>'
                + (f'<p class="lead">{lead}</p>' if lead else "") + "</div>", unsafe_allow_html=True)


def section(kicker: str, title: str, caption: str = ""):
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    eyebrow(kicker)
    st.subheader(title)
    if caption:
        st.caption(caption)


def callout(html: str, kind: str = "decision"):
    st.markdown(f'<div class="callout {kind}">{html}</div>', unsafe_allow_html=True)


def badge(key: str, best: str | None = None) -> str:
    spec = SPECS[key]
    if spec.role == "leak":
        return "Leaks: not usable"
    if key == best:
        return "Best: in production"
    return "Baseline" if spec.role == "baseline" else "Candidate"


def legend_chips(keys) -> str:
    chips = "".join(f'<span class="chip"><i style="background:{color_of(k)}"></i>{k}</span>' for k in keys)
    return f'<div class="chips">{chips}</div>'


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Serif+Display&display=swap');

:root {
    --ink: #1d2b24; --muted: #6a776f; --green: #245c43; --green-soft: #e8efe6;
    --line: #dce5dc; --paper: #f5f7f2; --card: rgba(255,255,255,.86); --amber: #e6a25c;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], button, input, textarea {
    font-family: 'DM Sans', system-ui, sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 92% 0%, #e8eee2 0, transparent 32%), var(--paper);
    color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1240px; }

/* sidebar */
[data-testid="stSidebar"] { background: #203d30; border-right: 1px solid #345746; }
[data-testid="stSidebar"] :is(p, span, label, li, h1, h2, h3, small, summary, div[data-testid="stMarkdownContainer"]) { color: #eef0e6; }
[data-testid="stSidebarNavSeparator"] { border-color: rgba(255,255,255,.12); }
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] { color: #a9c2b3 !important; letter-spacing: .1em; }
[data-testid="stSidebarNavLink"] { border-radius: 6px; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(255,255,255,.14); }
[data-testid="stSidebarNavLink"] span { color: #eef0e6 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { border-color: rgba(255,255,255,.18); background: rgba(255,255,255,.03); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebar"] [data-testid="stExpander"] svg { fill: #eef0e6; color: #eef0e6; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.06); border: 1px dashed rgba(255,255,255,.3); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] :is(span, small, svg) { color: #cfd9d1 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background: transparent; color: #eef0e6; border-color: rgba(255,255,255,.35); }
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] { background: var(--amber); box-shadow: none; }
[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"], [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] { color: #cfd9d1 !important; }
.side-brand { padding: .2rem 0 .6rem; }
.side-brand b { font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size: 1.35rem; color: #f4f4e9; }
.side-brand span { display: block; font-size: .78rem; color: #a9c2b3; margin-top: .1rem; }

/* type */
h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
h1, .hero h1 { font-family: 'DM Serif Display', Georgia, serif !important; font-weight: 400 !important; }
h2, h3 { font-family: 'DM Sans', system-ui, sans-serif !important; }
h2 { font-weight: 700 !important; font-size: 1.6rem !important; }
h3 { font-weight: 600 !important; font-size: 1.25rem !important; }
[data-testid="stCaptionContainer"] { color: var(--muted); }
.eyebrow { color: var(--green); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 0 0 .4rem; }
.hero { margin-bottom: 1.2rem; }
.hero h1 { font-size: 2.6rem; line-height: 1.12; margin: 0 0 .5rem; padding: 0; }
.hero .lead { color: var(--muted); font-size: 1.05rem; max-width: 760px; line-height: 1.6; margin: 0; }
.section-rule { border-top: 1px solid var(--line); margin: 1.8rem 0 1.1rem; }

/* metric tiles */
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .85rem; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 600; }

/* callouts */
.callout { border-radius: 6px; padding: 16px 18px; line-height: 1.65; color: var(--ink); margin: .4rem 0 .8rem; border-left: 4px solid; }
.callout.decision { background: var(--green-soft); border-color: var(--green); }
.callout.warn { background: #fbf1e3; border-color: #d98a2b; }
.callout.leak { background: #fbe9e8; border-color: #d03b3b; }
.callout.info { background: #eaf1fa; border-color: #2a78d6; }
.callout .label { display: block; font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: .25rem; }

/* notebook task framing */
.goal { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: .3rem 0 1rem; }
.goal > div { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; line-height: 1.5; }
.goal span { display: block; font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--green); margin-bottom: .2rem; }
@media (max-width: 760px) { .goal { grid-template-columns: 1fr; } .hero h1 { font-size: 2rem; } }
.question { background: #fffdf7; border: 1px solid #eadfc8; border-radius: 8px; padding: 14px 16px; margin: 1.2rem 0 .4rem; line-height: 1.55; }
.question .qnum { display: inline-block; background: #f3e6cc; color: #7a5317; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; border-radius: 4px; padding: 2px 8px; margin-right: 8px; }
.code-note { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem; background: #eef2ec; border-radius: 6px; padding: 10px 12px; color: #33463c; white-space: pre-wrap; }

/* chips + cards */
.chips { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: .2rem 0 .6rem; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: .85rem; color: var(--ink); }
.chip i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 18px 18px 6px; min-height: 178px; }
.tile h4 { margin: 0 0 .3rem; font-size: 1.05rem; color: var(--ink); }
.tile p { color: var(--muted); font-size: .92rem; line-height: 1.5; margin: 0 0 .6rem; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: .4rem 0 .2rem; }
.flow span { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; font-size: .86rem; }
.flow em { color: var(--muted); font-style: normal; }

/* tabs, tables, pills */
[data-testid="stTabs"] button[role="tab"] p { font-weight: 500; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
[data-testid="stExpander"] details { background: var(--card); border-color: var(--line); }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
