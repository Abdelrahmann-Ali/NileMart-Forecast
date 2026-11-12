"""Every model from the notebook, compared on the same held-out weeks."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import nilemart_ui as ui
from nilemart import TARGET

raw, data = ui.get_data()
n_test = ui.test_weeks(data)
train, test, R = ui.fit_models(data, n_test)
best = ui.best_plannable(R)

ui.hero("Model comparison", "Which model should NileMart plan with?",
        f"Five models from the notebook, each trained on weeks {int(train.week.min())}–{int(train.week.max())} and "
        f"scored on the {n_test} most recent weeks ({int(test.week.min())}–{int(test.week.max())}), which none of "
        "them saw in training.")

# ----------------------------------------------------------------- filter row
f1, f2 = st.columns([3, 1])
with f1:
    picked = st.multiselect("Models to compare", ui.ORDER, default=[k for k in ui.ORDER if k != "Model C"],
                            help="Model C is off by default: it uses store visitors, which leak the answer.")
with f2:
    rank_by = st.selectbox("Rank by", ["MAE", "RMSE", "R²", "MAPE"], help="Lower is better, except R².")
if not picked:
    st.info("Pick at least one model.")
    st.stop()
keys = sorted(picked, key=lambda k: (-R[k]["test"][rank_by]) if rank_by == "R²" else R[k]["test"][rank_by])
st.markdown(ui.legend_chips(keys), unsafe_allow_html=True)

# ----------------------------------------------------------------- leaderboard
board = pd.DataFrame([{
    "Rank": i, "Model": k, "Inputs": ui.SPECS[k].inputs,
    "MAE": R[k]["test"]["MAE"], "RMSE": R[k]["test"]["RMSE"], "R²": R[k]["test"]["R²"],
    "MAPE %": R[k]["test"]["MAPE"], "Within ±10 %": R[k]["test"]["Within 10%"], "Bias": R[k]["test"]["Bias"],
    "Train MAE": R[k]["train"]["MAE"], "Overfit gap": R[k]["test"]["MAE"] - R[k]["train"]["MAE"],
    "Status": ui.badge(k, best)} for i, k in enumerate(keys, 1)])
num = st.column_config.NumberColumn
st.dataframe(board, hide_index=True, width="stretch", column_config={
    "Rank": num(width="small"),
    "MAE": num("MAE (k)", format="%.1f", help="Typical miss in EGP thousand"),
    "RMSE": num("RMSE (k)", format="%.1f", help="Punishes big misses more"),
    "R²": st.column_config.ProgressColumn("R²", format="%.3f", min_value=0, max_value=1),
    "MAPE %": num(format="%.1f%%", help="Average miss as a share of actual sales"),
    "Within ±10 %": num(format="%.0f%%", help="Share of test weeks forecast within 10% of actual"),
    "Bias": num("Bias (k)", format="%+.1f", help="Average actual − forecast. Positive = forecasts run low"),
    "Train MAE": num("Train MAE (k)", format="%.1f"),
    "Overfit gap": num("Test − train MAE", format="%+.1f", help="Large positive gap = memorised the training weeks"),
})

usable = [k for k in keys if ui.SPECS[k].usable]
if len(usable) >= 2:
    w, r = sorted(usable, key=lambda k: R[k]["test"]["MAE"])[:2]
    gap = R[r]["test"]["MAE"] - R[w]["test"]["MAE"]
    ui.callout(f'<span class="label">Verdict</span>Among the models usable for planning, <b>{w}</b> has the lowest '
               f'test error: <b>{ui.fmt_k(R[w]["test"]["MAE"])}</b> against {ui.fmt_k(R[r]["test"]["MAE"])} for '
               f'{r}, a {gap / R[r]["test"]["MAE"]:.0%} smaller typical miss. '
               + ("It is also the simplest to explain: one coefficient per input." if "Model B" == w else ""))
if "Model C" in keys:
    ui.callout("Model C ranks on raw scores only. It reads <b>store visitors</b>, known only after the week ends, "
               "so it cannot be used to plan next week.", "leak")

# ----------------------------------------------------------------- metric small multiples
ui.section("Scores", "The three notebook metrics side by side",
           "Same model order in every panel. MAE and RMSE in EGP thousand (lower is better); R² (higher is better).")
mdf = pd.DataFrame([{"model": k, **R[k]["test"]} for k in keys])
for col, (m, fmt, title) in zip(st.columns(3), [("MAE", ",.1f", "MAE · lower is better"),
                                               ("RMSE", ",.1f", "RMSE · lower is better"),
                                               ("R²", ".3f", "R² · higher is better")]):
    with col:
        ui.metric_bars(mdf.assign(**{m: mdf[m].clip(lower=0)}), m, keys, fmt, title, height=40 + 34 * len(keys))

# ----------------------------------------------------------------- deep dives
ui.section("Deep dive", "Where the models differ")
tabs = st.tabs(["Forecast vs actual", "Predicted vs actual", "Errors", "Head-to-head",
                "Rolling backtest", "Overfitting", "What drives each model"])

with tabs[0]:
    show_train = st.toggle("Also show the last 26 training weeks (in-sample fit)", value=False)
    wide = pd.DataFrame({"week": test["week"].values, "Actual": test[TARGET].values,
                         **{k: R[k]["pred"] for k in keys}})
    extra = None
    if show_train:
        tail = slice(-26, None)
        wide = pd.concat([pd.DataFrame({"week": train["week"].values[tail], "Actual": train[TARGET].values[tail],
                                        **{k: R[k]["train_pred"][tail] for k in keys}}), wide])
        extra = ui.band(test.week.min() - .5, test.week.max() + .5, opacity=.06)
    ui.time_lines(wide, ["Actual", *keys], height=380, extra=extra)
    st.caption("Hover for every model's forecast that week. Black = actual sales; dashed = Model C (leaks)."
               + (" Shaded = test weeks." if show_train else ""))

with tabs[1]:
    lo = min(test[TARGET].min(), min(R[k]["pred"].min() for k in keys)) * .95
    hi = max(test[TARGET].max(), max(R[k]["pred"].max() for k in keys)) * 1.03
    diag = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(color=ui.AXIS, strokeDash=[4, 4]).encode(
        x=alt.X("v:Q", scale=alt.Scale(domain=[lo, hi])), y=alt.Y("v:Q", scale=alt.Scale(domain=[lo, hi])))
    cols = st.columns(min(3, len(keys)))
    for i, k in enumerate(keys):
        d = pd.DataFrame({"week": test.week.values, "actual": test[TARGET].values, "forecast": R[k]["pred"]})
        pts = alt.Chart(d).mark_circle(size=60, color=ui.color_of(k), opacity=.9, stroke="white", strokeWidth=1).encode(
            x=alt.X("actual:Q", title="Actual (EGP k)", scale=alt.Scale(domain=[lo, hi])),
            y=alt.Y("forecast:Q", title="Forecast (EGP k)", scale=alt.Scale(domain=[lo, hi])),
            tooltip=["week:Q", alt.Tooltip("actual:Q", format=",.1f"), alt.Tooltip("forecast:Q", format=",.1f")])
        with cols[i % len(cols)]:
            ui.show((diag + pts).properties(title=f"{k} · R² {R[k]['test']['R²']:.2f}"), 260)
    st.caption("Dashed line = perfect forecast. Points above it were over-forecast, below it under-forecast.")

with tabs[2]:
    err = pd.concat([pd.DataFrame({"model": k, "week": test.week.values, "error": test[TARGET].values - R[k]["pred"]})
                     for k in keys])
    c1, c2 = st.columns(2)
    with c1:
        box = alt.Chart(err).mark_boxplot(size=18, extent=1.5, median={"color": "white"}, outliers={"size": 25}).encode(
            y=alt.Y("model:N", sort=keys, title=None), x=alt.X("error:Q", title="Actual − forecast (EGP k)"),
            color=alt.Color("model:N", scale=ui.color_scale(keys), legend=None))
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ui.INK).encode(x="x:Q")
        ui.show((zero + box).properties(title="Spread of weekly errors"), 60 + 40 * len(keys))
        st.caption("Narrow box centred on zero = consistent and unbiased.")
    with c2:
        err["abs"] = err["error"].abs()
        err["cumulative"] = err.groupby("model")["abs"].cumsum()
        cum = err.pivot(index="week", columns="model", values="cumulative").reset_index()
        ui.time_lines(cum, keys, y_title="Cumulative absolute error (EGP k)", height=60 + 40 * len(keys))
        st.caption("Steeper line = misses piling up faster. Jumps show the weeks a model got badly wrong.")
    summary = err.groupby("model").agg(mean_error=("error", "mean"), median_abs=("abs", "median"),
                                       worst_miss=("abs", "max"),
                                       under=("error", lambda e: int((e > 0).sum()))).reindex(keys)
    summary["worst week"] = [int(err[err.model == k].sort_values("abs").week.iloc[-1]) for k in keys]
    st.dataframe(summary.round(1).rename(columns={"mean_error": "mean error (k)", "median_abs": "median |error| (k)",
                                                  "worst_miss": "worst |error| (k)",
                                                  "under": f"weeks under-forecast (of {len(test)})"}),
                 width="stretch")

with tabs[3]:
    h1, h2 = st.columns(2)
    x = h1.selectbox("Model", keys, index=0, key="h2h_x")
    y = h2.selectbox("against", [k for k in keys if k != x] or keys, index=0, key="h2h_y")
    ex, ey = np.abs(test[TARGET].values - R[x]["pred"]), np.abs(test[TARGET].values - R[y]["pred"])
    d = pd.DataFrame({"week": test.week.values, "diff": ey - ex})
    d["closer"] = np.where(d["diff"] >= 0, x, y)
    wins = int((d["diff"] > 0).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Weeks {x} was closer", f"{wins} of {len(d)}")
    k2.metric(f"Weeks {y} was closer", f"{int((d['diff'] < 0).sum())} of {len(d)}")
    k3.metric(f"Avg advantage of {x}", ui.fmt_k(d["diff"].mean(), sign=True),
              help=f"Mean of |error {y}| − |error {x}|. Positive favours {x}.")
    pair = [x, y] if x != y else [x]
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("week:O", title="Week (test period)"),
        y=alt.Y("diff:Q", title=f"How much closer {x} was (EGP k)"),
        color=alt.Color("closer:N", title="Closer forecast", scale=ui.color_scale(pair)),
        tooltip=["week:O", alt.Tooltip("diff:Q", format="+.1f", title=f"|err {y}| − |err {x}|"), "closer:N"])
    ui.show(bars + ui.hline(0), 300)
    st.caption(f"Bars above zero: {x} was closer to actual sales that week; below zero: {y} was.")

with tabs[4]:
    bt = ui.backtest(data)
    bt = bt[bt.model.isin(keys)]
    c1, c2 = st.columns([3, 2])
    with c1:
        order = bt.drop_duplicates("fold").sort_values("fold")["weeks"].tolist()
        base = alt.Chart(bt).encode(
            x=alt.X("weeks:N", sort=order, title="Weeks forecast in each fold", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("MAE:Q", title="MAE (EGP k)"), color=alt.Color("model:N", scale=ui.color_scale(keys), sort=keys,
                                                                   title=None),
            strokeDash=alt.StrokeDash("model:N", scale=ui.dash_scale(keys), sort=keys, title=None),
            tooltip=["model:N", "weeks:N", "train_weeks:Q", alt.Tooltip("MAE:Q", format=",.1f")])
        ui.show(base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=70, stroke="white", strokeWidth=1.5),
                320)
    with c2:
        stab = bt.groupby("model")["MAE"].agg(["mean", "std", "min", "max"]).reindex(keys)
        winners = bt[bt.model.isin(usable)].loc[lambda d: d.groupby("fold")["MAE"].idxmin()].model.value_counts()
        stab["folds won"] = winners.reindex(keys).fillna(0).astype(int)
        st.dataframe(stab.round(1).rename(columns={"mean": "mean MAE", "std": "± std"}), width="stretch")
        st.caption("Folds won counts usable models only.")
    st.caption(f"Expanding window: each fold trains on every week before its block, then forecasts the block. "
               f"This repeats the honest past-to-future test {bt.fold.nunique()} times, so a model can't win on "
               "one lucky test window.")

with tabs[5]:
    gap = pd.DataFrame([{"model": k, "Train MAE": R[k]["train"]["MAE"], "Test MAE": R[k]["test"]["MAE"]} for k in keys])
    long = gap.melt("model", var_name="set", value_name="MAE")
    yax = alt.Y("model:N", sort=keys, title=None, axis=alt.Axis(ticks=False, domain=False))
    link = alt.Chart(gap).mark_rule(color=ui.AXIS, strokeWidth=3).encode(y=yax, x="Train MAE:Q", x2="Test MAE:Q")
    dots = alt.Chart(long).mark_point(filled=True, size=160, stroke="white", strokeWidth=1.5).encode(
        y=yax, x=alt.X("MAE:Q", title="MAE (EGP k)"), shape=alt.Shape("set:N", title=None,
                                                                   scale=alt.Scale(range=["circle", "diamond"])),
        color=alt.Color("model:N", scale=ui.color_scale(keys), legend=None),
        opacity=alt.Opacity("set:N", scale=alt.Scale(range=[.55, 1]), legend=None),
        tooltip=["model:N", "set:N", alt.Tooltip("MAE:Q", format=",.1f")])
    ui.show(link + dots, 60 + 46 * len(keys))
    st.caption("Circle = error on the weeks the model trained on; diamond = error on new weeks. "
               "A long bar means the model memorised the past: the trees fit training weeks far better than "
               "they forecast new ones, while the linear models score about the same on both.")

with tabs[6]:
    imp = ui.importance(R, data, keys)
    feats = list(ui.FEATURE_LABELS.values())
    c1, c2 = st.columns([3, 2])
    with c1:
        ui.show(alt.Chart(imp).mark_bar(cornerRadiusEnd=3, height=12).encode(
            y=alt.Y("feature:N", sort=feats, title=None), yOffset=alt.YOffset("model:N", sort=keys),
            x=alt.X("share:Q", title="Share of the model's predictions explained", axis=alt.Axis(format="%")),
            color=alt.Color("model:N", scale=ui.color_scale(keys), sort=keys, title=None),
            tooltip=["model:N", "feature:N", alt.Tooltip("share:Q", format=".1%")]), 360)
        st.caption("Trees: built-in feature importance. Linear models: coefficient × typical swing in the input, "
                   "scaled to 100%.")
    with c2:
        lin = [k for k in keys if hasattr(R[k]["model"], "coef_")]
        if lin:
            coefs = pd.DataFrame({k: pd.Series(R[k]["model"].coef_, index=[ui.FEATURE_LABELS[f] for f in R[k]["features"]])
                                  for k in lin}).reindex(feats).dropna(how="all")
            st.markdown("**Linear coefficients** (EGP k per unit)")
            st.dataframe(coefs.round(2), width="stretch")
        st.markdown("Every model leans on **holiday weeks** and **ad spend**. Model C hands most of the credit to "
                    "store visitors, which is why its ad effect looks tiny.")

# ----------------------------------------------------------------- tuning + downloads
with st.expander("How the decision tree's settings were chosen (GridSearchCV + TimeSeriesSplit)"):
    grid, params = ui.tree_grid(data, n_test)
    heat = alt.Chart(grid).encode(
        x=alt.X("min_samples_leaf:O", title="min_samples_leaf"), y=alt.Y("max_depth:O", title="max_depth", sort="descending"))
    rect = heat.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("cv_mae:Q", title="CV MAE (k)", scale=alt.Scale(range=["#cde2fb", "#184f95"])),
        tooltip=["max_depth:O", "min_samples_leaf:O", alt.Tooltip("cv_mae:Q", format=",.1f")])
    txt = heat.mark_text(fontSize=12).encode(
        text=alt.Text("cv_mae:Q", format=",.1f"),
        color=alt.condition(alt.datum.cv_mae > grid.cv_mae.median(), alt.value("white"), alt.value(ui.INK)))
    c1, c2 = st.columns([2, 3])
    with c1:
        ui.show(rect + txt, 230)
    with c2:
        st.markdown(f"Best: **max_depth = {params['max_depth']}**, **min_samples_leaf = {params['min_samples_leaf']}** "
                    f"(cross-validated MAE {grid.cv_mae.min():.1f}k). The search used only the training weeks, split "
                    "in time order, so the test weeks stayed untouched. Even the best tree loses to Model B on the "
                    "test weeks.")

preds = pd.DataFrame({"week": test.week.values, "actual": test[TARGET].values, **{k: R[k]["pred"] for k in keys}})
d1, d2, _ = st.columns([1, 1, 2])
d1.download_button("Download leaderboard (CSV)", board.to_csv(index=False), "model_leaderboard.csv", "text/csv",
                   width="stretch")
d2.download_button("Download test forecasts (CSV)", preds.to_csv(index=False), "test_forecasts.csv", "text/csv",
                   width="stretch")
