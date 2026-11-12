"""The model development notebook, task by task, with every answer computed from the live data."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import nilemart_ui as ui
from nilemart import FEATURES, TARGET

VIS = ui.VISITORS
raw, data = ui.get_data()
n_test = ui.test_weeks(data)
train, test, R = ui.fit_models(data, n_test)
A, B, C = R["Model A"], R["Model B"], R["Model C"]
slope_a = A["model"].coef_[0]
coef_b = pd.Series(B["model"].coef_, index=FEATURES)
slope_b = coef_b["ad_spend_k"]
ret_a, ret_b = slope_a * ui.MARGIN, slope_b * ui.MARGIN
filled = raw.loc[raw["ad_spend_k"].isna(), "week"].astype(int).tolist()
bad = raw[~raw[TARGET].between(ui.SALES_MIN_K, ui.SALES_MAX_K)]
week_type = np.where(data["is_holiday_period"] == 1, "Holiday week", "Normal week")
sc = data.assign(week_type=week_type)
HOL = alt.Scale(domain=["Normal week", "Holiday week"], range=[ui.NORMAL_C, ui.HOLIDAY_C])

STEPS = ["Explore", "Clean", "Relationship", "Split by time", "Model A", "Evaluate A", "Model B",
         "Compare & errors", "The leak", "Ad return", "Plan next week", "Memo", "Beyond linear"]


# ----------------------------------------------------------------- framing helpers
def task(n: int, title: str, goal: str, why: str):
    kicker = f"Notebook · Task {n} of 11" if n <= 11 else "Notebook · " + ("Final deliverable" if n == 12
                                                                           else "Extension")
    ui.eyebrow(kicker)
    st.header(title, anchor=False)
    st.markdown(f'<div class="goal"><div><span>Goal</span>{goal}</div><div><span>Why it matters</span>{why}</div></div>',
                unsafe_allow_html=True)


def question(n: int, text: str, answer: str):
    st.markdown(f'<div class="question"><span class="qnum">Question {n}</span>{text}</div>', unsafe_allow_html=True)
    with st.expander("Show the answer"):
        st.markdown(answer)


def code(text: str):
    st.markdown(f'<div class="code-note">{text}</div>', unsafe_allow_html=True)


def test_vs_pred(pred, name):
    wide = pd.DataFrame({"week": test["week"].values, "Actual": test[TARGET].values, name: pred})
    ui.time_lines(wide, ["Actual", name], height=300)


# ----------------------------------------------------------------- tasks
def t1():
    task(1, "Load and explore the data", "Get to know the data before you model anything.",
         "Most project failures start with data nobody looked at.")
    miss = raw.isna().sum()
    k = st.columns(4)
    k[0].metric("Rows × columns", f"{raw.shape[0]} × {raw.shape[1]}")
    k[1].metric("Missing values", int(miss.sum()), help="All in ad_spend_k")
    k[2].metric("Lowest weekly sales", f"EGP {raw[TARGET].min():,.0f}k")
    k[3].metric("Highest weekly sales", f"EGP {raw[TARGET].max():,.0f}k")

    tabs = st.tabs(["First rows", "Summary statistics", "Missing values", "Outliers (IQR rule)"])
    tabs[0].dataframe(raw.head(10), hide_index=True, width="stretch")
    tabs[1].dataframe(raw.describe().round(2), width="stretch")
    with tabs[2]:
        st.dataframe(miss[miss > 0].rename("missing"), width="content")
        st.caption(f"Weeks with no ad spend recorded: {', '.join(map(str, filled))}.")
    with tabs[3]:
        rows = []
        for col in ["ad_spend_k", "discount_pct", VIS, TARGET]:
            s = raw[col].dropna()
            q1, q3 = s.quantile([.25, .75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            out = raw[(raw[col] < lo) | (raw[col] > hi)]
            rows.append({"column": col, "lower bound": round(lo, 1), "upper bound": round(hi, 1),
                         "outliers": len(out), "weeks": ", ".join(map(str, out["week"]))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    plot = raw.assign(flag=np.where(raw[TARGET].between(ui.SALES_MIN_K, ui.SALES_MAX_K), "Plausible week",
                                    "Impossible value"))
    pts = alt.Chart(plot).mark_circle(size=60, opacity=.9).encode(
        x=alt.X("week:Q", title="Week", axis=alt.Axis(format="d")),
        y=alt.Y(f"{TARGET}:Q", title="Sales (EGP k, log scale)", scale=alt.Scale(type="log"),
                axis=alt.Axis(values=[30, 100, 300, 1000, 3000], format=",d")),
        color=alt.Color("flag:N", title=None, scale=alt.Scale(domain=["Plausible week", "Impossible value"],
                                                              range=[ui.TRAIN_C, ui.OVER_C])),
        tooltip=["week:Q", alt.Tooltip(f"{TARGET}:Q", title="sales_k"), alt.Tooltip(f"{VIS}:Q", title=VIS)])
    labels = alt.Chart(plot[plot.flag == "Impossible value"]).mark_text(dx=10, align="left", color=ui.INK).encode(
        x="week:Q", y=f"{TARGET}:Q", text=alt.Text("label:N")).transform_calculate(
        label="'Week ' + datum.week + ': ' + format(datum.sales_k, ',.1f') + 'k'")
    ui.show(pts + labels, 280)

    lo_w, hi_w = raw.loc[raw[TARGET].idxmin()], raw.loc[raw[TARGET].idxmax()]
    spv = (data[TARGET] / data[VIS])
    question(1, "Which column has missing values, and how many? Look at the minimum and maximum of "
                "<code>sales_k</code>. What looks wrong?",
             f"Only `ad_spend_k` has missing values: **{len(filled)} weeks** ({', '.join(map(str, filled))}).\n\n"
             f"`sales_k` runs from **{lo_w[TARGET]:,.1f}k** (week {int(lo_w.week)}) to **{hi_w[TARGET]:,.1f}k** "
             f"(week {int(hi_w.week)}), and both ends look wrong. Week {int(lo_w.week)} had "
             f"{lo_w[VIS]:.1f}k visitors but sold under 1 EGP per visitor, while every other week sells about "
             f"{spv.min():.0f}–{spv.max():.0f} EGP per visitor. Week {int(hi_w.week)} is roughly ten times too high. "
             f"The IQR check flags exactly these two weeks. They also pull the mean ({raw[TARGET].mean():.1f}k) "
             f"above the median ({raw[TARGET].median():.1f}k).")


def t2():
    task(2, "Clean the data", "Fix the missing values and remove the two broken weeks.",
         "A model learns from every row, including the broken ones.")
    k = st.columns(3)
    k[0].metric("Rows before", len(raw))
    k[1].metric("Rows after", len(data), delta=len(data) - len(raw))
    k[2].metric("Ad spend filled with the median", f"{len(filled)} weeks",
                help=f"Median = {raw['ad_spend_k'].median():.1f}k")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weeks removed**")
        st.dataframe(bad[["week", TARGET, VIS]].assign(**{"EGP per visitor": bad[TARGET] / bad[VIS]}).round(2),
                     hide_index=True, width="stretch")
    with c2:
        st.markdown("**Weeks filled**")
        st.dataframe(pd.DataFrame({"week": filled, "ad_spend_k (filled)": raw["ad_spend_k"].median()}),
                     hide_index=True, width="stretch")
    code("clean = df.copy()\nclean['ad_spend_k'] = clean['ad_spend_k'].fillna(clean['ad_spend_k'].median())\n"
         f"clean = clean[clean['sales_k'].between({ui.SALES_MIN_K}, {ui.SALES_MAX_K})]   "
         "# same rule as src/nilemart/data.py")
    question(2, "Why is it better to investigate these two weeks than to delete any unusual row automatically? "
                "Give a business example of an unusual week you should keep.",
             "An unusual value is not always a wrong value. Deleting every outlier automatically throws away the "
             "weeks that matter most for planning. Weeks 45 and 97 were removed only because Operations confirmed "
             "one was a system outage and the other a typing error.\n\n"
             "A week to keep: Eid al-Fitr or the last week of Ramadan, when sales jump far above a normal week. "
             "That spike is real demand; deleting it would make the model under-forecast every future holiday.")


def t3():
    task(3, "Look at the relationship", "See how ad spend and sales move together.",
         "A chart often reveals a business pattern before any model does.")
    ui.show(alt.Chart(sc).mark_circle(size=75, opacity=.85, stroke="white", strokeWidth=1).encode(
        x=alt.X("ad_spend_k:Q", title="Ad spend (EGP k)"), y=alt.Y(f"{TARGET}:Q", title="Sales (EGP k)"),
        color=alt.Color("week_type:N", scale=HOL, title=None),
        tooltip=["week:Q", "ad_spend_k:Q", f"{TARGET}:Q", "discount_pct:Q", "week_type:N"]), 380)
    means = sc.groupby("week_type")[["ad_spend_k", TARGET]].mean()
    st.dataframe(means.round(1).rename(columns={"ad_spend_k": "avg ad spend (k)", TARGET: "avg sales (k)"}),
                 width="content")
    h, n = means.loc["Holiday week"], means.loc["Normal week"]
    question(3, "Where do the holiday weeks sit on the chart? Could this affect how much credit we give to ads?",
             f"Holiday weeks sit **top-right**: the highest ad spend (about {h.ad_spend_k:.0f}k vs "
             f"{n.ad_spend_k:.0f}k in normal weeks) *and* the highest sales (about {h[TARGET]:.0f}k vs "
             f"{n[TARGET]:.0f}k). NileMart spends more on ads exactly when people shop more anyway.\n\n"
             "So yes: a line through all the points mixes two effects. An ads-only model cannot separate the "
             "holiday boost from the ad effect, so it gives ads credit for part of the holiday.")


def t4():
    task(4, f"Split by time", f"Train on the past and test on the most recent {n_test} weeks.",
         "A forecast is always about the future, so the test must look like the future.")
    split = data[["week", TARGET]].assign(set=["Train"] * len(train) + ["Test"] * len(test))
    ui.show(alt.Chart(split).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X("week:Q", title="Week", axis=alt.Axis(format="d")), y=alt.Y(f"{TARGET}:Q", title="Sales (EGP k)"),
        color=alt.Color("set:N", title=None, scale=alt.Scale(domain=["Train", "Test"], range=[ui.TRAIN_C, ui.TEST_C])),
        tooltip=["week:Q", f"{TARGET}:Q", "set:N"]), 230)
    k = st.columns(2)
    k[0].metric("Train", f"{len(train)} weeks", help=f"Weeks {int(train.week.min())}–{int(train.week.max())}")
    k[1].metric("Test", f"{len(test)} weeks", help=f"Weeks {int(test.week.min())}–{int(test.week.max())}")
    code(f"train = clean.iloc[:-{n_test}]\ntest  = clean.iloc[-{n_test}:]   # never shuffle")
    question(4, "What would go wrong if we shuffled the weeks at random before splitting?",
             "The test would no longer look like the future. Training weeks would sit on both sides of each test "
             "week, so the model would learn from week 140 and then be tested on week 139, and it would see "
             "almost every holiday season in training. Scores would look better than real use, where we always "
             "forecast a week we have not seen yet.")


def t5():
    task(5, "Build a simple model", "Predict sales from ad spend alone. This is your baseline, Model A.",
         "Start simple, so every added feature has to prove its value.")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.latex(rf"\text{{sales}} = {A['model'].intercept_:.1f} + {slope_a:.3f}\times\text{{ad spend}}")
        st.metric("Slope", f"{slope_a:.3f}", help="EGP k of sales per EGP k of ads")
        st.metric("Intercept", f"{A['model'].intercept_:.1f}")
    with c2:
        xs = pd.DataFrame({"ad_spend_k": np.linspace(data.ad_spend_k.min(), data.ad_spend_k.max(), 50)})
        xs["fit"] = A["model"].predict(xs[["ad_spend_k"]])
        pts = alt.Chart(sc).mark_circle(size=50, opacity=.75).encode(
            x=alt.X("ad_spend_k:Q", title="Ad spend (EGP k)"), y=alt.Y(f"{TARGET}:Q", title="Sales (EGP k)"),
            color=alt.Color("week_type:N", scale=HOL, title=None), tooltip=["week:Q", "ad_spend_k:Q", f"{TARGET}:Q"])
        line = alt.Chart(xs).mark_line(color=ui.color_of("Model A"), strokeWidth=3).encode(x="ad_spend_k:Q", y="fit:Q")
        ui.show(pts + line, 320)
        st.caption("Line = Model A's fit.")
    question(5, "Complete the sentence: “Each extra EGP 1,000 of ads goes with about EGP ____ more sales.” "
                "Why “goes with” and not “causes”?",
             f"“Each extra EGP 1,000 of ads goes with about **EGP {slope_a * 1000:,.0f}** more sales.” "
             f"(slope = {slope_a:.2f})\n\nThe model only measures how ads and sales move together in past data. "
             "Something else can push both up at once: in holiday weeks NileMart spends more on ads *and* "
             "customers buy more anyway. Proving cause needs a controlled experiment.")


def t6():
    task(6, "Evaluate Model A", f"Measure how far the forecast misses on the {n_test} test weeks.",
         "Managers plan stock and staff around the size of the error.")
    m = A["test"]
    k = st.columns(3)
    k[0].metric("MAE", ui.fmt_k(m["MAE"]))
    k[1].metric("RMSE", ui.fmt_k(m["RMSE"]))
    k[2].metric("R²", f"{m['R²']:.3f}")
    test_vs_pred(A["pred"], "Model A")
    question(6, "Explain your MAE to the marketing manager in one sentence, in EGP and with no technical words.",
             f"On a typical week, this forecast is off from actual sales by about "
             f"**{m['MAE'] * 1000:,.0f} EGP**, either above or below.")


def t7():
    task(7, "Add the features she can plan for",
         "Build Model B with ad spend, discount, holiday period and competitor promotion.",
         "Real decisions depend on several factors at once.")
    c1, c2 = st.columns([3, 2])
    with c1:
        tbl = pd.DataFrame({
            "coefficient": coef_b.round(2),
            "in plain words": [f"{slope_b:+.2f}k sales per extra 1k of ads",
                               f"{coef_b['discount_pct']:+.2f}k per extra 1% of discount",
                               f"{coef_b['is_holiday_period']:+.1f}k in a holiday week",
                               f"{coef_b['competitor_promo']:+.1f}k when the competitor runs a promo"]})
        st.dataframe(tbl, width="stretch")
        m = B["test"]
        k = st.columns(3)
        k[0].metric("MAE", ui.fmt_k(m["MAE"]), delta=f"{m['MAE'] - A['test']['MAE']:+.1f}k vs A",
                    delta_color="inverse")
        k[1].metric("RMSE", ui.fmt_k(m["RMSE"]), delta=f"{m['RMSE'] - A['test']['RMSE']:+.1f}k vs A",
                    delta_color="inverse")
        k[2].metric("R²", f"{m['R²']:.3f}", delta=f"{m['R²'] - A['test']['R²']:+.3f} vs A")
    with c2:
        sl = pd.DataFrame({"model": ["Model A", "Model B"], "slope": [slope_a, slope_b]})
        ui.metric_bars(sl, "slope", ["Model A", "Model B"], ".2f", "Sales per 1k of ads (EGP k)", height=150)
        st.caption(f"The ad slope falls from {slope_a:.2f} to {slope_b:.2f} once holidays get their own coefficient.")
    question(7, "Interpret the <code>is_holiday_period</code> and <code>competitor_promo</code> coefficients in "
                "plain business language.",
             f"- **`is_holiday_period` = {coef_b['is_holiday_period']:+.0f}:** in a Ramadan, Eid or year-end week, "
             f"sales are about **EGP {abs(coef_b['is_holiday_period']) * 1000:,.0f} higher** than a normal week "
             "with the same ad spend, discount and competitor activity.\n"
             f"- **`competitor_promo` = {coef_b['competitor_promo']:+.0f}:** when the main competitor runs a "
             f"promotion, NileMart sells about **EGP {abs(coef_b['competitor_promo']) * 1000:,.0f} less**, "
             "everything else equal.")
    question(8, "The ad spend slope dropped between Model A and Model B. Using your chart from Task 3, explain why.",
             f"The slope fell from {slope_a:.2f} to {slope_b:.2f} because Model A was giving ads credit for the "
             "holiday boost. Holiday weeks have both the biggest budgets and the biggest sales, and with ads as "
             "its only input Model A could explain them only through ad spend. Model B has a separate holiday "
             f"feature, so what remains for ads is a smaller, more honest effect: about "
             f"EGP {slope_b * 1000:,.0f} more sales per extra EGP 1,000.")


def t8():
    task(8, "Compare the models and inspect the errors",
         "Put both models side by side, then look at where Model B still misses.",
         "A single score can hide a repeating problem.")
    err = pd.DataFrame({"week": test["week"].values, "error": test[TARGET].values - B["pred"]})
    err["direction"] = np.where(err["error"] >= 0, "Under-predicted (actual above)", "Over-predicted (actual below)")
    c1, c2 = st.columns([3, 2])
    with c1:
        bars = alt.Chart(err).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("week:O", title="Week (test period)"), y=alt.Y("error:Q", title="Actual − predicted (EGP k)"),
            color=alt.Color("direction:N", title=None, scale=alt.Scale(
                domain=["Under-predicted (actual above)", "Over-predicted (actual below)"],
                range=[ui.UNDER_C, ui.OVER_C])),
            tooltip=["week:O", alt.Tooltip("error:Q", format="+.1f")])
        ui.show(bars + ui.hline(0), 300)
    with c2:
        res = pd.DataFrame({k: {m: R[k]["test"][m] for m in ["MAE", "RMSE", "R²"]} for k in ["Model A", "Model B"]}).T
        st.dataframe(res.round(3), width="stretch")
        worst = err.reindex(err["error"].abs().sort_values(ascending=False).index).head(3)
        st.markdown("**Largest misses**")
        for _, r in worst.iterrows():
            tag = " · ad spend was filled with the median" if int(r.week) in filled else ""
            st.markdown(f"- Week {int(r.week)}: {r.error:+.1f}k{tag}")
    a, b = A["test"], B["test"]
    worst_filled = [int(w) for w in worst.week if int(w) in filled]
    question(9, "By how much did Model B reduce the MAE compared with Model A? Do the errors look random around "
                "zero, or do you see a pattern?",
             f"Model B cut the MAE from **{a['MAE']:.1f}k to {b['MAE']:.1f}k EGP**, a drop of "
             f"**{a['MAE'] - b['MAE']:.1f}k ({(a['MAE'] - b['MAE']) / a['MAE']:.0%})**. RMSE fell from "
             f"{a['RMSE']:.1f}k to {b['RMSE']:.1f}k and R² rose from {a['R²']:.2f} to {b['R²']:.2f}.\n\n"
             f"The errors are mostly around zero but lean slightly positive: {(err.error > 0).sum()} of {len(err)} "
             f"weeks are under-predicted and the average error is {err.error.mean():+.1f}k."
             + (f" The biggest misses include week(s) {', '.join(map(str, worst_filled))}, whose missing ad spend "
                "was filled with the median: the cleaning choice shows up directly in the errors."
                if worst_filled else ""))
    st.page_link("views/compare.py", label="See all five models compared", icon=":material/leaderboard:")



def t9():
    task(9, "Spot the leak", "Add <code>store_visitors_k</code> to Model B, name it Model C, then decide whether to use it.",
         "Leakage produces great scores in the lab and useless forecasts in real life.")
    k = st.columns(4)
    k[0].metric("Model B R²", f"{B['test']['R²']:.3f}")
    k[1].metric("Model C R²", f"{C['test']['R²']:.3f}", delta=f"{C['test']['R²'] - B['test']['R²']:+.3f}")
    k[2].metric("Model B MAE", ui.fmt_k(B["test"]["MAE"]))
    k[3].metric("Model C MAE", ui.fmt_k(C["test"]["MAE"]), delta=f"{C['test']['MAE'] - B['test']['MAE']:+.1f}k",
                delta_color="inverse")
    spv = (data[TARGET] / data[VIS]).mean()
    c1, c2 = st.columns([3, 2])
    with c1:
        ui.show(alt.Chart(sc).mark_circle(size=55, opacity=.8).encode(
            x=alt.X(f"{VIS}:Q", title="Store visitors (k), known only after the week"),
            y=alt.Y(f"{TARGET}:Q", title="Sales (EGP k)"), color=alt.Color("week_type:N", scale=HOL, title=None),
            tooltip=["week:Q", f"{VIS}:Q", f"{TARGET}:Q"]), 300)
    with c2:
        ui.callout(f"<b>Don't plan with Model C.</b> Sales run about <b>{spv:.1f} EGP per visitor</b> every week, "
                   "so visitors are almost the answer itself. The chart is close to a straight line.", "leak")
        st.caption(f"Model C's ad coefficient: {C['model'].coef_[0]:.2f} (vs {slope_b:.2f} in Model B).")
    question(10, "Model C looks much better. Should the manager use it to plan next week's budget?",
             f"No. Model C's score (R² {C['test']['R²']:.2f}, MAE {C['test']['MAE']:.1f}k) is not achievable in "
             "real use because it relies on **store visitors, which are only known after the week ends**. When "
             "she sets next week's budget she knows the ad spend, discount, holiday calendar and maybe the "
             "competitor's plans, but not how many people will walk in. That is **data leakage**.\n\n"
             f"It would also mislead her about ads: with visitors included, the ad slope drops to about "
             f"{C['model'].coef_[0]:.2f}, because ads work mostly *by* bringing visitors in. She should plan with Model B.")


def t10():
    task(10, "Is one more EGP of ads worth it?", "Turn the slope into a money decision.",
         f"Sales are not profit. NileMart keeps a {ui.MARGIN:.0%} gross margin on sales.")
    roi = pd.DataFrame({"model": ["Model A", "Model B"], "return": [ret_a, ret_b]})
    c1, c2 = st.columns([3, 2])
    with c1:
        base = alt.Chart(roi).encode(y=alt.Y("model:N", title=None, axis=alt.Axis(ticks=False, domain=False)),
                                     x=alt.X("return:Q", title="Gross profit per EGP of ads",
                                             scale=alt.Scale(domain=[0, max(1.25, roi["return"].max() * 1.15)])))
        bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
            color=alt.Color("model:N", legend=None, scale=ui.color_scale(["Model A", "Model B"])),
            tooltip=["model:N", alt.Tooltip("return:Q", format=".3f")])
        txt = base.mark_text(align="left", dx=6, color=ui.INK).encode(text=alt.Text("return:Q", format=".2f"))
        even = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=ui.OVER_C, strokeDash=[6, 4], size=2).encode(x="x:Q")
        even_t = alt.Chart(pd.DataFrame({"x": [1], "t": ["break-even"]})).mark_text(
            align="left", dx=4, dy=-60, color=ui.OVER_C).encode(x="x:Q", text="t:N")
        ui.show(bars + txt + even + even_t, 170)
    with c2:
        for name, r in [("Model A", ret_a), ("Model B", ret_b)]:
            st.metric(f"{name} return per EGP", f"{r:.3f}", delta="pays for itself" if r >= 1 else "loses money",
                      delta_color="normal" if r >= 1 else "inverse")
    code(f"return = slope × margin    # Model B: {slope_b:.3f} × {ui.MARGIN} = {ret_b:.3f}")
    question(11, "What decision does each model suggest? Which one do you trust more, and why? What test could "
                 "the company run to measure the true effect of ads?",
             f"- **Model A** (return {ret_a:.2f}): ads " + ("roughly pay for themselves" if ret_a >= 1 else "lose money")
             + ".\n"
             f"- **Model B** (return {ret_b:.2f}): each extra EGP returns about {ret_b:.2f} EGP, so it suggests "
             + ("raising" if ret_b >= 1 else "**not** raising") + " the budget.\n\n"
             "Model B is more trustworthy: it separates the holiday effect from the ad effect and halves the test "
             "error. It is still a correlation, so the company should run a **controlled experiment**: randomly "
             "give some weeks or regions extra ad spend, keep the rest normal, and compare sales.")


def t11():
    task(11, "Plan next week", "Forecast sales under three budgets, then try one extreme budget.",
         "This is how the model gets used in the planning meeting.")
    plans = pd.DataFrame({"ad_spend_k": [60, 80, 100], "discount_pct": 10, "is_holiday_period": 0,
                          "competitor_promo": 1})
    plans["forecast"] = B["model"].predict(plans[FEATURES])
    extreme = plans.iloc[[0]].assign(ad_spend_k=400)
    ext = float(B["model"].predict(extreme[FEATURES])[0])
    max_ad = train["ad_spend_k"].max()
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(plans.assign(**{"profit after ads": plans.forecast * ui.MARGIN - plans.ad_spend_k}).round(1),
                     hide_index=True, width="stretch")
        st.metric("Extreme plan: 400k of ads", ui.fmt_k(ext))
        st.metric("Largest ad spend in training", ui.fmt_k(max_ad))
    with c2:
        grid = pd.DataFrame({"ad_spend_k": np.arange(0, 410, 10), "discount_pct": 10, "is_holiday_period": 0,
                             "competitor_promo": 1})
        grid["forecast"] = B["model"].predict(grid[FEATURES])
        line = alt.Chart(grid).mark_line(color=ui.color_of("Model B"), strokeWidth=2.5).encode(
            x=alt.X("ad_spend_k:Q", title="Ad spend (EGP k)"), y=alt.Y("forecast:Q", title="Forecast sales (EGP k)"),
            tooltip=[alt.Tooltip("ad_spend_k:Q"), alt.Tooltip("forecast:Q", format=",.1f")])
        pts = alt.Chart(pd.concat([plans, extreme.assign(forecast=ext)])).mark_point(
            filled=True, size=90, color=ui.INK).encode(x="ad_spend_k:Q", y="forecast:Q",
                                                      tooltip=["ad_spend_k:Q", alt.Tooltip("forecast:Q", format=",.1f")])
        ui.show(ui.band(train.ad_spend_k.min(), max_ad) + line + pts, 300)
        st.caption("Shaded band = budgets seen in training. Beyond it the model just extends a straight line.")
    question(12, "Would you trust the forecast for 400 thousand in ads?",
             f"No. The model forecasts about **{ext:,.0f}k** of sales for 400k of ads, but the largest ad spend it "
             f"saw in training was **{max_ad:.1f}k**. A regression line is only reliable inside the range it "
             "learned from; beyond that it simply extends the straight line. Real ads have diminishing returns "
             f"(audiences saturate, shoppers only buy so much food), and 400k is {400 / max_ad:.1f}× anything "
             "observed. The forecast is a guess that is almost certainly too high.")
    st.page_link("views/planner.py", label="Try your own plan in the weekly planner", icon=":material/event_note:")


def t12():
    task(12, "Your recommendation", "A memo to the marketing manager in five sentences at most.",
         "The model is only useful if the decision-maker can act on it.")
    ui.callout(
        f"I recommend <b>Model B</b>, which forecasts weekly sales from ad spend, discount, holidays and competitor "
        f"promotions; it is typically off by about <b>EGP {B['test']['MAE'] * 1000:,.0f}</b> a week. "
        f"Each extra EGP 1,000 of ads goes with about EGP {slope_b * 1000:,.0f} more sales, which at our "
        f"{ui.MARGIN:.0%} margin returns only <b>{ret_b:.2f} EGP in profit per EGP spent</b>, so raising the ad "
        f"budget {'pays' if ret_b >= 1 else 'does not pay'} for itself. The main limitation is that this is a "
        f"correlation, and the model has only seen budgets up to about EGP {train.ad_spend_k.max():,.0f}k. "
        "The next step is a controlled test that randomly varies ad spend across weeks or regions. "
        "Until then, keep the ad budget at its current level.")
    st.markdown("**The memo covers the four required points**")
    st.markdown("1. Recommended model and its typical error in EGP\n2. What it says about spending more on ads\n"
                "3. One risk or limitation\n4. One next step")


def t13():
    task(13, "Beyond linear regression, and saving the best model",
         "Check whether a decision tree or random forest beats Model B, then save the winner.",
         "A flexible model is only better if it forecasts new weeks better.")
    keys = ["Model B", "Decision tree", "Random forest"]
    tbl = pd.DataFrame({k: {"Test MAE": R[k]["test"]["MAE"], "Test RMSE": R[k]["test"]["RMSE"],
                            "Test R²": R[k]["test"]["R²"], "Train MAE": R[k]["train"]["MAE"]} for k in keys}).T
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(tbl.round(3), width="stretch")
        _, params = ui.tree_grid(data, n_test)
        st.caption(f"Tree settings chosen with GridSearchCV + TimeSeriesSplit on the training weeks: "
                   f"max_depth={params['max_depth']}, min_samples_leaf={params['min_samples_leaf']}.")
        best = min(keys, key=lambda k: R[k]["test"]["MAE"])
        st.markdown(f"The trees fit the training weeks more closely but miss new weeks by more: they memorise past "
                    f"weeks instead of learning the pattern. **{best}** has the lowest test error.")
    with c2:
        st.markdown("**Saving the best model** (what `src/nilemart/train.py` does)")
        code("1. score each candidate on the last test weeks\n2. pick the lowest MAE\n"
             "3. refit it on ALL clean weeks\n4. joblib.dump(model + metadata)")
        _, meta = ui.production()
        if meta:
            st.metric("Saved model", meta["model"].replace("_", " "),
                      help=f"Trained {meta['trained_at']} on {meta['training_rows']} weeks")
    st.page_link("views/compare.py", label="Open the full model comparison", icon=":material/leaderboard:")


RENDER = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]

# ----------------------------------------------------------------- stepper
if "nb_step" not in st.session_state:
    st.session_state.nb_step = st.session_state.get("_nb_last", 1)


def _keep():
    if st.session_state.nb_step is None:
        st.session_state.nb_step = st.session_state.get("_nb_last", 1)
    st.session_state._nb_last = st.session_state.nb_step


def _go(delta: int):
    st.session_state.nb_step = min(max((st.session_state.nb_step or 1) + delta, 1), len(STEPS))
    st.session_state._nb_last = st.session_state.nb_step


ui.hero("Notebook walkthrough", "Forecasting weekly sales at NileMart",
        "The model development notebook, one task at a time. Numbers and answers are recomputed from the data, "
        "so they update if you upload new weeks or change the test window in the sidebar.")
st.pills("Step", options=list(range(1, len(STEPS) + 1)), format_func=lambda i: f"{i} · {STEPS[i - 1]}",
         key="nb_step", on_change=_keep, label_visibility="collapsed")
step = st.session_state.nb_step or 1
st.progress(step / len(STEPS))
st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

RENDER[step - 1]()

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
b1, mid, b2 = st.columns([1, 3, 1])
b1.button("← Previous", on_click=_go, args=(-1,), disabled=step == 1, width="stretch")
mid.caption(f"<div style='text-align:center'>Step {step} of {len(STEPS)}</div>", unsafe_allow_html=True)
b2.button("Next →", on_click=_go, args=(1,), disabled=step == len(STEPS), width="stretch")
