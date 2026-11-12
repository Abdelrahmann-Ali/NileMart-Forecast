"""Weekly planner: forecast next week with the production model and check it against the other models."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import nilemart_ui as ui
from nilemart.predict import Plan, forecast

_, meta = ui.production()
if meta is None:
    st.error("No saved model found. Run `python start.py train` (or `python -m nilemart.train`) first.")
    st.stop()

raw, data = ui.get_data()
n_test = ui.test_weeks(data)
_, _, R = ui.fit_models(data, n_test)
full = ui.fit_full(data)
lo_ad, hi_ad = meta["feature_ranges"]["ad_spend_k"]
M = meta["gross_margin"]

ui.hero("Commercial planning", "Next week, at a glance",
        f"Forecasts come from the production model ({meta['model'].replace('_', ' ')}, refit on all "
        f"{meta['training_rows']} clean weeks). Its typical error on unseen weeks is "
        f"EGP {meta['selected_test_mae_k']:.0f}k.")

left, right = st.columns([1, 2.6], gap="large")
with left:
    ui.eyebrow("Next week's plan")
    with st.container(border=True):
        ad = st.slider("Ad spend (EGP thousand)", 0, 200, 80, step=5)
        disc = st.select_slider("Average discount", [0, 5, 10, 15, 20], value=10, format_func=lambda v: f"{v}%")
        hol = st.toggle("Holiday week (Ramadan, Eid, year-end)")
        comp = st.toggle("Competitor running a promotion", value=True)
        st.caption(f"Ad budgets seen in training: {lo_ad:.0f}k – {hi_ad:.0f}k.")

plan = Plan(ad, disc, int(hol), int(comp))
res = forecast(plan)
sales = res["forecast_sales_k"]

with right:
    ui.eyebrow("Your forecast")
    c1, c2 = st.columns(2)
    c1.metric("Expected sales", ui.fmt_k(sales), help="Production model forecast")
    c2.metric("Likely range", f"{res['range_k'][0]:,.0f}k – {res['range_k'][1]:,.0f}k",
              help="Forecast ± the typical error on held-out weeks")
    c3, c4 = st.columns(2)
    c3.metric("Gross profit", ui.fmt_k(sales * M), help=f"Sales × {M:.0%} margin")
    c4.metric("Profit after ads", ui.fmt_k(sales * M - ad), help="Gross profit minus the ad budget")
    for w in res["warnings"]:
        ui.callout(f"<b>Outside the model's experience:</b> {w}. Treat this forecast as a rough estimate.", "warn")
    if "profit_per_extra_egp_of_ads" in res:
        r = res["profit_per_extra_egp_of_ads"]
        ui.callout(f'<span class="label">Budget signal</span>Each additional EGP 1 in ads is associated with '
                   f'<b>EGP {r:.2f}</b> in gross profit at a {M:.0%} margin, so more ad spend '
                   f'<b>{"pays for itself" if r >= 1 else "does not pay for itself"}</b>. Every extra EGP 10k of ads '
                   f'changes profit after ads by about <b>EGP {(r - 1) * 10:+.1f}k</b>.')

# ----------------------------------------------------------------- scenario analysis
ui.section("Scenario analysis", "How does the budget change the outcome?",
           "Production model, holding your discount and week conditions constant.")
budgets = np.arange(0, 205, 5)
grid = pd.DataFrame([{"ad_spend_k": b, "sales": forecast(Plan(b, disc, int(hol), int(comp)))["forecast_sales_k"]}
                     for b in budgets])
grid["Profit after ads"] = grid["sales"] * M - grid["ad_spend_k"]
grid["Gross profit"] = grid["sales"] * M
c1, c2 = st.columns([1.4, 1])
with c1:
    long = grid.melt("ad_spend_k", ["Gross profit", "Profit after ads"], var_name="series", value_name="k")
    lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("ad_spend_k:Q", title="Ad spend (EGP k)"), y=alt.Y("k:Q", title="EGP k"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["Gross profit", "Profit after ads"],
                                                                range=["#9fb7a8", ui.GREEN])),
        tooltip=["series:N", "ad_spend_k:Q", alt.Tooltip("k:Q", format=",.1f")])
    here = alt.Chart(pd.DataFrame({"x": [ad], "y": [sales * M - ad]})).mark_point(
        filled=True, size=110, color=ui.INK).encode(x="x:Q", y="y:Q")
    ui.show(ui.band(lo_ad, hi_ad) + lines + ui.vline(ad) + here, 320)
    st.caption("Shaded band = budgets seen in training. Dot = your plan.")
with c2:
    table = grid[grid.ad_spend_k.isin(sorted(set(range(40, 160, 20)) | {ad}))].copy()
    table["vs your plan"] = table["Profit after ads"] - (sales * M - ad)
    st.dataframe(table.rename(columns={"ad_spend_k": "Ad spend (k)", "sales": "Sales (k)"})
                 [["Ad spend (k)", "Sales (k)", "Profit after ads", "vs your plan"]].round(1),
                 hide_index=True, width="stretch",
                 column_config={"vs your plan": st.column_config.NumberColumn(format="%+.1f")})

# ----------------------------------------------------------------- model cross-check
ui.section("Model cross-check", "What would each model forecast for this plan?",
           "Every plannable model refit on all clean weeks. Model C is left out: it needs next week's store visitors.")
frame = plan.as_frame()
check = pd.DataFrame([{"model": k, "forecast": float(full[k].predict(frame[list(ui.SPECS[k].features)])[0]),
                       "error": R[k]["test"]["MAE"]} for k in ui.PLANNABLE])
check["lo"], check["hi"] = check.forecast - check.error, check.forecast + check.error
keys = list(check.sort_values("error").model)
c1, c2 = st.columns([1, 1.3])
with c1:
    yax = alt.Y("model:N", sort=keys, title=None, axis=alt.Axis(ticks=False, domain=False))
    rng = alt.Chart(check).mark_rule(strokeWidth=3, opacity=.35).encode(
        y=yax, x=alt.X("lo:Q", title="Forecast sales (EGP k) ± typical error", scale=alt.Scale(zero=False)), x2="hi:Q",
        color=alt.Color("model:N", scale=ui.color_scale(keys), legend=None))
    dot = alt.Chart(check).mark_point(filled=True, size=140, stroke="white", strokeWidth=1.5).encode(
        y=yax, x="forecast:Q", color=alt.Color("model:N", scale=ui.color_scale(keys), legend=None),
        tooltip=["model:N", alt.Tooltip("forecast:Q", format=",.1f"), alt.Tooltip("error:Q", format=",.1f",
                                                                                   title="typical error")])
    ui.show(rng + dot + ui.vline(sales, color=ui.GREEN), 60 + 44 * len(keys))
    st.caption("Line = each model's forecast ± its typical test error. Green rule = production forecast.")
with c2:
    curve = pd.DataFrame({"ad_spend_k": budgets, "discount_pct": disc, "is_holiday_period": int(hol),
                          "competitor_promo": int(comp)})
    wide = pd.DataFrame({"week": budgets, **{k: full[k].predict(curve[list(ui.SPECS[k].features)]) for k in keys}})
    long = wide.melt("week", keys, var_name="model", value_name="sales")
    resp = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("week:Q", title="Ad spend (EGP k)"), y=alt.Y("sales:Q", title="Forecast sales (EGP k)",
                                                            scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=ui.color_scale(keys), sort=keys, title=None),
        tooltip=["model:N", alt.Tooltip("week:Q", title="ad spend"), alt.Tooltip("sales:Q", format=",.1f")])
    ui.show(ui.band(lo_ad, hi_ad) + resp + ui.vline(ad), 300)
    st.caption("How each model responds to the ad budget. Trees go flat outside the training band and the linear "
               "models keep climbing in a straight line: neither is trustworthy there.")
spread = check.loc[check.model != "Model A", "forecast"]
if spread.max() - spread.min() > meta["selected_test_mae_k"]:
    ui.callout(f"The models disagree by <b>{spread.max() - spread.min():,.0f}k</b> on this plan, more than the "
               "typical error. Treat the forecast with extra caution.", "warn")
else:
    ui.callout(f"The candidate models agree within <b>{spread.max() - spread.min():,.0f}k</b> on this plan, inside "
               "the typical error.", "info")

# ----------------------------------------------------------------- saved scenarios
ui.section("Compare plans", "Line up the options for the planning meeting")
st.session_state.setdefault("scenarios", [])
b1, b2, _ = st.columns([1, 1, 3])
if b1.button("Add this plan", icon=":material/add:", width="stretch"):
    st.session_state.scenarios.append({
        "Plan": f"#{len(st.session_state.scenarios) + 1}", "Ad spend (k)": ad, "Discount %": disc,
        "Holiday": "Yes" if hol else "No", "Competitor promo": "Yes" if comp else "No",
        "Sales (k)": sales, "Range (k)": f"{res['range_k'][0]:,.0f} – {res['range_k'][1]:,.0f}",
        "Profit after ads (k)": sales * M - ad, "Warnings": len(res["warnings"])})
if b2.button("Clear", icon=":material/delete:", width="stretch", disabled=not st.session_state.scenarios):
    st.session_state.scenarios = []
    st.rerun()
if st.session_state.scenarios:
    sc = pd.DataFrame(st.session_state.scenarios)
    sc["vs plan #1 (k)"] = sc["Profit after ads (k)"] - sc["Profit after ads (k)"].iloc[0]
    best_i = sc["Profit after ads (k)"].idxmax()
    st.dataframe(sc.round(1), hide_index=True, width="stretch",
                 column_config={"vs plan #1 (k)": st.column_config.NumberColumn(format="%+.1f")})
    st.caption(f"Highest profit after ads: plan {sc.loc[best_i, 'Plan']}.")
else:
    st.caption("Set a plan above, then add it here to compare several side by side.")

with st.expander("About this model"):
    st.write(
        f"Trained on {meta['training_rows']} clean weeks with a time-based split: the last {meta['test_weeks']} "
        "weeks were held out to measure accuracy, then the winner was refit on everything. Store visitors are "
        "deliberately excluded because they are only known after the week ends. The ad effect is a correlation, "
        "not proven cause; a randomized ad test is the way to confirm it.")
    st.page_link("views/compare.py", label="See how it beat the other models", icon=":material/leaderboard:")
