"""Landing page: the question, the answer, and where to go next."""

import pandas as pd
import streamlit as st

import nilemart_ui as ui
from nilemart import FEATURES

raw, data = ui.get_data()
n_test = ui.test_weeks(data)
train, test, R = ui.fit_models(data, n_test)
best = ui.best_plannable(R)
coef_b = pd.Series(R["Model B"]["model"].coef_, index=FEATURES)
slope_a, slope_b = R["Model A"]["model"].coef_[0], coef_b["ad_spend_k"]
ret_b = slope_b * ui.MARGIN

ui.hero(
    "NileMart &nbsp;/&nbsp; Weekly sales forecast",
    "If we spend more on ads next week, is it worth the money?",
    "Before each week the marketing manager sets the ad budget and the discount. This app retraces the "
    "notebook that answers her question with three years of weekly data, compares every model that was tried, "
    "and turns the winner into a planning tool.",
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Weeks of data", f"{len(raw)}", delta=f"{len(data) - len(raw)} after cleaning", delta_color="off")
k2.metric("Typical forecast error", ui.fmt_k(R[best]["test"]["MAE"]),
          help=f"{best}: mean absolute error on the latest {n_test} weeks, never seen in training.")
k3.metric("Sales per extra EGP 1k of ads", f"EGP {slope_b * 1000:,.0f}",
          delta=f"{(slope_b - slope_a) * 1000:,.0f} vs ads-only model", delta_color="off")
k4.metric("Profit per EGP of ads", f"{ret_b:.2f}", delta=f"{ret_b - 1:+.2f} vs break-even",
          delta_color="normal")

verdict = "pays for itself" if ret_b >= 1 else "does not pay for itself"
ui.callout(
    f'<span class="label">The answer</span>Each extra EGP 1,000 of ads goes with about '
    f'<b>EGP {slope_b * 1000:,.0f}</b> more sales. At a {ui.MARGIN:.0%} gross margin that returns '
    f'<b>EGP {ret_b:.2f} of profit per EGP spent</b>, so raising the ad budget <b>{verdict}</b>. '
    f'It is a correlation, not proven cause: a randomized ad test is the way to confirm it.'
)

# ----------------------------------------------------------------- leaderboard
ui.section("Model comparison", "Five models, one honest test",
           f"Every model trained on weeks {int(train.week.min())}–{int(train.week.max())} and scored on weeks "
           f"{int(test.week.min())}–{int(test.week.max())}. Lower MAE is better.")
rows = pd.DataFrame([{"model": k, "MAE": R[k]["test"]["MAE"], "R²": R[k]["test"]["R²"],
                      "Status": ui.badge(k, best)} for k in ui.ORDER])
rows = rows.sort_values("MAE")
c1, c2 = st.columns([1.1, 1])
with c1:
    ui.metric_bars(rows, "MAE", list(rows.model), ",.1f", "Test MAE, EGP thousand", height=230)
with c2:
    st.dataframe(rows.rename(columns={"model": "Model", "MAE": "Test MAE (EGP k)"}), hide_index=True,
                 width="stretch", height=230,
                 column_config={"Test MAE (EGP k)": st.column_config.NumberColumn(format="%.1f"),
                                "R²": st.column_config.NumberColumn(format="%.3f")})
leak = R["Model C"]["test"]
ui.callout(f'Model C scores best on paper (MAE {leak["MAE"]:.1f}k, R² {leak["R²"]:.2f}) but it uses store '
           'visitors, which are only known after the week ends. That is data leakage, so it is excluded.', "leak")

# ----------------------------------------------------------------- where to go
ui.section("Explore", "Pick up where you need to")
tiles = [
    ("Notebook walkthrough", "Follow the notebook task by task: explore, clean, split by time, Models A–C, "
     "the ad-return decision and the memo. Every question answered from the live data.",
     "views/notebook.py", ":material/menu_book:"),
    ("Model comparison", "Leaderboard, forecasts against actual sales, error patterns, head-to-head, a rolling "
     "backtest, overfitting and what drives each model.", "views/compare.py", ":material/leaderboard:"),
    ("Weekly planner", "Set next week's budget, discount and conditions. See the forecast, the likely range, "
     "the profit after ads, and what every model would say.", "views/planner.py", ":material/event_note:"),
]
for col, (title, text, page, icon) in zip(st.columns(3), tiles):
    with col:
        st.markdown(f'<div class="tile"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
        st.page_link(page, label=f"Open {title.lower()}", icon=icon)

# ----------------------------------------------------------------- pipeline
ui.section("How it fits together", "From notebook to production")
st.markdown(
    '<div class="flow"><span>notebooks/01_model_development.ipynb</span><em>→</em>'
    '<span>src/nilemart/train.py</span><em>→</em><span>models/model.json</span><em>→</em>'
    '<span>Weekly planner</span><em>+</em><span>REST API /forecast</span></div>',
    unsafe_allow_html=True)
_, meta = ui.production()
if meta:
    st.caption(f"Production model: {meta['model'].replace('_', ' ')}, trained {meta['trained_at'][:10]} on "
               f"{meta['training_rows']} weeks, test MAE {meta['selected_test_mae_k']:.1f}k. "
               "Retrain with `python start.py train`.")
else:
    st.warning("No saved model yet. Run `python start.py train` to create models/model.json.")
