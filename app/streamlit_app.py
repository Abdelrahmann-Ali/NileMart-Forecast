"""NileMart Sales Forecast: one app for the notebook story, the model comparison and weekly planning.

Run:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="NileMart Sales Forecast", page_icon="🛒", layout="wide")

import nilemart_ui as ui  # noqa: E402

ui.inject_css()

nav = st.navigation({
    "NileMart": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "The analysis": [
        st.Page("views/notebook.py", title="Notebook walkthrough", icon=":material/menu_book:", url_path="notebook"),
        st.Page("views/compare.py", title="Model comparison", icon=":material/leaderboard:", url_path="compare"),
    ],
    "Planning": [
        st.Page("views/planner.py", title="Weekly planner", icon=":material/event_note:", url_path="planner"),
    ],
})

with st.sidebar:
    st.markdown('<div class="side-brand"><b>NileMart</b><span>Weekly sales forecast · v1.0</span></div>',
                unsafe_allow_html=True)
    with st.expander("Data & evaluation settings"):
        st.file_uploader("Weekly sales CSV (optional)", type="csv", key="upload",
                         help="Same columns as data/nilemart_weekly_sales.csv. Leave empty to use the bundled data.")
        st.slider("Test weeks (most recent)", 10, 52, 30, key="test_weeks",
                  help="How many of the latest weeks are held out to score the models. The notebook uses 30.")
    st.caption("Analysis pages refit the models live. The planner uses the saved production model.")

nav.run()
