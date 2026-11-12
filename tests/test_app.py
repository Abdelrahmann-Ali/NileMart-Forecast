"""Smoke tests: every page of the Streamlit app renders without an exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")
PAGES = ["views/overview.py", "views/notebook.py", "views/compare.py", "views/planner.py"]


def _run(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180).run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _run(page)
    assert not at.exception, [e.message for e in at.exception]


def test_every_notebook_step_renders():
    at = _run("views/notebook.py")
    for step in range(2, 14):
        at.session_state["nb_step"] = step
        at.run()
        assert not at.exception, (step, [e.message for e in at.exception])


def test_compare_with_leaky_model_and_other_metric():
    at = _run("views/compare.py")
    at.multiselect[0].set_value(["Model B", "Model C", "Random forest"]).run()
    at.selectbox[0].set_value("R²").run()
    assert not at.exception, [e.message for e in at.exception]
