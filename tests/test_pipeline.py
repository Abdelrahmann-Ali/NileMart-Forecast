from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nilemart.data import clean, load_raw, time_split
from nilemart.predict import Plan, forecast
from nilemart.train import run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/nilemart_weekly_sales.csv"


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    report = run(DATA, d / "model.json", d / "metrics.json")
    return d / "model.json", report


def test_clean_removes_bad_weeks_and_fills_gaps():
    df = clean(load_raw(DATA))
    assert len(df) == 154
    assert df["ad_spend_k"].isna().sum() == 0
    assert not df["week"].isin([45, 97]).any()


def test_clean_does_not_mutate_input():
    raw = load_raw(DATA)
    clean(raw)
    assert len(raw) == 156


def test_time_split_keeps_order():
    train, test = time_split(clean(load_raw(DATA)))
    assert train["week"].max() < test["week"].min()
    assert len(test) == 30


def test_missing_column_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"week": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_raw(p)


def test_linear_model_selected_and_accurate(trained):
    _, report = trained
    assert report["model"] == "linear_regression"
    assert report["selected_test_mae_k"] < 30


def test_more_ads_more_sales(trained):
    path, _ = trained
    lo = forecast(Plan(60, 10, 0, 1), path)["forecast_sales_k"]
    hi = forecast(Plan(100, 10, 0, 1), path)["forecast_sales_k"]
    assert hi > lo


def test_out_of_range_warns(trained):
    path, _ = trained
    assert forecast(Plan(400, 10, 0, 1), path)["warnings"]
    assert not forecast(Plan(80, 10, 0, 1), path)["warnings"]


def test_api():
    from api.main import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/forecast", json={"ad_spend_k": 80, "discount_pct": 10, "is_holiday_period": 0, "competitor_promo": 1})
    assert r.status_code == 200 and r.json()["forecast_sales_k"] > 0
    bad = c.post("/forecast", json={"ad_spend_k": 80, "discount_pct": 10, "is_holiday_period": 3, "competitor_promo": 1})
    assert bad.status_code == 422
