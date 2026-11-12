"""Load the saved model and turn a weekly plan into a forecast."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

from nilemart import FEATURES

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models/model.json"


@dataclass
class Plan:
    ad_spend_k: float
    discount_pct: float
    is_holiday_period: int
    competitor_promo: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([[getattr(self, f) for f in FEATURES]], columns=FEATURES)


@lru_cache(maxsize=4)
def load_model(path: str | Path = DEFAULT_MODEL):
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    return bundle["model"], bundle["meta"]


def out_of_range(plan: Plan, meta: dict) -> list[str]:
    """Inputs outside what the model saw in training. Forecasts there are guesses."""
    warnings = []
    for f, (lo, hi) in meta["feature_ranges"].items():
        v = getattr(plan, f)
        if v < lo or v > hi:
            warnings.append(f"{f}={v} is outside the training range [{lo:g}, {hi:g}]")
    return warnings


def forecast(plan: Plan, model_path: str | Path = DEFAULT_MODEL) -> dict:
    model, meta = load_model(str(model_path))
    pred = model["intercept"] + sum(
        model["coefficients"][feature] * getattr(plan, feature) for feature in FEATURES
    )
    mae = meta["selected_test_mae_k"]
    result = {
        "forecast_sales_k": round(pred, 1),
        "typical_error_k": mae,
        "range_k": [round(pred - mae, 1), round(pred + mae, 1)],
        "warnings": out_of_range(plan, meta),
        "model": meta["model"],
        "model_version": meta["version"],
    }
    if "coefficients" in meta:
        slope = meta["coefficients"]["ad_spend_k"]
        result["profit_per_extra_egp_of_ads"] = round(slope * meta["gross_margin"], 3)
    return result
