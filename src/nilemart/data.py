"""Loading, validating and cleaning the weekly sales data."""

from pathlib import Path

import pandas as pd

from nilemart import FEATURES, TARGET

REQUIRED_COLUMNS = ["week", *FEATURES, "store_visitors_k", TARGET]

# Sales outside this band are recording errors, not real weeks.
# Operations confirmed week 45 was a system outage and week 97 a typing error.
SALES_MIN_K, SALES_MAX_K = 100, 1500


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.sort_values("week").reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ad spend with the median and drop impossible sales weeks."""
    out = df.copy()
    out["ad_spend_k"] = out["ad_spend_k"].fillna(out["ad_spend_k"].median())
    out = out[out[TARGET].between(SALES_MIN_K, SALES_MAX_K)]
    for flag in ("is_holiday_period", "competitor_promo"):
        if not out[flag].isin([0, 1]).all():
            raise ValueError(f"{flag} must be 0 or 1")
    return out.reset_index(drop=True)


def time_split(df: pd.DataFrame, test_weeks: int = 30):
    """Train on the past, test on the most recent weeks. Never shuffle."""
    return df.iloc[:-test_weeks], df.iloc[-test_weeks:]
