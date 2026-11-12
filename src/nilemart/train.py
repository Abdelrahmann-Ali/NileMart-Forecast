"""Train candidate models, pick the best on held-out weeks, save it.

Run:  python -m nilemart.train
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn import metrics as M
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from nilemart import FEATURES, GROSS_MARGIN, TARGET, __version__
from nilemart.data import clean, load_raw, time_split

ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42),
}


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae_k": round(float(M.mean_absolute_error(y_true, y_pred)), 2),
        "rmse_k": round(float(M.root_mean_squared_error(y_true, y_pred)), 2),
        "r2": round(float(M.r2_score(y_true, y_pred)), 3),
    }


def run(data_path: Path, model_path: Path, report_path: Path, test_weeks: int = 30) -> dict:
    data = clean(load_raw(data_path))
    train, test = time_split(data, test_weeks)

    scores = {}
    for name, model in CANDIDATES.items():
        fitted = clone(model).fit(train[FEATURES], train[TARGET])
        scores[name] = evaluate(test[TARGET], fitted.predict(test[FEATURES]))

    best_name = min(scores, key=lambda n: scores[n]["mae_k"])
    final = clone(CANDIDATES[best_name]).fit(data[FEATURES], data[TARGET])

    residuals = data[TARGET] - final.predict(data[FEATURES])
    report = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": best_name,
        "features": FEATURES,
        "training_rows": len(data),
        "test_weeks": test_weeks,
        "test_scores": scores,
        "selected_test_mae_k": scores[best_name]["mae_k"],
        "residual_std_k": round(float(np.std(residuals, ddof=len(FEATURES) + 1)), 2),
        "gross_margin": GROSS_MARGIN,
        "feature_ranges": {
            f: [float(data[f].min()), float(data[f].max())] for f in FEATURES
        },
    }
    if hasattr(final, "coef_"):
        report["intercept"] = round(float(final.intercept_), 4)
        report["coefficients"] = {f: round(float(c), 4) for f, c in zip(FEATURES, final.coef_)}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": {
            "type": "linear_regression",
            "intercept": float(final.intercept_),
            "coefficients": {f: float(c) for f, c in zip(FEATURES, final.coef_)},
        },
        "meta": report,
    }
    model_path.write_text(json.dumps(model_bundle, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/nilemart_weekly_sales.csv")
    p.add_argument("--model", type=Path, default=ROOT / "models/model.json")
    p.add_argument("--report", type=Path, default=ROOT / "reports/metrics.json")
    p.add_argument("--test-weeks", type=int, default=30)
    a = p.parse_args()
    report = run(a.data, a.model, a.report, a.test_weeks)
    print(f"Selected: {report['model']}  (test MAE {report['selected_test_mae_k']}k EGP)")
    for name, s in report["test_scores"].items():
        print(f"  {name:18s} MAE {s['mae_k']:6.2f}  RMSE {s['rmse_k']:6.2f}  R² {s['r2']:.3f}")
    print(f"Saved model to {a.model}")


if __name__ == "__main__":
    main()
