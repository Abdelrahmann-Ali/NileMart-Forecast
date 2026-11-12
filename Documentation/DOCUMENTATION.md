# NileMart Sales Forecast: Project Documentation

Version 1.0.0 · Python 3.10+ · Last updated November 2026

This document covers the whole project: the business problem, the data, how the model was built and chosen, every module in the code, the API, the Streamlit app, testing, deployment and maintenance. For a short introduction, see the [README](../README.md).

---

## Contents

1. [Overview](#1-overview)
2. [Quick start](#2-quick-start)
3. [Repository layout](#3-repository-layout)
4. [The data](#4-the-data)
5. [Methodology](#5-methodology)
6. [Results and recommendation](#6-results-and-recommendation)
7. [Architecture](#7-architecture)
8. [The `nilemart` package](#8-the-nilemart-package)
9. [Training pipeline](#9-training-pipeline)
10. [REST API](#10-rest-api)
11. [Streamlit app](#11-streamlit-app)
12. [The launcher (`start.py`)](#12-the-launcher-startpy)
13. [The notebook](#13-the-notebook)
14. [Testing](#14-testing)
15. [CI, Docker and deployment](#15-ci-docker-and-deployment)
16. [Maintenance and extension](#16-maintenance-and-extension)
17. [Troubleshooting](#17-troubleshooting)
18. [Glossary](#18-glossary)

---

## 1. Overview

### The business problem

NileMart is an Egyptian grocery retailer. Before each week, the marketing manager sets the ad budget and the average discount, and asks:

> *"If I spend more on ads next week, how much will sales grow, and is it worth the money?"*

The project answers this with a regression model trained on three years (156 weeks) of weekly sales data. The model is delivered three ways:

| Deliverable | For whom | Where |
|---|---|---|
| Model development notebook | Analysts, reviewers | `notebooks/01_model_development.ipynb` |
| Streamlit app (overview, notebook walkthrough, model comparison, weekly planner) | Marketing manager, stakeholders | `app/` |
| REST API | Other systems | `api/` |

### The answer, in one paragraph

A linear regression on four inputs the manager knows in advance (ad spend, discount, holiday week, competitor promotion) forecasts weekly sales with a typical error of **EGP 23,300** on weeks it never saw. Each extra EGP 1,000 of ads goes with about **EGP 2,580** more sales. At NileMart's 30% gross margin, that returns **EGP 0.77 of profit per EGP spent**, so **raising the ad budget does not pay for itself**. This is a correlation, not a proven cause; a randomized ad test is the recommended next step.

---

## 2. Quick start

### Requirements

- Python 3.10 or newer (tested on 3.12 and 3.14)
- The packages in `requirements.txt`: pandas, numpy, scikit-learn, streamlit, fastapi, uvicorn, pytest, httpx

### One command

| Platform | How |
|---|---|
| Windows | `python start.py` |
| macOS / Linux | `python start.py` |
| Any terminal | `python start.py` |

The launcher installs the requirements and trains the model if needed, then shows a menu. Choose **1** to open the app at http://localhost:8501. Press **Ctrl+C** to stop. See [section 12](#12-the-launcher-startpy) for every option.

### Manual setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export PYTHONPATH=src              # Windows PowerShell: $env:PYTHONPATH = "src"
python -m nilemart.train           # train and save models/model.json
pytest -q                          # run the tests

streamlit run app/streamlit_app.py # the app        -> http://localhost:8501
uvicorn api.main:app --reload      # the API        -> http://localhost:8000/docs
```

---

## 3. Repository layout

```
nilemart-forecast/
├── data/
│   └── nilemart_weekly_sales.csv      Raw weekly data, 156 rows × 7 columns
├── src/nilemart/                      The core Python package
│   ├── __init__.py                    Feature list, target, gross margin, version
│   ├── data.py                        Load, validate, clean, time split
│   ├── train.py                       Fit candidates, pick the best, save it
│   └── predict.py                     Load the saved model, forecast a plan
├── models/
│   └── model.json                    Trained coefficients + metadata (created by train.py)
├── reports/
│   └── metrics.json                   Scores and metadata from the last training run
├── api/
│   └── main.py                        FastAPI service
├── app/                               Streamlit app
│   ├── streamlit_app.py               Entry point: page config, theme, navigation, sidebar
│   ├── nilemart_ui.py                 Shared theme, model registry, cached data/models, chart helpers
│   └── views/
│       ├── overview.py                Landing page
│       ├── notebook.py                Notebook walkthrough (13 steps)
│       ├── compare.py                 Model comparison
│       └── planner.py                 Weekly planner
├── notebooks/
│   └── 01_model_development.ipynb     Model development notebook (solved)
├── tests/
│   ├── test_pipeline.py               Data, training, prediction and API tests
│   └── test_app.py                    Streamlit page smoke tests
├── .streamlit/config.toml             App theme (light, NileMart green)
├── .github/workflows/ci.yml           GitHub Actions: retrain + test on every push
├── Dockerfile                         Container image (API by default)
├── start.py                            One-command launcher
├── pyproject.toml                     Package metadata and pytest settings
├── requirements.txt                   Dependencies
└── README.md                          Short introduction
```

`Project_1_NileMart_Sales_Forecast_solved.ipynb` in the root is a copy of the solved notebook (the same file as `notebooks/01_model_development.ipynb`).

---

## 4. The data

### Data dictionary

File: `data/nilemart_weekly_sales.csv`, one row per week, weeks 1 to 156.

| Column | Type | Unit | Meaning | Used by the model? |
|---|---|---|---|---|
| `week` | int | — | Week number, 1 to 156 | Ordering only |
| `ad_spend_k` | float | EGP thousand | Ad spend that week | Yes |
| `discount_pct` | float | % | Average discount that week (0, 5, 10, 15 or 20) | Yes |
| `is_holiday_period` | 0/1 | — | 1 for Ramadan, Eid and year-end weeks | Yes |
| `competitor_promo` | 0/1 | — | 1 if the main competitor ran a promotion | Yes |
| `store_visitors_k` | float | thousands | Store visitors that week | **No: leaks the answer** (see 5.4) |
| `sales_k` | float | EGP thousand | **Target:** sales that week | Target |

### Data quality issues found

| Issue | Rows | Evidence | Action |
|---|---|---|---|
| Missing ad spend | Weeks 18, 59, 91, 128, 142 | `isna()` | Filled with the median (78.0k) |
| Impossible low sales | Week 45: 31.5k | 49.6k visitors but under 1 EGP per visitor; every other week sells 7.9–9.2 EGP per visitor | Removed. Operations confirmed a system outage |
| Impossible high sales | Week 97: 4,215k | About 10× a normal week from 41k visitors | Removed. Operations confirmed a typing error |

The IQR outlier rule flags exactly weeks 45 and 97 on `sales_k`. The two errors also pull the raw mean (452.5k) well above the median (411.3k).

### Cleaning rules (`src/nilemart/data.py`)

1. Check that every required column exists, else raise `ValueError`.
2. Sort by `week`.
3. Fill missing `ad_spend_k` with the column median.
4. Keep only weeks with `100 ≤ sales_k ≤ 1500` (constants `SALES_MIN_K`, `SALES_MAX_K`).
5. Check that `is_holiday_period` and `competitor_promo` contain only 0 and 1, else raise `ValueError`.

Result: **154 clean weeks**. The input DataFrame is never modified.

**Why investigate instead of deleting all outliers:** an unusual week is not always a wrong week. An Eid week can look extreme but is real demand the model must learn. Only rows confirmed as recording errors are removed.

---

## 5. Methodology

### 5.1 Time-based split

The data is ordered in time and the model forecasts the future, so the test set must be the future:

- **Train:** all clean weeks except the last 30 (124 weeks, weeks 1–126)
- **Test:** the last 30 clean weeks (weeks 127–156)

The data is never shuffled. A random split would let the model learn from weeks after the ones it is tested on, and would place almost every holiday season in training, so scores would look better than real use.

### 5.2 Models compared

All models are fitted on the training weeks and scored on the same 30 test weeks.

| Model | Algorithm | Inputs | Role |
|---|---|---|---|
| **Model A** | Linear regression | `ad_spend_k` only | Baseline |
| **Model B** | Linear regression | The 4 plannable inputs | **Selected** |
| **Model C** | Linear regression | 4 inputs + `store_visitors_k` | Leakage demonstration; not usable |
| **Decision tree** | `DecisionTreeRegressor(max_depth=6, min_samples_leaf=3)` | The 4 plannable inputs | Candidate |
| **Random forest** | `RandomForestRegressor(n_estimators=300, min_samples_leaf=2)` | The 4 plannable inputs | Candidate |

The production training script (`train.py`) considers only the three models that use plannable inputs: linear regression (Model B), the decision tree and the random forest. Models A and C exist in the notebook and the app to explain the reasoning.

**Tree tuning.** The tree's settings were chosen with `GridSearchCV` over `max_depth ∈ {2,3,4,5,6}` and `min_samples_leaf ∈ {3,5,10}`, using `TimeSeriesSplit(n_splits=5)` on the training weeks only, so the test weeks stayed untouched. Best: `max_depth=6`, `min_samples_leaf=3` (cross-validated MAE 42.7k). The random forest settings came from a similar search in the notebook.

### 5.3 Metrics

| Metric | Meaning | Better |
|---|---|---|
| **MAE** | Average size of a miss, in EGP thousand. The headline metric: managers plan stock and staff around it | Lower |
| **RMSE** | Like MAE but punishes big misses more | Lower |
| **R²** | Share of week-to-week variation explained | Higher |
| MAPE | Average miss as a % of actual sales (app only) | Lower |
| Bias | Average of actual − forecast; positive means forecasts run low (app only) | Closer to 0 |
| Within ±10% | Share of test weeks forecast within 10% of actual (app only) | Higher |

Model selection uses **test MAE**.

### 5.4 Data leakage: why store visitors are excluded

Adding `store_visitors_k` (Model C) raises R² from 0.89 to 0.99 and cuts MAE from 23.3k to 9.7k. It is still rejected because **visitors are only known after the week ends**. The manager cannot enter them when setting next week's budget. Visitors are almost the answer itself: sales run at a steady 8.5 EGP per visitor (standard deviation 0.24). Model C would also mislead: its ad coefficient falls to 0.28, because ads work mainly *by* bringing visitors in.

### 5.5 Confounding: why the holiday control matters

Holiday weeks have both the biggest ad budgets (about 114k vs 73k) and the biggest sales (about 599k vs 393k). Model A, with ad spend alone, credits ads with part of the holiday boost: its slope is **3.49**. Model B gives holidays their own coefficient, and the ad slope falls to **2.59**. Model A would wrongly suggest that ads pay for themselves (return 1.05).

### 5.6 From slope to money

```
return per EGP of ads = ad-spend coefficient × gross margin
                      = 2.58 × 0.30 ≈ 0.77   (production model)
```

Above 1 means one extra EGP of ads pays for itself in gross profit. Below 1, it does not.

### 5.7 Range guard

The model has only seen ad spend from 40.2k to 147.1k EGP. A linear model extends a straight line beyond that (400k of ads would forecast about 1,228k of sales), while real ads have diminishing returns. Any plan outside a feature's training range gets a warning in the API response and in the planner.

---

## 6. Results and recommendation

### Held-out test (30 most recent weeks)

| Model | MAE (k) | RMSE (k) | R² | MAPE | Train MAE (k) | Usable? |
|---|---|---|---|---|---|---|
| Model C (+ visitors) | 9.7 | 11.6 | 0.987 | 2.2% | 8.3 | No, leaks |
| **Model B (linear, 4 inputs)** | **23.3** | **33.6** | **0.891** | **5.4%** | 19.6 | **Yes, selected** |
| Random forest | 34.9 | 44.7 | 0.806 | 7.8% | 15.2 | Yes |
| Decision tree | 43.0 | 56.1 | 0.694 | 9.9% | 18.3 | Yes |
| Model A (ad spend only) | 48.7 | 61.9 | 0.628 | 10.3% | 41.2 | Yes, baseline |

The trees fit the training weeks far better than they forecast new ones (train MAE 15–18k vs test 35–43k): they memorise past weeks. Linear regression scores about the same on both, which means it learned the pattern.

### Rolling backtest (5 expanding-window folds of about 20 weeks)

| Model | Mean MAE (k) | ± std |
|---|---|---|
| Model C (leaks) | 9.1 | 1.3 |
| **Model B** | **20.9** | **2.4** |
| Random forest | 30.4 | 6.6 |
| Decision tree | 35.2 | 6.6 |
| Model A | 43.6 | 9.3 |

Model B wins every fold among usable models and is also the most stable.

### Production coefficients (Model B refit on all 154 weeks)

`sales_k = 183.6 + 2.580 × ad_spend_k + 4.867 × discount_pct + 99.97 × is_holiday_period − 38.79 × competitor_promo`

| Input | Coefficient | In plain words |
|---|---|---|
| Ad spend | +2.58 | +EGP 2,580 of sales per extra EGP 1,000 of ads |
| Discount | +4.87 | +EGP 4,870 per extra 1% of discount |
| Holiday week | +99.97 | About EGP 100,000 more in a Ramadan, Eid or year-end week |
| Competitor promo | −38.79 | About EGP 38,800 less when the competitor runs a promotion |

The notebook's numbers (slope 2.59, holiday +99, competitor −42) come from the model fitted on the 124 training weeks only. Production refits on all 154 weeks, so its coefficients differ slightly.

### Recommendation memo

> I recommend Model B, which forecasts weekly sales from ad spend, discount, holidays and competitor promotions; it is typically off by about EGP 23,000 a week. Each extra EGP 1,000 of ads goes with about EGP 2,580 more sales, which at our 30% margin returns only about 0.77 EGP in profit per EGP spent, so raising the ad budget does not pay for itself. The main limitation is that this is a correlation, and the model has only seen budgets up to about EGP 147,000. The next step is a controlled test that randomly varies ad spend across weeks or regions. Until then, keep the ad budget at its current level.

### Limitations

- **Correlation, not cause.** Unrecorded factors may move with ad spend. Only a randomized experiment measures the true ad effect.
- **Linear and additive.** Real ad returns flatten at high budgets; the model cannot show that inside or outside the training range.
- **Filled values.** The two biggest Model B misses (weeks 128 and 142) are weeks whose missing ad spend was filled with the median, so the true spend was probably lower.
- **Small data.** 154 weeks; some errors come in short runs, which hints at local events or seasonality the model does not capture.
- **The "likely range" is a simple band** of forecast ± test MAE, not a statistical prediction interval.

---

## 7. Architecture

```
                  data/nilemart_weekly_sales.csv
                               │
                    src/nilemart/data.py      load_raw → clean → time_split
                               │
                    src/nilemart/train.py     score candidates on last 30 weeks,
                               │              pick lowest MAE, refit on all weeks
                 ┌─────────────┴─────────────┐
      models/model.json              reports/metrics.json
                 │
       src/nilemart/predict.py        Plan → forecast() → dict
          ┌──────┴───────┐
     api/main.py     app/views/planner.py
     (FastAPI)       (Streamlit)

  app/nilemart_ui.py also imports data.py and train.py directly, so the
  analysis pages (overview, notebook, compare) refit the models live with
  exactly the same cleaning, split and model settings as production.
```

Design principles:

- **One source of truth.** The feature list, cleaning rules and candidate models are defined once in `src/nilemart` and reused by training, the API, the app and the tests.
- **Production uses the saved artifact.** The API and planner never retrain; they load the JSON model at `models/model.json`.
- **Analysis is live.** The analysis pages refit on the current data, so changing the test window or uploading a new CSV updates every number.

---

## 8. The `nilemart` package

Located in `src/nilemart/`. Add `src` to `PYTHONPATH` (or `pip install -e .`) to import it.

### `nilemart/__init__.py`

| Name | Value | Meaning |
|---|---|---|
| `FEATURES` | `["ad_spend_k", "discount_pct", "is_holiday_period", "competitor_promo"]` | Model inputs, in order |
| `TARGET` | `"sales_k"` | Target column |
| `GROSS_MARGIN` | `0.30` | Gross margin used for profit maths |
| `__version__` | `"1.0.0"` | Saved into the model metadata |

### `nilemart/data.py`

| Function | Signature | Description |
|---|---|---|
| `load_raw` | `load_raw(path) -> DataFrame` | Reads a CSV (path or file-like), checks required columns (`ValueError` if missing), sorts by week |
| `clean` | `clean(df) -> DataFrame` | Fills missing ad spend with the median, drops weeks outside `[SALES_MIN_K, SALES_MAX_K]`, validates 0/1 flags. Returns a copy |
| `time_split` | `time_split(df, test_weeks=30) -> (train, test)` | Last `test_weeks` rows are the test set. Never shuffles |

Constants: `REQUIRED_COLUMNS`, `SALES_MIN_K = 100`, `SALES_MAX_K = 1500`.

### `nilemart/train.py`

| Name | Description |
|---|---|
| `CANDIDATES` | Dict of unfitted estimators: `linear_regression`, `decision_tree`, `random_forest` |
| `evaluate(y_true, y_pred) -> dict` | Returns `{"mae_k", "rmse_k", "r2"}`, rounded |
| `run(data_path, model_path, report_path, test_weeks=30) -> dict` | The full pipeline (see [section 9](#9-training-pipeline)). Returns the report dict |
| `main()` | Command-line entry point |

### `nilemart/predict.py`

| Name | Description |
|---|---|
| `DEFAULT_MODEL` | `models/model.json` at the repository root |
| `Plan` | Dataclass with `ad_spend_k`, `discount_pct`, `is_holiday_period`, `competitor_promo`. `as_frame()` returns a one-row DataFrame with columns in `FEATURES` order |
| `load_model(path=DEFAULT_MODEL) -> (model, meta)` | Reads the JSON bundle. **Cached** with `lru_cache`, so a running process keeps the model it first loaded |
| `out_of_range(plan, meta) -> list[str]` | One warning per input outside its training range |
| `forecast(plan, model_path=DEFAULT_MODEL) -> dict` | Forecast for one plan (format below) |

`forecast()` returns:

| Key | Type | Meaning |
|---|---|---|
| `forecast_sales_k` | float | Forecast sales, EGP thousand |
| `typical_error_k` | float | Test MAE of the selected model |
| `range_k` | [float, float] | Forecast ± typical error |
| `warnings` | list[str] | Out-of-range inputs (empty if none) |
| `model` | str | Selected model name, e.g. `linear_regression` |
| `model_version` | str | Package version at training time |
| `profit_per_extra_egp_of_ads` | float | Ad coefficient × margin. Only present for linear models |

Example:

```python
from nilemart.predict import Plan, forecast
forecast(Plan(ad_spend_k=80, discount_pct=10, is_holiday_period=0, competitor_promo=1))
# {'forecast_sales_k': 399.9, 'typical_error_k': 23.31, 'range_k': [376.6, 423.2],
#  'warnings': [], 'model': 'linear_regression', 'model_version': '1.0.0',
#  'profit_per_extra_egp_of_ads': 0.774}
```

---

## 9. Training pipeline

### Running it

```bash
python -m nilemart.train                    # defaults
python -m nilemart.train --test-weeks 26    # different holdout
python -m nilemart.train --data other.csv --model out/model.json --report out/metrics.json
```

| Option | Default |
|---|---|
| `--data` | `data/nilemart_weekly_sales.csv` |
| `--model` | `models/model.json` |
| `--report` | `reports/metrics.json` |
| `--test-weeks` | `30` |

### What it does

1. Load and clean the data (`load_raw`, `clean`).
2. Split by time (`time_split`).
3. Fit each candidate on the training weeks and score it on the test weeks.
4. Pick the candidate with the lowest test MAE.
5. **Refit a fresh copy of the winner on all clean weeks**, so production uses every week of data.
6. Compute residual spread, training ranges and, for linear models, the intercept and coefficients.
7. Save the model bundle and the report.

Sample output:

```
Selected: linear_regression  (test MAE 23.31k EGP)
  linear_regression  MAE  23.31  RMSE  33.55  R² 0.891
  decision_tree      MAE  43.05  RMSE  56.11  R² 0.694
  random_forest      MAE  34.87  RMSE  44.74  R² 0.805
Saved model to models/model.json
```

### Outputs

**`models/model.json`** is a JSON object with a `model` section (model type, intercept, and coefficients) and a `meta` section containing the same training report written to `reports/metrics.json`. The production predictor evaluates this saved linear model directly; it does not deserialize a Python estimator.

**`reports/metrics.json`** fields:

| Field | Meaning |
|---|---|
| `version`, `trained_at` | Package version and UTC timestamp |
| `model` | Selected candidate name |
| `features` | Input columns, in order |
| `training_rows` | Clean weeks used for the final refit (154) |
| `test_weeks` | Holdout size used for selection |
| `test_scores` | MAE / RMSE / R² for every candidate |
| `selected_test_mae_k` | Test MAE of the winner; used as the "typical error" |
| `residual_std_k` | Residual standard deviation of the final model (stored, not currently used) |
| `gross_margin` | 0.30 |
| `feature_ranges` | `[min, max]` of each input in the training data; drives the range guard |
| `intercept`, `coefficients` | Linear models only |

---

## 10. REST API

`api/main.py`, built with FastAPI. Start it with `uvicorn api.main:app --reload` or `python start.py api`. Interactive docs: http://localhost:8000/docs.

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status", "model", "version", "trained_at"}` |
| POST | `/forecast` | One plan | One forecast (the `forecast()` dict above) |
| POST | `/forecast/batch` | List of plans | List of forecasts, same order |

### Plan schema

| Field | Type | Validation | Description |
|---|---|---|---|
| `ad_spend_k` | float | ≥ 0 | Ad spend, EGP thousand |
| `discount_pct` | float | 0–100 | Average discount, % |
| `is_holiday_period` | int | 0 or 1 | 1 for Ramadan, Eid or year-end |
| `competitor_promo` | int | 0 or 1 | 1 if the competitor runs a promotion |

Values that fail validation return **HTTP 422**. Values that pass validation but sit outside the training range (for example `ad_spend_k = 250` or `discount_pct = 40`) return a normal forecast with a message in `warnings`.

### Examples

```bash
curl -X POST localhost:8000/forecast -H "Content-Type: application/json" \
  -d '{"ad_spend_k": 80, "discount_pct": 10, "is_holiday_period": 0, "competitor_promo": 1}'
```

```json
{
  "forecast_sales_k": 399.9,
  "typical_error_k": 23.31,
  "range_k": [376.6, 423.2],
  "warnings": [],
  "model": "linear_regression",
  "model_version": "1.0.0",
  "profit_per_extra_egp_of_ads": 0.774
}
```

Out of range:

```json
"warnings": ["ad_spend_k=250.0 is outside the training range [40.2, 147.1]"]
```

Batch:

```bash
curl -X POST localhost:8000/forecast/batch -H "Content-Type: application/json" \
  -d '[{"ad_spend_k": 60, "discount_pct": 10, "is_holiday_period": 0, "competitor_promo": 1},
       {"ad_spend_k": 100, "discount_pct": 10, "is_holiday_period": 0, "competitor_promo": 1}]'
```

---

## 11. Streamlit app

Start it with `python start.py` or `streamlit run app/streamlit_app.py` (port 8501).

### Pages

| Page | URL | Data source | Contents |
|---|---|---|---|
| **Overview** | `/` | Live refit | The question and answer, four headline metrics, five-model leaderboard, leakage warning, links to other pages, notebook-to-production flow |
| **Notebook walkthrough** | `/notebook` | Live refit | 13 steps: Tasks 1–11, the memo, and trees + saving the model. Each step shows the notebook's Goal and Why it matters, the charts and tables, and its notebook question with an answer computed from the data. Pills at the top and Previous/Next buttons at the bottom move between steps |
| **Model comparison** | `/compare` | Live refit | See below |
| **Weekly planner** | `/planner` | **Saved production model** | See below |

**Model comparison** contents:

- Model picker (Model C off by default) and a "rank by" selector (MAE, RMSE, R², MAPE)
- Leaderboard: MAE, RMSE, R², MAPE, % of weeks within ±10%, bias, train MAE, test − train gap, status
- Verdict: the best model usable for planning and its margin over the runner-up
- Metric panels: MAE, RMSE and R² side by side, same model order in each
- Tabs:
  - *Forecast vs actual*: lines over the test weeks with a hover crosshair (optionally with the last 26 training weeks)
  - *Predicted vs actual*: one scatter per model against the perfect-forecast line
  - *Errors*: error spread per model, cumulative absolute error, worst weeks
  - *Head-to-head*: pick two models; per-week bars show which was closer
  - *Rolling backtest*: MAE per expanding-window fold, mean ± std, folds won
  - *Overfitting*: train vs test MAE per model
  - *What drives each model*: share of predictions explained by each input, plus linear coefficients
- Tree tuning heatmap (GridSearchCV results)
- CSV downloads of the leaderboard and the test forecasts

**Weekly planner** contents:

- Plan inputs: ad spend (0–200k), discount (0–20%), holiday week, competitor promotion
- Expected sales, likely range, gross profit, profit after ads
- Out-of-range warnings and the budget signal (profit per EGP of ads)
- Scenario analysis: gross profit and profit after ads across budgets, with the training range shaded
- Model cross-check: each plannable model's forecast for the plan (refit on all weeks) ± its typical error, and how each responds to the budget; warns when models disagree by more than the typical error
- Compare plans: add the current plan to a table and compare several side by side (kept for the browser session)

### Sidebar: data and evaluation settings

| Setting | Effect |
|---|---|
| Weekly sales CSV (optional) | Use an uploaded file with the same columns instead of `data/`. If it cannot be read, the app shows an error and falls back to the bundled data |
| Test weeks (10–52, default 30) | Size of the holdout for the analysis pages (capped at a third of the data) |

The planner always uses the saved production model and is not affected by these settings, except for its model cross-check chart.

### Code structure

| File | Role |
|---|---|
| `streamlit_app.py` | Adds `app/` and `src/` to `sys.path`, sets page config, injects CSS, defines navigation (`st.navigation`), renders the sidebar settings, runs the selected page |
| `nilemart_ui.py` | Shared module: palette, model registry (`SPECS`), data loading, cached model fitting, chart helpers, page furniture (hero, sections, callouts), CSS |
| `views/*.py` | One script per page |

Key items in `nilemart_ui.py`:

| Name | Purpose |
|---|---|
| `SPECS` | Registry of the five models: name, algorithm, inputs, features, estimator, colour, role (`baseline` / `candidate` / `leak`). The tree and forest estimators are taken from `nilemart.train.CANDIDATES` |
| `ORDER`, `PLANNABLE` | All model names; models usable for planning (everything except Model C) |
| `get_data()` | Raw and cleaned data (upload or bundled), cached |
| `fit_models(data, n_test)` | Fits all five on the training weeks; returns `(train, test, results)` with predictions and scores. Cached |
| `fit_full(data)` | Plannable models refit on all weeks (planner cross-check). Cached |
| `backtest(data)` | 5-fold expanding-window backtest. Cached |
| `tree_grid(data, n_test)` | The notebook's tree GridSearchCV. Cached |
| `importance(R, data, keys)` | Input shares: tree importances, or \|coefficient × input std\| for linear models |
| `production()` | `(model, meta)` from `models/model.json`, or `(None, None)` if not trained |
| `show()`, `time_lines()`, `metric_bars()`, `band()`, `vline()`, `hline()` | Altair chart helpers with the shared style |

### Visual conventions

- Each model keeps one colour on every chart: Model B blue, Random forest orange, Decision tree aqua, Model A yellow, Model C grey with a **dashed** line (so the leaky model is never identified by colour alone).
- Holiday weeks are amber, normal weeks dark green. Errors use blue for under-forecasts (actual above) and red for over-forecasts.
- Every chart has a tooltip; the numbers behind charts are also shown in tables.
- Fonts: DM Serif Display for page titles, DM Sans elsewhere (loaded from Google Fonts). Theme colours are in `.streamlit/config.toml` and the CSS in `nilemart_ui.py`.

---

## 12. The launcher (`start.py`)

```bash
python start.py [option]
```

| Option | Starts | Opens |
|---|---|---|
| *(none)* | Menu | — |
| `app` | Streamlit app on :8501 | Overview |
| `walkthrough` | Streamlit app | `/notebook` |
| `compare` | Streamlit app | `/compare` |
| `planner` | Streamlit app | `/planner` |
| `api` | FastAPI on :8000 | `/docs` |
| `all` | App + API | Both |
| `train` | Retrains the model | — |
| `test` | Runs pytest | — |

Before starting anything, the launcher installs `requirements.txt` if a key package is missing and trains the model if `models/model.json` does not exist. It sets `PYTHONPATH` to `src` and the project root for the processes it starts. Run `python start.py` on any platform.

---

## 13. The notebook

`notebooks/01_model_development.ipynb` is the solved model development notebook. It was originally written as a guided exercise (Goal, Why it matters, Steps, Hint, self-check cell and a written question per task). It expects `nilemart_weekly_sales.csv` in the working directory (in Colab, upload it first).

| Notebook section | Content | App step |
|---|---|---|
| Task 1 · Load and explore | Shape, summary, missing values, IQR outliers, box plots · Question 1 | Walkthrough 1 |
| Task 2 · Clean | Median fill, remove weeks 45 and 97 · Question 2 | Walkthrough 2 |
| Task 3 · Relationship | Ad spend vs sales scatter by holiday · Question 3 | Walkthrough 3 |
| Task 4 · Split by time | Last 30 weeks as test · Question 4 | Walkthrough 4 |
| Task 5 · Model A | Ads-only regression, slope 3.49 · Question 5 | Walkthrough 5 |
| Task 6 · Evaluate Model A | MAE 48.7k, RMSE 61.9k, R² 0.63 · Question 6 | Walkthrough 6 |
| Task 7 · Model B | Four plannable inputs, coefficients · Questions 7–8 | Walkthrough 7 |
| Task 8 · Compare and inspect errors | Results table, error bars · Question 9 | Walkthrough 8 |
| Task 9 · Spot the leak | Model C with visitors · Question 10 | Walkthrough 9 |
| Task 10 · Ad return | Slope × margin · Question 11 | Walkthrough 10 |
| Task 11 · Plan next week | 60/80/100k plans, 400k extreme · Question 12 | Walkthrough 11 |
| Final deliverable | Five-sentence memo | Walkthrough 12 |
| Decision tree, random forest, comparison charts | GridSearchCV + TimeSeriesSplit; five-model comparison | Walkthrough 13, Model comparison page |
| Saving the best model | Pick lowest MAE, refit on all weeks, save a notebook artifact with `joblib` | Walkthrough 13; the production trainer writes `models/model.json` |

---

## 14. Testing

Run with `pytest -q` (or `python start.py test`). `pyproject.toml` puts `src` and the project root on the path.

### `tests/test_pipeline.py`

| Test | Checks |
|---|---|
| `test_clean_removes_bad_weeks_and_fills_gaps` | 154 rows, no missing ad spend, weeks 45 and 97 gone |
| `test_clean_does_not_mutate_input` | Raw DataFrame still has 156 rows |
| `test_time_split_keeps_order` | All train weeks precede all test weeks; 30 test rows |
| `test_missing_column_rejected` | `load_raw` raises `ValueError` |
| `test_linear_model_selected_and_accurate` | Training picks linear regression with test MAE < 30k |
| `test_more_ads_more_sales` | Forecast rises with ad spend |
| `test_out_of_range_warns` | 400k ad spend warns; 80k does not |
| `test_api` | `/health` ok, `/forecast` returns a positive forecast, an invalid flag returns 422 |

Training tests use a temporary directory, so they never overwrite `models/`.

### `tests/test_app.py`

Uses Streamlit's `AppTest` to run the app headlessly:

| Test | Checks |
|---|---|
| `test_page_renders` (×4) | Overview, notebook, compare and planner render with no exception |
| `test_every_notebook_step_renders` | All 13 walkthrough steps render |
| `test_compare_with_leaky_model_and_other_metric` | Comparison page works with Model C selected and ranking by R² |

The planner and API use `models/model.json`. It is included in the repository; CI retrains it before running the tests.

---

## 15. CI, Docker and deployment

### GitHub Actions (`.github/workflows/ci.yml`)

On every push and pull request: set up Python 3.12, install requirements, retrain the model (`PYTHONPATH=src python -m nilemart.train`), run `pytest -q`.

### Docker

The image installs the requirements, copies the project and **trains the model at build time**.

```bash
docker build -t nilemart .
docker run -p 8000:8000 nilemart                                                              # API (default)
docker run -p 8501:8501 nilemart streamlit run app/streamlit_app.py --server.address=0.0.0.0  # app
```

### Hosting options

| Option | How |
|---|---|
| Streamlit Community Cloud (free) | Push to GitHub, go to share.streamlit.io, choose the repo, set the main file to `app/streamlit_app.py`. The production model, `models/model.json`, must be included in the repository. |
| Render / Railway / Fly.io | Deploy the Dockerfile. The default command serves the API; override it to serve the app |

---

## 16. Maintenance and extension

### Retraining with new weeks

1. Append rows to `data/nilemart_weekly_sales.csv` with the same columns.
2. Run `python -m nilemart.train` (or `python start.py train`).
3. Check the printed scores and `reports/metrics.json`.
4. **Restart the API and the app.** Both cache the loaded model (see [Troubleshooting](#17-troubleshooting)).

Before retraining, check new rows for recording errors. The sales band of 100–1500k catches the known kind of error, but a new kind of problem needs a new rule in `clean()`.

### Adding a candidate model

1. Add the estimator to `CANDIDATES` in `src/nilemart/train.py`. Training will score it and select it if it wins.
2. To show it in the app, add a `ModelSpec` to `SPECS` in `app/nilemart_ui.py` with an unused colour from the palette, and an algorithm label.
3. Non-linear winners do not have `coefficients`, so the API omits `profit_per_extra_egp_of_ads` and the planner hides the budget signal.

### Adding an input feature

1. Add the column to the CSV and to `FEATURES` in `src/nilemart/__init__.py`.
2. Add the field to `Plan` (`predict.py`) and `PlanIn` (`api/main.py`).
3. Add its label to `FEATURE_LABELS` and an input widget in `app/views/planner.py`.
4. Make sure it is known **before** the week starts, or it leaks like store visitors.
5. Update the tests.

### Adding an app page

1. Create `app/views/<name>.py`. Start with `import nilemart_ui as ui` and `raw, data = ui.get_data()`.
2. Register it in the `st.navigation` dict in `app/streamlit_app.py` with a title, icon and `url_path`.
3. Add it to `PAGES` in `tests/test_app.py`.

---

## 17. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python` is not recognised | Python not installed or not on PATH | Install Python 3.10+ from python.org and tick "Add Python to PATH" |
| `ModuleNotFoundError: nilemart` | `src` not on the path | Use `python start.py`, or set `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = "src"`) |
| Planner says "No saved model found" | `models/model.json` missing | `python start.py train` |
| API or planner still shows old numbers after retraining | `load_model` and Streamlit caches keep the first model loaded | Restart the API / app process |
| Port 8501 or 8000 already in use | Another instance is running | Stop it, or run Streamlit/Uvicorn directly with `--server.port 8502` / `--port 8001`; `start.py` uses the default ports |
| Uploaded CSV rejected | Missing columns, or a flag column has values other than 0/1 | Match the columns in [section 4](#4-the-data) |
| First page load is slow | Models, backtest and tree grid are being fitted | One-off per data/settings combination; later loads use the cache |

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **EGP k** | Thousands of Egyptian pounds. All money columns are in EGP thousand |
| **Plannable input** | Something the manager knows before the week starts: ad spend, discount, holiday calendar, competitor promotion |
| **Data leakage** | Using information at training time that would not be available when making a real forecast |
| **Confounding** | A third factor (holidays) that moves both the input (ads) and the outcome (sales), making the input look more powerful than it is |
| **Time-based split** | Train on earlier weeks, test on later weeks, never shuffle |
| **Rolling (expanding-window) backtest** | Repeat the time split several times, each time training on everything before a block of weeks and testing on that block |
| **MAE / RMSE / R²** | Error metrics; see [5.3](#53-metrics) |
| **Overfitting** | A model fits its training data much better than new data, because it memorised rather than learned |
| **Gross margin** | Share of sales kept as gross profit (30%) |
| **Return per EGP of ads** | Ad coefficient × gross margin; above 1 means extra ad spend pays for itself |
| **Range guard** | Warning when a plan's inputs fall outside the values seen in training |
