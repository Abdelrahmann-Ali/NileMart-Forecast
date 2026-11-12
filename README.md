# NileMart Sales Forecast

NileMart is an Egyptian grocery retailer. This project helps its marketing team plan weekly advertising by forecasting sales from information available before the week begins: ad spend, average discount, holiday status, and competitor promotions. You can also upload your own weekly-sales CSV in the app and rerun the analysis on that data.

It includes a model-development notebook, a Streamlit app for exploring results and planning budgets, and a FastAPI service for forecasts. The analysis is designed to answer a practical question: **does increasing the ad budget improve sales enough to cover its cost?**

**Full documentation:** [Documentation/DOCUMENTATION.md](Documentation/DOCUMENTATION.md) covers the dataset, methodology, results, code, API, app, testing, deployment, and maintenance.

## Screenshots

<!-- Add screenshots here. -->

## What the Analysis Does

1. **Cleans the weekly data.** The source contains 156 weeks. Five missing ad-spend values are filled with the median, and two confirmed sales-recording errors (weeks 45 and 97) are removed, leaving 154 usable weeks.
2. **Tests on future weeks.** It trains on earlier weeks and evaluates on the latest 30 clean weeks. This time-based split reflects real forecasting and avoids training on future information.
3. **Compares candidate models.** Linear regression, a decision tree, and a random forest are scored using mean absolute error (MAE), root mean squared error (RMSE), and R-squared. The app adds error, bias, overfitting, and rolling-backtest views.
4. **Checks for data leakage.** Store visitors are excluded because they are only known after a week ends. Using them would make test scores look better without helping a real plan.
5. **Connects forecasts to a decision.** The planner estimates sales, an uncertainty range, gross profit, and profit after ads across possible budgets. It warns when a plan is outside the ad-spend range represented in the training data.

## Results

Scores below are from the latest 30 clean weeks, held out from training. MAE and RMSE are in thousands of EGP; lower is better. Higher R-squared indicates more of the sales variation is explained.

| Model | MAE (EGP k) | RMSE (EGP k) | R-squared |
|---|---:|---:|---:|
| **Linear regression (selected)** | **23.31** | **33.55** | **0.891** |
| Random forest | 34.87 | 44.73 | 0.806 |
| Decision tree | 43.05 | 56.11 | 0.694 |

**Business finding:** the selected model associates each additional EGP 1,000 of ad spend with about EGP 2,580 in additional weekly sales. At the assumed 30% gross margin, that is about **EGP 0.77 in gross profit per EGP spent on ads**. Under these assumptions, the additional ad spend does not pay for itself.

This is an observational analysis, not a causal experiment. The relationship between advertising and sales may be affected by other factors. A randomized test across weeks or regions would be needed to establish the causal effect of changing ad spend.

## The App

The Streamlit app has four pages:

| Page | Purpose |
|---|---|
| **Overview** | Summarizes the business question, key result, model scores, and analysis pipeline. |
| **Notebook walkthrough** | Explains the analysis step by step, including data cleaning, time-based evaluation, model choices, and the business recommendation. |
| **Model comparison** | Compares forecasts and error metrics; explores forecast-versus-actual plots, error patterns, rolling backtests, overfitting, and model drivers. |
| **Weekly planner** | Lets you set a budget and weekly conditions, then inspect the forecast, likely range, profit after ads, budget scenarios, and model cross-checks. |

Use **Data & evaluation settings** in the sidebar to upload a CSV or change the number of test weeks. The analysis pages refit their models using the selected data and settings.

## Quick Start

Requires Python 3.10 or newer. From the project folder, run the launcher. On Windows, select a compatible interpreter explicitly if `python` points to an older installation:

```powershell
py -3.10 start.py app
```

On macOS or Linux:

```bash
python3 start.py app
```

The launcher installs missing requirements, uses the saved model when available, and opens the app at [http://localhost:8501](http://localhost:8501). Press `Ctrl+C` in the terminal to stop it.

Other launcher options:

```bash
python start.py             # interactive menu
python start.py compare     # open the model comparison page
python start.py planner     # open the weekly planner
python start.py api         # start the API at http://localhost:8000/docs
python start.py all         # start both the app and API
python start.py train       # retrain the production model
python start.py test        # run the test suite
```

On Windows, replace `python` in these commands with `py -3.10` if needed.

## API

Start the API with `python start.py api`, then open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive documentation.

Example request:

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"ad_spend_k":80,"discount_pct":10,"is_holiday_period":0,"competitor_promo":1}'
```

The response includes the sales forecast, likely range, estimated ad-profit return, and any out-of-range warnings. `POST /forecast/batch` accepts multiple plans.

## Project Structure

```text
data/                       Weekly sales dataset
src/nilemart/               Data cleaning, training, and prediction package
notebooks/                  Model-development notebook
models/model.json           Portable production model and metadata
reports/metrics.json        Latest training and evaluation results
app/                        Streamlit app and page views
api/                        FastAPI service
tests/                      Pipeline, API, and app tests
Documentation/              Full project documentation
start.py                    Cross-platform launcher
```

## Retraining and Tests

Add new weekly rows to `data/nilemart_weekly_sales.csv`, then run:

```bash
python start.py train
python start.py test
```

Training compares candidate models on the latest 30 weeks, selects the one with the lowest MAE, refits it on all clean data, and updates `models/model.json` and `reports/metrics.json`. The model is stored as JSON. GitHub Actions runs the training and test workflow on pushes.
