# Sales Forecasting Analytics

## Project overview

This repository analyzes the existing retail transactions in `Data/train.csv`.
The Streamlit dashboard retains its five pages: Sales Overview, Forecast
Explorer, Anomaly Report, Product Segments, and Executive Summary.

The forecast target is **sales revenue**. The dataset does not contain quantity,
profit, discount, or inventory fields, so the project does not forecast unit
demand or make inventory recommendations.

## Methodology

- Dates are parsed as day-month-year; the 11 missing Postal Code values are
  retained.
- Monthly revenue is compared on one chronological test period: the final three
  observed months. SARIMA, Prophet, XGBoost, and a naive last-value baseline
  share these dates.
- XGBoost uses lag and shifted rolling features. Multi-month test predictions
  are recursive and use earlier predictions, not actual test-period sales.
- Model results report MAE, RMSE, WAPE, and bias. The three-month test period is
  small and does not establish production readiness.
- Weekly anomaly results exclude partial boundary weeks and show the actual
  week, revenue, high/low direction, and which methods flagged it. Flags do not
  explain their causes.
- Product segments are ABC revenue-contribution groups (approximately the first
  80%, next to 95%, and remaining revenue), not growth or demand categories.

## Current monthly holdout results

Results from the current three-month run:

| Model | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Naive baseline | 14,537.51 | 19,112.18 | 15.66% | -7.17% |
| SARIMA | 18,031.40 | 19,009.18 | 19.43% | -7.26% |
| Prophet | 20,250.79 | 22,318.41 | 21.82% | -16.78% |
| XGBoost | 16,658.09 | 19,688.76 | 17.95% | -17.95% |

The naive baseline had the lowest WAPE on this short holdout and was selected
for a three-month future forecast. Future forecast dates are 2019-01-31,
2019-02-28, and 2019-03-31. These scores are not a guarantee of future
performance. The previously reported 13.42% XGBoost MAPE is not used.

## Installation and run instructions (Windows)

Run commands from the repository root.

Install the existing application and notebook dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the monthly model comparison and future forecast:

```powershell
python -m sales_forecasting --data Data\train.csv --output Forecasting_Output --charts Charts
```

Start the Streamlit dashboard:

```powershell
python -m streamlit run app.py
```

Open the existing analysis notebook:

```powershell
jupyter notebook Notebook\analysis.ipynb
```

Run tests:

```powershell
python -B -m unittest discover -s tests -v
```

The application resolves the dataset and chart paths relative to `app.py`. The
notebook locates the project root from its working directory or a parent
directory. Monthly output files are written to `Forecasting_Output/`, and
monthly charts are written to `Charts/`.

For a complete walkthrough of the data, methodology, dashboard pages, forecast
results, anomaly dates, ABC segments, outputs, tests, and limitations, see
[PROJECT_GUIDE.md](./PROJECT_GUIDE.md).

## Limitations

The data has less than four complete annual cycles, no external regressors, and
only three test months in the current comparison. Anomaly flags are statistical
outliers, not explanations. ABC groups summarize historical revenue
concentration only. No causal, unit-demand, profit, or inventory conclusions
can be drawn from the available columns.

## Author

Roshan Manjal
