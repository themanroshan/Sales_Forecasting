"""Sales revenue forecasting utilities and the monthly portfolio pipeline."""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


WEEK_FREQUENCY = "W-SUN"
SEASONAL_PERIOD = 52
DEFAULT_HORIZON = 4
DEFAULT_INITIAL_TRAIN = 156
DEFAULT_STEP = 4
DEFAULT_MODELS = (
    "naive",
    "seasonal_naive",
    "moving_average_4",
    "sarima",
    "prophet",
    "xgboost",
)
MODEL_COMPLEXITY = {
    "naive": 0,
    "moving_average_4": 1,
    "seasonal_naive": 2,
    "sarima": 3,
    "prophet": 4,
    "xgboost": 5,
}


class ForecastingDataError(ValueError):
    """Raised when the weekly series is not suitable for time-series evaluation."""


class MissingModelDependencyError(ImportError):
    """Raised when an optional forecasting model dependency is unavailable."""


class ModelConvergenceError(RuntimeError):
    """Raised when a fitted model reports that optimization did not converge."""


@dataclass(frozen=True)
class ForecastResult:
    dates: pd.DatetimeIndex
    forecast: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastingRun:
    weekly_sales: pd.Series
    weekly_diagnostics: dict[str, Any]
    backtest_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_metrics: pd.DataFrame
    future_forecast: pd.DataFrame
    selected_model: str
    selection_reason: str
    output_directory: Path


@dataclass(frozen=True)
class MonthlyForecastRun:
    monthly_sales: pd.Series
    test_predictions: pd.DataFrame
    model_metrics: pd.DataFrame
    future_forecast: pd.DataFrame
    selected_model: str
    output_directory: Path


def _week_end_dates(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period(WEEK_FREQUENCY).dt.end_time.dt.normalize()


def aggregate_weekly_sales(
    transactions: pd.DataFrame, drop_partial_boundary_weeks: bool = True
) -> pd.Series:
    """Aggregate transaction revenue by Sunday-ending week without filling gaps.

    Boundary weeks that are not fully represented by the available source date
    range are excluded by default so they are not treated as full weeks.
    """
    required = ("Order Date", "Sales")
    missing_columns = [column for column in required if column not in transactions]
    if missing_columns:
        raise ForecastingDataError(
            f"Missing required columns for weekly aggregation: {', '.join(missing_columns)}"
        )
    if transactions.empty:
        raise ForecastingDataError("Cannot aggregate an empty transaction dataset.")

    dates = pd.to_datetime(transactions["Order Date"], errors="coerce")
    sales = pd.to_numeric(transactions["Sales"], errors="coerce")
    finite_sales = np.isfinite(sales.to_numpy(dtype=float, na_value=np.nan))
    invalid_sales = sales.isna() | ~pd.Series(finite_sales, index=transactions.index) | sales.lt(0)
    if dates.isna().any():
        raise ForecastingDataError(
            f"Cannot aggregate {int(dates.isna().sum())} rows with invalid Order Date values."
        )
    if invalid_sales.any():
        raise ForecastingDataError(
            f"Cannot aggregate {int(invalid_sales.sum())} rows with invalid Sales values."
        )

    week_ends = _week_end_dates(dates)
    weekly = sales.groupby(week_ends).sum().sort_index()
    weekly.index = pd.DatetimeIndex(weekly.index, name="Week Ending")
    weekly.name = "Sales"

    excluded_partial_weeks: list[str] = []
    if drop_partial_boundary_weeks and not weekly.empty:
        source_start = dates.min().normalize()
        source_end = dates.max().normalize()
        first_period = source_start.to_period(WEEK_FREQUENCY)
        last_period = source_end.to_period(WEEK_FREQUENCY)
        partial_periods: set[pd.Period] = set()
        if source_start > first_period.start_time.normalize():
            partial_periods.add(first_period)
        if source_end < last_period.end_time.normalize():
            partial_periods.add(last_period)
        if partial_periods:
            excluded_partial_weeks = [
                period.end_time.normalize().strftime("%Y-%m-%d")
                for period in sorted(partial_periods)
            ]
            retained = ~weekly.index.isin(
                pd.to_datetime(excluded_partial_weeks)
            )
            weekly = weekly.loc[retained]

    weekly.attrs["excluded_partial_boundary_weeks"] = excluded_partial_weeks
    weekly.attrs["partial_boundary_policy"] = (
        "Excluded partial boundary weeks; missing interior weeks are not filled."
        if drop_partial_boundary_weeks
        else "Included boundary weeks as observed; missing weeks are not filled."
    )
    if weekly.empty:
        raise ForecastingDataError(
            "No complete Sunday-ending weeks remain after checking partial boundaries."
        )
    return weekly.astype(float)


def missing_week_dates(weekly_sales: pd.Series) -> pd.DatetimeIndex:
    """Return absent Sunday-ending weeks; never infer or fill their sales."""
    if weekly_sales.empty:
        return pd.DatetimeIndex([])
    expected = pd.date_range(
        weekly_sales.index.min(), weekly_sales.index.max(), freq=WEEK_FREQUENCY
    )
    return expected.difference(pd.DatetimeIndex(weekly_sales.index))


def validate_weekly_sales(
    weekly_sales: pd.Series, require_contiguous: bool = False
) -> pd.Series:
    """Validate weekly index, types, and values without filling absent weeks."""
    if not isinstance(weekly_sales, pd.Series) or weekly_sales.empty:
        raise ForecastingDataError("Weekly sales must be a non-empty pandas Series.")
    if not isinstance(weekly_sales.index, pd.DatetimeIndex):
        raise ForecastingDataError("Weekly sales must use a DatetimeIndex.")
    if weekly_sales.index.hasnans:
        raise ForecastingDataError("Weekly sales contain invalid week-ending dates.")
    if not weekly_sales.index.is_monotonic_increasing:
        raise ForecastingDataError("Weekly sales dates must be sorted chronologically.")
    if weekly_sales.index.has_duplicates:
        raise ForecastingDataError("Weekly sales dates must be unique.")
    if not (weekly_sales.index.dayofweek == 6).all():
        raise ForecastingDataError("Every weekly date must be a Sunday week ending.")

    values = pd.to_numeric(weekly_sales, errors="coerce")
    finite_values = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    if values.isna().any() or not finite_values.all():
        raise ForecastingDataError("Weekly sales contain missing or non-finite values.")
    if values.lt(0).any():
        raise ForecastingDataError("Weekly sales cannot be negative.")
    if require_contiguous:
        missing = missing_week_dates(weekly_sales)
        if len(missing):
            examples = ", ".join(date.strftime("%Y-%m-%d") for date in missing[:5])
            raise ForecastingDataError(
                f"Found {len(missing)} missing weeks ({examples}); they were not "
                "filled because the source does not establish whether they are "
                "zero-sales weeks or incomplete records."
            )
    return values.astype(float).rename(weekly_sales.name or "Sales")


def inspect_weekly_sales(
    weekly_sales: pd.Series,
    excluded_partial_boundary_weeks: Sequence[str] = (),
) -> dict[str, Any]:
    """Summarize weekly completeness, trend, seasonality, and IQR outliers."""
    series = validate_weekly_sales(weekly_sales)
    missing = missing_week_dates(series)
    expected_weeks = len(
        pd.date_range(series.index.min(), series.index.max(), freq=WEEK_FREQUENCY)
    )
    values = series.to_numpy(dtype=float)
    x_values = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x_values, values, 1)[0]) if len(values) > 1 else 0.0
    lag_52_correlation = (
        float(series.autocorr(lag=SEASONAL_PERIOD))
        if len(series) > SEASONAL_PERIOD
        else None
    )
    if lag_52_correlation is not None and not np.isfinite(lag_52_correlation):
        lag_52_correlation = None
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    unusual = series.loc[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]

    return {
        "week_ending_convention": WEEK_FREQUENCY,
        "weekly_observations": int(len(series)),
        "expected_weeks_in_covered_range": int(expected_weeks),
        "missing_week_count": int(len(missing)),
        "missing_weeks": [date.strftime("%Y-%m-%d") for date in missing],
        "zero_sales_weeks": int(series.eq(0).sum()),
        "first_week_ending": series.index.min().strftime("%Y-%m-%d"),
        "last_week_ending": series.index.max().strftime("%Y-%m-%d"),
        "minimum_weekly_sales": float(series.min()),
        "maximum_weekly_sales": float(series.max()),
        "mean_weekly_sales": float(series.mean()),
        "linear_trend_sales_per_week": slope,
        "linear_trend_direction": (
            "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        ),
        "lag_52_sales_correlation": lag_52_correlation,
        "yearly_seasonality_cycles_available": float(len(series) / SEASONAL_PERIOD),
        "unusual_weeks_iqr_rule": [
            {"week_ending": date.strftime("%Y-%m-%d"), "sales": float(value)}
            for date, value in unusual.items()
        ],
        "excluded_partial_boundary_weeks": list(excluded_partial_boundary_weeks),
        "missing_week_decision": (
            "No missing weeks were filled. An absent week cannot be assumed to have "
            "zero sales without source evidence."
        ),
        "boundary_week_decision": (
            "Partial first or last weeks are excluded from model training by default."
        ),
        "annual_seasonality_note": (
            "Yearly seasonality is considered exploratory: the available history "
            "contains fewer than five annual cycles."
        ),
    }


def _validated_training_series(
    training_sales: pd.Series, require_seasonal_history: bool = False
) -> pd.Series:
    series = validate_weekly_sales(training_sales, require_contiguous=True)
    if require_seasonal_history and len(series) < SEASONAL_PERIOD:
        raise ForecastingDataError(
            f"At least {SEASONAL_PERIOD} weeks are required for seasonal-naive forecasting."
        )
    series.attrs.update(training_sales.attrs)
    return series


def future_week_dates(
    weekly_sales: pd.Series, horizon: int = DEFAULT_HORIZON
) -> pd.DatetimeIndex:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one week.")
    series = _validated_training_series(weekly_sales)
    return pd.date_range(
        series.index[-1] + pd.Timedelta(weeks=1),
        periods=horizon,
        freq=WEEK_FREQUENCY,
        name=series.index.name or "Week Ending",
    )


def naive_forecast(
    training_sales: pd.Series, horizon: int = DEFAULT_HORIZON
) -> np.ndarray:
    series = _validated_training_series(training_sales)
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one week.")
    return np.repeat(float(series.iloc[-1]), horizon)


def seasonal_naive_forecast(
    training_sales: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    seasonal_period: int = SEASONAL_PERIOD,
) -> np.ndarray:
    series = _validated_training_series(training_sales)
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one week.")
    if seasonal_period < 1 or len(series) < seasonal_period:
        raise ForecastingDataError(
            f"Seasonal-naive forecasting needs at least {seasonal_period} weeks."
        )
    previous_season = series.to_numpy(dtype=float)[-seasonal_period:]
    repeats = int(np.ceil(horizon / seasonal_period))
    return np.tile(previous_season, repeats)[:horizon]


def moving_average_forecast(
    training_sales: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    window: int = 4,
) -> np.ndarray:
    series = _validated_training_series(training_sales)
    if horizon < 1 or window < 1:
        raise ValueError("Forecast horizon and moving-average window must be positive.")
    if len(series) < window:
        raise ForecastingDataError(
            f"Moving-average forecasting needs at least {window} weeks."
        )
    return np.repeat(float(series.iloc[-window:].mean()), horizon)


def make_xgboost_features(
    weekly_sales: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build target-week features from observations strictly before that week."""
    series = _validated_training_series(weekly_sales)
    features = pd.DataFrame(index=series.index)
    for lag in (1, 2, 3, 4, SEASONAL_PERIOD):
        features[f"lag_{lag}"] = series.shift(lag)

    prior_sales = series.shift(1)
    features["previous_4_week_mean"] = prior_sales.rolling(4).mean()
    features["previous_13_week_mean"] = prior_sales.rolling(13).mean()
    features["previous_13_week_std"] = prior_sales.rolling(13).std()

    week_of_year = pd.Series(
        series.index.isocalendar().week.to_numpy(dtype=float),
        index=series.index,
    )
    angle = 2 * np.pi * week_of_year / SEASONAL_PERIOD
    features["week_of_year_sin"] = np.sin(angle)
    features["week_of_year_cos"] = np.cos(angle)

    target = series.rename("target")
    valid_rows = features.notna().all(axis=1)
    return features.loc[valid_rows], target.loc[valid_rows]


def _latest_xgboost_features(
    history: pd.Series, forecast_date: pd.Timestamp
) -> pd.DataFrame:
    if len(history) < SEASONAL_PERIOD:
        raise ForecastingDataError(
            f"XGBoost features need at least {SEASONAL_PERIOD} historical weeks."
        )
    prior_sales = history.astype(float)
    week_of_year = float(forecast_date.isocalendar().week)
    angle = 2 * np.pi * week_of_year / SEASONAL_PERIOD
    row = {
        "lag_1": float(prior_sales.iloc[-1]),
        "lag_2": float(prior_sales.iloc[-2]),
        "lag_3": float(prior_sales.iloc[-3]),
        "lag_4": float(prior_sales.iloc[-4]),
        "lag_52": float(prior_sales.iloc[-SEASONAL_PERIOD]),
        "previous_4_week_mean": float(prior_sales.iloc[-4:].mean()),
        "previous_13_week_mean": float(prior_sales.iloc[-13:].mean()),
        "previous_13_week_std": float(prior_sales.iloc[-13:].std()),
        "week_of_year_sin": float(np.sin(angle)),
        "week_of_year_cos": float(np.cos(angle)),
    }
    return pd.DataFrame([row])


def recursive_xgboost_forecast(
    training_sales: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    estimator_factory: Callable[[], Any] | None = None,
) -> ForecastResult:
    """Fit on past targets and recursively feed predictions into later features."""
    series = _validated_training_series(training_sales, require_seasonal_history=True)
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one week.")
    features, target = make_xgboost_features(series)
    if features.empty:
        raise ForecastingDataError("Not enough training weeks to build XGBoost features.")

    if estimator_factory is None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise MissingModelDependencyError(
                "XGBoost is unavailable; install it with "
                "`python -m pip install xgboost`."
            ) from exc
        estimator_factory = lambda: XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )

    model = estimator_factory()
    model.fit(features, target)
    dates = future_week_dates(series, horizon)
    history = series.copy()
    predictions: list[float] = []
    for date in dates:
        row = _latest_xgboost_features(history, date)
        prediction = float(np.asarray(model.predict(row)).reshape(-1)[0])
        predictions.append(prediction)
        history.loc[date] = prediction

    return ForecastResult(dates=dates, forecast=np.asarray(predictions, dtype=float))


def _forecast_sarima(
    training_sales: pd.Series, horizon: int
) -> ForecastResult:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise MissingModelDependencyError(
            "Statsmodels is unavailable; install it with "
            "`python -m pip install statsmodels`."
        ) from exc

    series = _validated_training_series(training_sales)
    dates = future_week_dates(series, horizon)
    model_index = pd.DatetimeIndex(series.index, freq=WEEK_FREQUENCY)
    model_series = pd.Series(
        series.to_numpy(dtype=float), index=model_index, name=series.name
    )
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = SARIMAX(
            model_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
        ).fit(disp=False, maxiter=100)
        captured_warnings = [str(item.message) for item in caught]

    if not bool(getattr(fitted, "mle_retvals", {}).get("converged", True)):
        details = "; ".join(captured_warnings) or "optimizer reported converged=False"
        raise ModelConvergenceError(f"SARIMA did not converge: {details}")

    forecast = fitted.get_forecast(steps=horizon)
    intervals = forecast.conf_int(alpha=0.2).to_numpy(dtype=float)
    return ForecastResult(
        dates=dates,
        forecast=np.asarray(forecast.predicted_mean, dtype=float),
        lower=intervals[:, 0],
        upper=intervals[:, 1],
        warnings=tuple(captured_warnings),
    )


def _forecast_prophet(training_sales: pd.Series, horizon: int) -> ForecastResult:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise MissingModelDependencyError(
            "Prophet is unavailable; install it with "
            "`python -m pip install prophet`."
        ) from exc

    series = _validated_training_series(training_sales)
    dates = future_week_dates(series, horizon)
    training = pd.DataFrame(
        {"ds": series.index, "y": series.to_numpy(dtype=float)}
    )
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.8,
    )
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(training)
        prediction = model.predict(pd.DataFrame({"ds": dates}))
        captured_warnings = [str(item.message) for item in caught]

    return ForecastResult(
        dates=dates,
        forecast=prediction["yhat"].to_numpy(dtype=float),
        lower=prediction["yhat_lower"].to_numpy(dtype=float),
        upper=prediction["yhat_upper"].to_numpy(dtype=float),
        warnings=tuple(captured_warnings),
    )


def forecast_model(
    model_name: str,
    training_sales: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    xgboost_estimator_factory: Callable[[], Any] | None = None,
) -> ForecastResult:
    """Fit one configured model on history only and return its future forecast."""
    if model_name == "naive":
        series = _validated_training_series(training_sales)
        return ForecastResult(
            future_week_dates(series, horizon), naive_forecast(series, horizon)
        )
    if model_name == "seasonal_naive":
        series = _validated_training_series(training_sales)
        return ForecastResult(
            future_week_dates(series, horizon),
            seasonal_naive_forecast(series, horizon),
        )
    if model_name == "moving_average_4":
        series = _validated_training_series(training_sales)
        return ForecastResult(
            future_week_dates(series, horizon),
            moving_average_forecast(series, horizon),
        )
    if model_name == "sarima":
        return _forecast_sarima(training_sales, horizon)
    if model_name == "prophet":
        return _forecast_prophet(training_sales, horizon)
    if model_name == "xgboost":
        return recursive_xgboost_forecast(
            training_sales,
            horizon,
            estimator_factory=xgboost_estimator_factory,
        )
    raise ValueError(f"Unknown forecast model: {model_name}")


def evaluate_forecast_metrics(
    actual: Sequence[float], forecast: Sequence[float]
) -> dict[str, float | int]:
    """Calculate pooled-safe error metrics; WAPE and bias are percentages."""
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    forecast_values = np.asarray(forecast, dtype=float).reshape(-1)
    if actual_values.size != forecast_values.size:
        raise ValueError("Actual and forecast arrays must have equal lengths.")
    if actual_values.size == 0:
        raise ValueError("At least one actual/forecast pair is required.")
    if not np.isfinite(actual_values).all() or not np.isfinite(forecast_values).all():
        raise ValueError("Metrics require finite actual and forecast values.")

    errors = forecast_values - actual_values
    denominator = float(np.abs(actual_values).sum())
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "wape_pct": (
            float(np.abs(errors).sum() / denominator * 100)
            if denominator > 0
            else float("nan")
        ),
        "bias_pct": (
            float(errors.sum() / denominator * 100)
            if denominator > 0
            else float("nan")
        ),
        "evaluated_predictions": int(actual_values.size),
    }


def rolling_origin_backtest(
    weekly_sales: pd.Series,
    initial_train: int = DEFAULT_INITIAL_TRAIN,
    horizon: int = DEFAULT_HORIZON,
    step: int = DEFAULT_STEP,
    models: Sequence[str] = DEFAULT_MODELS,
    xgboost_estimator_factory: Callable[[], Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on identical chronological forecast origins."""
    series = _validated_training_series(weekly_sales, require_seasonal_history=True)
    if initial_train < SEASONAL_PERIOD:
        raise ValueError(
            f"Initial training window must include at least {SEASONAL_PERIOD} weeks."
        )
    if horizon < 1 or step < 1:
        raise ValueError("Forecast horizon and backtest step must be positive.")
    if initial_train + horizon > len(series):
        raise ForecastingDataError(
            f"Need more than {initial_train + horizon - 1} weeks for the requested "
            "initial window and forecast horizon."
        )
    if not models:
        raise ValueError("At least one model must be selected for backtesting.")

    origins = list(range(initial_train, len(series) - horizon + 1, step))
    if not origins:
        raise ForecastingDataError("No rolling-origin folds fit the supplied series.")

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_number, origin in enumerate(origins, start=1):
        training = series.iloc[:origin]
        actuals = series.iloc[origin : origin + horizon]
        dates = pd.DatetimeIndex(actuals.index)
        if len(actuals) != horizon:
            raise ForecastingDataError(
                f"Fold {fold_number} does not contain the full {horizon}-week horizon."
            )

        for model_name in models:
            failure_reason = ""
            warning_text = ""
            lower: np.ndarray | None = None
            upper: np.ndarray | None = None
            try:
                result = forecast_model(
                    model_name,
                    training,
                    horizon,
                    xgboost_estimator_factory=xgboost_estimator_factory,
                )
                forecast_values = np.asarray(result.forecast, dtype=float).reshape(-1)
                if len(result.dates) != horizon or len(forecast_values) != horizon:
                    raise ValueError(
                        f"{model_name} returned {len(forecast_values)} predictions "
                        f"for a {horizon}-week horizon."
                    )
                if not np.isfinite(forecast_values).all():
                    raise ValueError(f"{model_name} returned non-finite predictions.")
                if not pd.DatetimeIndex(result.dates).equals(dates):
                    raise ValueError(
                        f"{model_name} forecast dates do not match the test horizon."
                    )
                lower = result.lower
                upper = result.upper
                warning_text = " | ".join(result.warnings)
                status = "ok"
            except Exception as exc:
                forecast_values = np.full(horizon, np.nan, dtype=float)
                status = "failed"
                failure_reason = f"{type(exc).__name__}: {exc}"

            if status == "ok":
                fold_metrics = evaluate_forecast_metrics(
                    actuals.to_numpy(dtype=float), forecast_values
                )
            else:
                fold_metrics = {
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "wape_pct": float("nan"),
                    "bias_pct": float("nan"),
                    "evaluated_predictions": 0,
                }

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "training_observations": int(len(training)),
                    "train_end": training.index[-1],
                    "forecast_start": dates[0],
                    "forecast_end": dates[-1],
                    "status": status,
                    "failure_reason": failure_reason,
                    "warnings": warning_text,
                    **fold_metrics,
                }
            )
            for position, date in enumerate(dates):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "training_observations": int(len(training)),
                        "train_end": training.index[-1],
                        "forecast_origin": dates[0],
                        "target_date": date,
                        "actual": float(actuals.iloc[position]),
                        "forecast": float(forecast_values[position]),
                        "lower": (
                            float(np.asarray(lower).reshape(-1)[position])
                            if lower is not None
                            else float("nan")
                        ),
                        "upper": (
                            float(np.asarray(upper).reshape(-1)[position])
                            if upper is not None
                            else float("nan")
                        ),
                        "status": status,
                        "failure_reason": failure_reason,
                        "warnings": warning_text,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    return predictions, fold_metrics


def compare_models(
    backtest_predictions: pd.DataFrame, fold_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Pool metrics only for models that completed every shared fold."""
    required_prediction_columns = {
        "model",
        "fold",
        "target_date",
        "actual",
        "forecast",
        "status",
    }
    missing_columns = required_prediction_columns.difference(
        backtest_predictions.columns
    )
    if missing_columns:
        raise ValueError(
            "Backtest predictions are missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if backtest_predictions.empty:
        raise ValueError("Backtest predictions are empty.")

    model_names = list(dict.fromkeys(backtest_predictions["model"].tolist()))
    expected_dates: set[pd.Timestamp] | None = None
    for model_name in model_names:
        model_rows = backtest_predictions.loc[
            backtest_predictions["model"].eq(model_name)
        ]
        date_set = set(pd.to_datetime(model_rows["target_date"]))
        if expected_dates is None:
            expected_dates = date_set
        elif date_set != expected_dates:
            raise ForecastingDataError(
                "Models were not evaluated on identical target dates."
            )
        actual_by_date = model_rows.drop_duplicates("target_date").set_index(
            "target_date"
        )["actual"]
        if not np.isfinite(actual_by_date.to_numpy(dtype=float)).all():
            raise ValueError(f"{model_name} has non-finite backtest actuals.")

    comparison_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        rows = backtest_predictions.loc[
            backtest_predictions["model"].eq(model_name)
        ].sort_values(["fold", "target_date"])
        expected_predictions = len(rows)
        successful = rows.loc[
            rows["status"].eq("ok") & np.isfinite(rows["forecast"])
        ]
        fold_rows = fold_metrics.loc[fold_metrics["model"].eq(model_name)]
        expected_folds = int(fold_rows["fold"].nunique())
        complete = (
            len(successful) == expected_predictions
            and expected_predictions > 0
            and len(fold_rows) == expected_folds
            and fold_rows["status"].eq("ok").all()
        )
        if complete:
            metric_values = evaluate_forecast_metrics(
                successful["actual"].to_numpy(dtype=float),
                successful["forecast"].to_numpy(dtype=float),
            )
            status = "complete"
        else:
            metric_values = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "wape_pct": float("nan"),
                "bias_pct": float("nan"),
                "evaluated_predictions": int(len(successful)),
            }
            status = "unavailable" if len(successful) == 0 else "incomplete"

        failure_reasons = (
            fold_rows["failure_reason"]
            if "failure_reason" in fold_rows
            else pd.Series(dtype=object)
        )
        failures = [
            str(value)
            for value in failure_reasons.dropna().unique()
            if str(value)
        ]
        comparison_rows.append(
            {
                "model": model_name,
                "status": status,
                "folds_expected": expected_folds,
                "folds_completed": int(fold_rows["status"].eq("ok").sum()),
                "predictions_expected": int(expected_predictions),
                "predictions_evaluated": int(metric_values["evaluated_predictions"]),
                "failed_folds": int(fold_rows["status"].ne("ok").sum()),
                "failure_reasons": " | ".join(failures),
                "mae": metric_values["mae"],
                "rmse": metric_values["rmse"],
                "wape_pct": metric_values["wape_pct"],
                "bias_pct": metric_values["bias_pct"],
            }
        )

    complete_rows = [
        row for row in comparison_rows if row["status"] == "complete"
    ]
    if len(complete_rows) > 1:
        expected_count = complete_rows[0]["predictions_expected"]
        expected_keys = set(
            backtest_predictions.loc[
                backtest_predictions["model"].eq(complete_rows[0]["model"]),
                ["fold", "target_date"],
            ].itertuples(index=False, name=None)
        )
        for row in complete_rows[1:]:
            keys = set(
                backtest_predictions.loc[
                    backtest_predictions["model"].eq(row["model"]),
                    ["fold", "target_date"],
                ].itertuples(index=False, name=None)
            )
            if row["predictions_expected"] != expected_count or keys != expected_keys:
                raise ForecastingDataError(
                    "Complete models do not share identical backtesting periods."
                )
    return pd.DataFrame(comparison_rows)


def select_best_model(
    model_metrics: pd.DataFrame, wape_tolerance_pct_points: float = 1.0
) -> tuple[str, str]:
    """Choose lowest pooled WAPE, preferring simpler models within a 1-point band."""
    eligible = model_metrics.loc[
        model_metrics["status"].eq("complete")
        & np.isfinite(model_metrics["wape_pct"])
    ].copy()
    if eligible.empty:
        eligible = model_metrics.loc[
            model_metrics["status"].eq("complete")
            & np.isfinite(model_metrics["mae"])
        ].copy()
        if eligible.empty:
            raise ForecastingDataError(
                "No complete model has finite metrics; a future forecast cannot be selected."
            )
        minimum_mae = float(eligible["mae"].min())
        candidates = eligible.loc[eligible["mae"].eq(minimum_mae)]
        selected = min(
            candidates["model"], key=lambda name: MODEL_COMPLEXITY.get(name, 100)
        )
        return str(selected), (
            "Pooled WAPE was undefined because total absolute actual sales were zero; "
            "selected the complete model with lowest MAE."
        )

    best_wape = float(eligible["wape_pct"].min())
    candidates = eligible.loc[
        eligible["wape_pct"] <= best_wape + wape_tolerance_pct_points
    ]
    selected = min(
        candidates["model"], key=lambda name: MODEL_COMPLEXITY.get(name, 100)
    )
    if selected == eligible.loc[eligible["wape_pct"].eq(best_wape), "model"].iloc[0]:
        reason = f"Selected the lowest pooled WAPE ({best_wape:.3f}%)."
    else:
        reason = (
            f"Selected {selected} as the simplest complete model within "
            f"{wape_tolerance_pct_points:.1f} WAPE percentage point(s) of the "
            f"minimum ({best_wape:.3f}%)."
        )
    return str(selected), reason


def forecast_future(
    weekly_sales: pd.Series,
    model_name: str,
    horizon: int = DEFAULT_HORIZON,
    xgboost_estimator_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """Refit a selected model on all available history and forecast future weeks."""
    series = _validated_training_series(
        weekly_sales,
        require_seasonal_history=model_name in ("seasonal_naive", "xgboost"),
    )
    result = forecast_model(
        model_name,
        series,
        horizon,
        xgboost_estimator_factory=xgboost_estimator_factory,
    )
    if len(result.dates) != horizon:
        raise ForecastingDataError("Final forecast returned an incorrect horizon.")
    if (pd.DatetimeIndex(result.dates) <= series.index[-1]).any():
        raise ForecastingDataError("Future forecast dates overlap historical dates.")
    return pd.DataFrame(
        {
            "date": result.dates,
            "forecast": np.asarray(result.forecast, dtype=float),
            "lower": (
                np.asarray(result.lower, dtype=float)
                if result.lower is not None
                else np.full(horizon, np.nan)
            ),
            "upper": (
                np.asarray(result.upper, dtype=float)
                if result.upper is not None
                else np.full(horizon, np.nan)
            ),
            "model": model_name,
            "status": "future_prediction",
        }
    )


def _write_charts(
    weekly_sales: pd.Series,
    predictions: pd.DataFrame,
    model_metrics: pd.DataFrame,
    future_forecast: pd.DataFrame,
    chart_directory: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete_models = model_metrics.loc[
        model_metrics["status"].eq("complete"), "model"
    ].tolist()
    plt.figure(figsize=(13, 6))
    plt.plot(weekly_sales.index, weekly_sales.values, color="black", label="Actual")
    for model_name in complete_models:
        model_rows = predictions.loc[
            predictions["model"].eq(model_name)
            & predictions["status"].eq("ok")
        ]
        plt.plot(
            model_rows["target_date"],
            model_rows["forecast"],
            marker=".",
            linewidth=1,
            label=model_name,
        )
    plt.title("Weekly sales: rolling-origin actuals and out-of-fold forecasts")
    plt.xlabel("Week ending Sunday")
    plt.ylabel("Sales revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_directory / "weekly_backtest_actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    residual_data = []
    residual_labels = []
    for model_name in complete_models:
        model_rows = predictions.loc[
            predictions["model"].eq(model_name)
            & predictions["status"].eq("ok")
        ]
        residual_data.append(
            (model_rows["forecast"] - model_rows["actual"]).to_numpy(dtype=float)
        )
        residual_labels.append(model_name)
    if residual_data:
        plt.boxplot(residual_data, tick_labels=residual_labels, showfliers=True)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(rotation=20, ha="right")
    plt.title("Rolling-origin forecast residuals (forecast − actual)")
    plt.ylabel("Revenue error")
    plt.tight_layout()
    plt.savefig(chart_directory / "weekly_backtest_residuals.png", dpi=150)
    plt.close()

    plotted_metrics = model_metrics.loc[
        model_metrics["status"].eq("complete")
        & np.isfinite(model_metrics["wape_pct"])
    ].sort_values("wape_pct")
    plt.figure(figsize=(10, 5))
    if not plotted_metrics.empty:
        plt.bar(plotted_metrics["model"], plotted_metrics["wape_pct"])
        plt.xticks(rotation=20, ha="right")
    plt.title("Pooled rolling-origin WAPE by complete model")
    plt.ylabel("WAPE (%)")
    plt.tight_layout()
    plt.savefig(chart_directory / "weekly_model_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(13, 6))
    history = weekly_sales.iloc[-min(104, len(weekly_sales)) :]
    plt.plot(history.index, history.values, color="black", label="Historical weekly sales")
    future_dates = pd.to_datetime(future_forecast["date"])
    plt.plot(
        future_dates,
        future_forecast["forecast"],
        marker="o",
        label=f"Future forecast: {future_forecast['model'].iloc[0]}",
    )
    if future_forecast["lower"].notna().all() and future_forecast["upper"].notna().all():
        plt.fill_between(
            future_dates,
            future_forecast["lower"].to_numpy(dtype=float),
            future_forecast["upper"].to_numpy(dtype=float),
            alpha=0.2,
            label="80% model interval",
        )
    plt.axvline(weekly_sales.index[-1], color="grey", linestyle="--")
    plt.title("Future weekly total sales forecast (not backtest predictions)")
    plt.xlabel("Week ending Sunday")
    plt.ylabel("Sales revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_directory / "weekly_future_forecast.png", dpi=150)
    plt.close()


def run_forecasting_pipeline(
    data_path: str | Path,
    output_directory: str | Path,
    chart_directory: str | Path,
    initial_train: int = DEFAULT_INITIAL_TRAIN,
    horizon: int = DEFAULT_HORIZON,
    step: int = DEFAULT_STEP,
    models: Sequence[str] = DEFAULT_MODELS,
) -> ForecastingRun:
    """Run weekly diagnostics, backtesting, selection, and a future forecast."""
    from sales_data import load_sales_data

    transactions, _ = load_sales_data(data_path)
    weekly_sales = aggregate_weekly_sales(transactions)
    weekly_sales = _validated_training_series(
        weekly_sales, require_seasonal_history=True
    )
    excluded_partial_weeks = weekly_sales.attrs.get(
        "excluded_partial_boundary_weeks", []
    )
    diagnostics = inspect_weekly_sales(
        weekly_sales,
        excluded_partial_boundary_weeks=excluded_partial_weeks,
    )
    diagnostics.update(
        {
            "source_transaction_rows": int(len(transactions)),
            "source_first_order_date": transactions["Order Date"].min().strftime(
                "%Y-%m-%d"
            ),
            "source_last_order_date": transactions["Order Date"].max().strftime(
                "%Y-%m-%d"
            ),
        }
    )
    missing = missing_week_dates(weekly_sales)
    if len(missing):
        validate_weekly_sales(weekly_sales, require_contiguous=True)
    if diagnostics["weekly_observations"] < initial_train + horizon:
        raise ForecastingDataError(
            f"Only {diagnostics['weekly_observations']} complete weeks are available; "
            f"the requested initial window and horizon need at least "
            f"{initial_train + horizon}."
        )

    output_path = Path(output_directory)
    charts_path = Path(chart_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    charts_path.mkdir(parents=True, exist_ok=True)
    predictions, fold_metrics = rolling_origin_backtest(
        weekly_sales,
        initial_train=initial_train,
        horizon=horizon,
        step=step,
        models=models,
    )
    model_metrics = compare_models(predictions, fold_metrics)
    diagnostics["model_execution_statuses"] = (
        model_metrics[
            [
                "model",
                "status",
                "folds_completed",
                "folds_expected",
                "predictions_evaluated",
                "failure_reasons",
            ]
        ].to_dict(orient="records")
    )
    diagnostics["sarima_warnings"] = sorted(
        {
            str(message)
            for message in fold_metrics.loc[
                fold_metrics["model"].eq("sarima"), "warnings"
            ].dropna()
            if str(message)
        }
    )

    failed_future_fits: list[str] = []
    selected_model = ""
    selection_reason = ""
    future_forecast: pd.DataFrame | None = None
    remaining = model_metrics.copy()
    while future_forecast is None:
        selected_model, selection_reason = select_best_model(remaining)
        try:
            future_forecast = forecast_future(
                weekly_sales, selected_model, horizon=horizon
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failed_future_fits.append(f"{selected_model}: {failure}")
            remaining.loc[
                remaining["model"].eq(selected_model), "status"
            ] = "future_fit_failed"
            remaining.loc[
                remaining["model"].eq(selected_model), "failure_reasons"
            ] = failure
            if not (remaining["status"] == "complete").any():
                raise ForecastingDataError(
                    "No backtest-complete model could be refit on full history. "
                    + " | ".join(failed_future_fits)
                ) from exc
            selection_reason = (
                "A top-ranked model failed during final refit; selected the next "
                "complete model. " + failure
            )

    model_metrics = remaining.copy()
    model_metrics["selected"] = model_metrics["model"].eq(selected_model)
    model_metrics["selection_reason"] = np.where(
        model_metrics["selected"], selection_reason, ""
    )
    future_forecast["selection_reason"] = selection_reason

    diagnostics.update(
        {
            "initial_training_weeks": int(initial_train),
            "forecast_horizon_weeks": int(horizon),
            "step_between_origins_weeks": int(step),
            "backtest_folds": int(fold_metrics["fold"].nunique()),
            "backtest_origins": [
                date.strftime("%Y-%m-%d")
                for date in sorted(
                    pd.to_datetime(fold_metrics["forecast_start"].unique())
                )
            ],
            "selected_model": selected_model,
            "selection_reason": selection_reason,
            "future_refit_failures": failed_future_fits,
            "negative_future_forecast_count": int(
                future_forecast["forecast"].lt(0).sum()
            ),
            "negative_prediction_policy": (
                "Predictions are not clipped; review the reported count and values "
                "against the revenue data before imposing a constraint."
            ),
            "negative_future_forecasts": [
                {
                    "date": row.date.strftime("%Y-%m-%d"),
                    "forecast": float(row.forecast),
                }
                for row in future_forecast.loc[
                    future_forecast["forecast"].lt(0), ["date", "forecast"]
                ].itertuples(index=False)
            ],
            "full_forecast_dates": [
                date.strftime("%Y-%m-%d") for date in future_forecast["date"]
            ],
        }
    )

    weekly_sales.rename("Sales").to_frame().to_csv(
        output_path / "weekly_sales.csv", index_label="Week Ending"
    )
    predictions.to_csv(output_path / "backtest_predictions.csv", index=False)
    fold_metrics.to_csv(output_path / "fold_metrics.csv", index=False)
    model_metrics.to_csv(output_path / "model_metrics.csv", index=False)
    future_forecast.to_csv(output_path / "future_forecast.csv", index=False)
    (output_path / "weekly_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_charts(
        weekly_sales, predictions, model_metrics, future_forecast, charts_path
    )

    return ForecastingRun(
        weekly_sales=weekly_sales,
        weekly_diagnostics=diagnostics,
        backtest_predictions=predictions,
        fold_metrics=fold_metrics,
        model_metrics=model_metrics,
        future_forecast=future_forecast,
        selected_model=selected_model,
        selection_reason=selection_reason,
        output_directory=output_path,
    )


def _validate_monthly_sales(monthly_sales: pd.Series) -> pd.Series:
    if not isinstance(monthly_sales, pd.Series) or monthly_sales.empty:
        raise ForecastingDataError("Monthly sales must be a non-empty pandas Series.")
    if not isinstance(monthly_sales.index, pd.DatetimeIndex):
        raise ForecastingDataError("Monthly sales must use a DatetimeIndex.")
    if (
        monthly_sales.index.hasnans
        or monthly_sales.index.has_duplicates
        or not monthly_sales.index.is_monotonic_increasing
    ):
        raise ForecastingDataError(
            "Monthly sales dates must be valid, unique, and chronological."
        )
    expected_dates = pd.date_range(
        monthly_sales.index.min(), monthly_sales.index.max(), freq="ME"
    )
    if not monthly_sales.index.equals(expected_dates):
        raise ForecastingDataError(
            "Monthly sales must have consecutive month-end dates; missing months "
            "are not inferred or filled."
        )
    values = pd.to_numeric(monthly_sales, errors="coerce")
    finite_values = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    if values.isna().any() or not finite_values.all() or values.lt(0).any():
        raise ForecastingDataError(
            "Monthly sales must contain finite, non-negative values."
        )
    return values.astype(float).rename(monthly_sales.name or "Sales")


def aggregate_monthly_sales(transactions: pd.DataFrame) -> pd.Series:
    """Aggregate transaction revenue by month without inventing missing months."""
    required = ("Order Date", "Sales")
    missing_columns = [column for column in required if column not in transactions]
    if missing_columns:
        raise ForecastingDataError(
            f"Missing required columns for monthly aggregation: {', '.join(missing_columns)}"
        )
    if transactions.empty:
        raise ForecastingDataError("Cannot aggregate an empty transaction dataset.")

    dates = pd.to_datetime(transactions["Order Date"], errors="coerce")
    sales = pd.to_numeric(transactions["Sales"], errors="coerce")
    finite_sales = np.isfinite(sales.to_numpy(dtype=float, na_value=np.nan))
    invalid_sales = (
        sales.isna()
        | ~pd.Series(finite_sales, index=transactions.index)
        | sales.lt(0)
    )
    if dates.isna().any() or invalid_sales.any():
        raise ForecastingDataError(
            "Monthly aggregation requires valid Order Date and non-negative Sales values."
        )

    monthly = sales.groupby(dates.dt.to_period("M")).sum().sort_index()
    month_ends = monthly.index.to_timestamp(how="end").normalize()
    result = pd.Series(
        monthly.to_numpy(dtype=float),
        index=pd.DatetimeIndex(month_ends, name="Month"),
        name="Sales",
    )
    return _validate_monthly_sales(result)


def monthly_future_dates(
    monthly_sales: pd.Series, horizon: int
) -> pd.DatetimeIndex:
    """Return month-end dates immediately following the observed history."""
    series = _validate_monthly_sales(monthly_sales)
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one month.")
    return pd.date_range(
        series.index[-1], periods=horizon + 1, freq="ME"
    )[1:].rename("Month")


def make_monthly_xgboost_features(
    monthly_sales: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create monthly features from prior sales only, plus known calendar fields."""
    series = _validate_monthly_sales(monthly_sales)
    features = pd.DataFrame(index=series.index)
    for lag in (1, 2, 3):
        features[f"lag_{lag}"] = series.shift(lag)
    features["previous_3_month_mean"] = series.shift(1).rolling(3).mean()
    features["month"] = series.index.month
    features["quarter"] = series.index.quarter
    features = features.dropna()
    return features, series.loc[features.index]


def _monthly_xgboost_feature_row(
    date: pd.Timestamp, history: pd.Series
) -> dict[str, float]:
    return {
        "lag_1": float(history.iloc[-1]),
        "lag_2": float(history.iloc[-2]),
        "lag_3": float(history.iloc[-3]),
        "previous_3_month_mean": float(history.iloc[-3:].mean()),
        "month": float(date.month),
        "quarter": float(date.quarter),
    }


def recursive_monthly_xgboost_forecast(
    training_sales: pd.Series,
    horizon: int,
    estimator_factory: Callable[[], Any] | None = None,
) -> ForecastResult:
    """Forecast multiple months recursively, feeding predictions back as lags."""
    series = _validate_monthly_sales(training_sales)
    if len(series) < 4:
        raise ForecastingDataError(
            "Monthly XGBoost needs at least four observations to build lag features."
        )
    dates = monthly_future_dates(series, horizon)
    features, target = make_monthly_xgboost_features(series)
    if estimator_factory is None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise MissingModelDependencyError(
                "XGBoost is unavailable; install it with `python -m pip install xgboost`."
            ) from exc
        estimator_factory = lambda: XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )

    estimator = estimator_factory()
    estimator.fit(features, target)
    history = series.copy()
    forecasts: list[float] = []
    for date in dates:
        row = pd.DataFrame(
            [_monthly_xgboost_feature_row(date, history)],
            index=pd.DatetimeIndex([date]),
            columns=features.columns,
        )
        prediction = float(np.asarray(estimator.predict(row)).reshape(-1)[0])
        if not np.isfinite(prediction):
            raise ForecastingDataError(
                f"XGBoost produced a non-finite forecast for {date:%Y-%m}."
            )
        forecasts.append(prediction)
        history.loc[date] = prediction
    return ForecastResult(dates=dates, forecast=np.asarray(forecasts, dtype=float))


def _forecast_monthly_sarima(
    training_sales: pd.Series, horizon: int
) -> ForecastResult:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise MissingModelDependencyError(
            "Statsmodels is unavailable; install it with "
            "`python -m pip install statsmodels`."
        ) from exc

    series = _validate_monthly_sales(training_sales)
    dates = monthly_future_dates(series, horizon)
    model_series = pd.Series(
        series.to_numpy(dtype=float),
        index=pd.date_range(series.index[0], periods=len(series), freq="ME"),
        name="Sales",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = SARIMAX(
            model_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        ).fit(disp=False, maxiter=100)
        prediction = fitted.get_forecast(steps=horizon)
    interval = prediction.conf_int(alpha=0.2)
    return ForecastResult(
        dates=dates,
        forecast=np.asarray(prediction.predicted_mean, dtype=float),
        lower=np.asarray(interval.iloc[:, 0], dtype=float),
        upper=np.asarray(interval.iloc[:, 1], dtype=float),
        warnings=tuple(str(item.message) for item in caught),
    )


def _forecast_monthly_prophet(
    training_sales: pd.Series, horizon: int
) -> ForecastResult:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise MissingModelDependencyError(
            "Prophet is unavailable; install it with `python -m pip install prophet`."
        ) from exc

    series = _validate_monthly_sales(training_sales)
    dates = monthly_future_dates(series, horizon)
    training = pd.DataFrame({"ds": series.index, "y": series.to_numpy(dtype=float)})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.8,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(training)
        prediction = model.predict(pd.DataFrame({"ds": dates}))
    return ForecastResult(
        dates=dates,
        forecast=prediction["yhat"].to_numpy(dtype=float),
        lower=prediction["yhat_lower"].to_numpy(dtype=float),
        upper=prediction["yhat_upper"].to_numpy(dtype=float),
        warnings=tuple(str(item.message) for item in caught),
    )


def _forecast_monthly_model(
    model_name: str, training_sales: pd.Series, horizon: int
) -> ForecastResult:
    if model_name == "naive":
        series = _validate_monthly_sales(training_sales)
        return ForecastResult(
            dates=monthly_future_dates(series, horizon),
            forecast=np.repeat(float(series.iloc[-1]), horizon),
        )
    if model_name == "sarima":
        return _forecast_monthly_sarima(training_sales, horizon)
    if model_name == "prophet":
        return _forecast_monthly_prophet(training_sales, horizon)
    if model_name == "xgboost":
        return recursive_monthly_xgboost_forecast(training_sales, horizon)
    raise ValueError(f"Unknown monthly forecasting model: {model_name}")


def run_monthly_forecasting_pipeline(
    data_path: str | Path,
    output_directory: str | Path,
    chart_directory: str | Path,
    horizon: int = 3,
) -> MonthlyForecastRun:
    """Compare the existing models on one chronological monthly holdout."""
    from sales_data import load_sales_data

    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one month.")
    transactions, _ = load_sales_data(data_path)
    monthly_sales = aggregate_monthly_sales(transactions)
    if len(monthly_sales) <= horizon + 3:
        raise ForecastingDataError(
            "Monthly history must exceed the holdout horizon and provide at least "
            "four training observations."
        )
    training_sales = monthly_sales.iloc[:-horizon]
    test_sales = monthly_sales.iloc[-horizon:]
    test_dates = pd.DatetimeIndex(test_sales.index, name="Month")

    model_names = ("naive", "sarima", "prophet", "xgboost")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    successful_results: dict[str, ForecastResult] = {}
    for model_name in model_names:
        try:
            result = _forecast_monthly_model(model_name, training_sales, horizon)
            if not result.dates.equals(test_dates):
                raise ForecastingDataError(
                    f"{model_name} did not return the common chronological test dates."
                )
            metrics = evaluate_forecast_metrics(
                test_sales.to_numpy(dtype=float), result.forecast
            )
            successful_results[model_name] = result
            for position, date in enumerate(test_dates):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "date": date,
                        "actual_sales": float(test_sales.iloc[position]),
                        "forecast_sales": float(result.forecast[position]),
                        "lower_80": (
                            float(result.lower[position])
                            if result.lower is not None
                            else np.nan
                        ),
                        "upper_80": (
                            float(result.upper[position])
                            if result.upper is not None
                            else np.nan
                        ),
                    }
                )
            metric_rows.append(
                {
                    "model": model_name,
                    "status": "complete",
                    **metrics,
                    "warnings": " | ".join(result.warnings),
                    "error": "",
                    "selected": False,
                    "future_refit_error": "",
                }
            )
        except Exception as exc:
            metric_rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "mae": np.nan,
                    "rmse": np.nan,
                    "wape_pct": np.nan,
                    "bias_pct": np.nan,
                    "evaluated_predictions": 0,
                    "warnings": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "selected": False,
                    "future_refit_error": "",
                }
            )

    model_metrics = pd.DataFrame(metric_rows)
    complete = model_metrics.loc[
        model_metrics["status"].eq("complete")
        & np.isfinite(model_metrics["wape_pct"])
    ].sort_values(["wape_pct", "mae", "model"])
    if complete.empty:
        errors = " | ".join(
            f"{row.model}: {row.error}"
            for row in model_metrics.itertuples(index=False)
        )
        raise ForecastingDataError(
            "No model completed the common monthly holdout. " + errors
        )

    output_path = Path(output_directory)
    charts_path = Path(chart_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    charts_path.mkdir(parents=True, exist_ok=True)
    future_fit_errors: dict[str, str] = {}
    selected_model = ""
    future_result: ForecastResult | None = None
    for candidate in complete["model"]:
        try:
            future_result = _forecast_monthly_model(
                candidate, monthly_sales, horizon
            )
            selected_model = candidate
            break
        except Exception as exc:
            future_fit_errors[candidate] = f"{type(exc).__name__}: {exc}"
            model_metrics.loc[
                model_metrics["model"].eq(candidate), "future_refit_error"
            ] = future_fit_errors[candidate]
    if future_result is None:
        raise ForecastingDataError(
            "No model could be refit for the future forecast. "
            + " | ".join(
                f"{model}: {error}" for model, error in future_fit_errors.items()
            )
        )

    model_metrics["selected"] = model_metrics["model"].eq(selected_model)
    test_predictions = pd.DataFrame(prediction_rows)
    future_forecast = pd.DataFrame(
        {
            "date": future_result.dates,
            "model": selected_model,
            "forecast_sales": future_result.forecast,
            "lower_80": future_result.lower,
            "upper_80": future_result.upper,
            "forecast_type": "future",
        }
    )
    model_metrics.to_csv(output_path / "monthly_model_metrics.csv", index=False)
    test_predictions.to_csv(
        output_path / "monthly_test_predictions.csv", index=False
    )
    future_forecast.to_csv(
        output_path / "monthly_future_forecast.csv", index=False
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    plt.plot(monthly_sales.index, monthly_sales, color="#526579", label="Historical sales")
    plt.axvspan(test_dates[0], test_dates[-1], color="#e0a84f", alpha=0.15)
    for model_name in successful_results:
        model_predictions = test_predictions.loc[
            test_predictions["model"].eq(model_name)
        ]
        plt.plot(
            model_predictions["date"],
            model_predictions["forecast_sales"],
            marker="o",
            label=f"{model_name} test prediction",
        )
    plt.title("Monthly sales revenue: historical holdout predictions")
    plt.xlabel("Month")
    plt.ylabel("Sales revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_path / "monthly_test_forecasts.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(
        monthly_sales.index[-12:],
        monthly_sales.iloc[-12:],
        color="#526579",
        marker="o",
        label="Historical monthly sales",
    )
    plt.plot(
        future_forecast["date"],
        future_forecast["forecast_sales"],
        color="#c47b36",
        marker="o",
        label=f"{selected_model} future forecast",
    )
    if future_result.lower is not None and future_result.upper is not None:
        plt.fill_between(
            future_forecast["date"],
            future_result.lower,
            future_result.upper,
            color="#c47b36",
            alpha=0.2,
            label="80% model interval",
        )
    plt.axvline(monthly_sales.index[-1], color="grey", linestyle="--")
    plt.title("Monthly sales revenue: future forecast beyond observed history")
    plt.xlabel("Month")
    plt.ylabel("Sales revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_path / "monthly_future_forecast.png", dpi=150)
    plt.close()

    return MonthlyForecastRun(
        monthly_sales=monthly_sales,
        test_predictions=test_predictions,
        model_metrics=model_metrics,
        future_forecast=future_forecast,
        selected_model=selected_model,
        output_directory=output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate monthly sales-revenue models on a chronological holdout."
    )
    parser.add_argument("--data", default="Data/train.csv")
    parser.add_argument("--output", default="Forecasting_Output")
    parser.add_argument("--charts", default="Charts")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    run = run_monthly_forecasting_pipeline(
        data_path=args.data,
        output_directory=args.output,
        chart_directory=args.charts,
        horizon=args.horizon,
    )
    print(f"Selected model: {run.selected_model}")
    print(run.model_metrics.to_string(index=False))
    print(
        "Future forecast saved to "
        f"{run.output_directory / 'monthly_future_forecast.csv'}"
    )


if __name__ == "__main__":
    main()
