import unittest

import numpy as np
import pandas as pd

from sales_forecasting import (
    aggregate_monthly_sales,
    aggregate_weekly_sales,
    compare_models,
    evaluate_forecast_metrics,
    forecast_future,
    future_week_dates,
    make_xgboost_features,
    make_monthly_xgboost_features,
    missing_week_dates,
    moving_average_forecast,
    naive_forecast,
    recursive_xgboost_forecast,
    recursive_monthly_xgboost_forecast,
    rolling_origin_backtest,
    seasonal_naive_forecast,
    select_best_model,
    validate_weekly_sales,
    monthly_future_dates,
)


def weekly_values(count=64):
    index = pd.date_range("2020-01-05", periods=count, freq="W-SUN")
    values = np.arange(100.0, 100.0 + count)
    return pd.Series(values, index=index, name="Sales")


def monthly_values(count=24):
    index = pd.date_range("2020-01-31", periods=count, freq="ME")
    values = np.arange(100.0, 100.0 + count)
    return pd.Series(values, index=index, name="Sales")


class SalesForecastingTests(unittest.TestCase):
    def test_weekly_sales_aggregation_uses_sunday_week_endings(self):
        transactions = pd.DataFrame(
            {
                "Order Date": pd.to_datetime(
                    ["2020-01-06", "2020-01-12", "2020-01-13", "2020-01-19"]
                ),
                "Sales": [10.0, 20.0, 30.0, 40.0],
            }
        )
        weekly = aggregate_weekly_sales(transactions)
        self.assertEqual(weekly.index.tolist(), list(pd.to_datetime(["2020-01-12", "2020-01-19"])))
        self.assertEqual(weekly.tolist(), [30.0, 70.0])

    def test_weekly_date_index_is_contiguous_and_sunday_ending(self):
        weekly = validate_weekly_sales(weekly_values(), require_contiguous=True)
        self.assertTrue((weekly.index.dayofweek == 6).all())
        self.assertEqual(len(missing_week_dates(weekly)), 0)

    def test_missing_weeks_remain_missing_and_are_reported(self):
        transactions = pd.DataFrame(
            {
                "Order Date": pd.to_datetime(
                    ["2020-01-06", "2020-01-12", "2020-01-20", "2020-01-26"]
                ),
                "Sales": [10.0, 20.0, 30.0, 40.0],
            }
        )
        weekly = aggregate_weekly_sales(transactions)
        self.assertEqual(len(weekly), 2)
        self.assertEqual(
            missing_week_dates(weekly).tolist(),
            list(pd.to_datetime(["2020-01-19"])),
        )
        with self.assertRaisesRegex(ValueError, "missing weeks"):
            validate_weekly_sales(weekly, require_contiguous=True)

    def test_partial_boundary_week_is_excluded_and_reported(self):
        transactions = pd.DataFrame(
            {
                "Order Date": pd.to_datetime(["2015-01-03", "2015-01-11"]),
                "Sales": [100.0, 200.0],
            }
        )
        weekly = aggregate_weekly_sales(transactions)
        self.assertEqual(weekly.index.tolist(), [pd.Timestamp("2015-01-11")])
        self.assertEqual(
            weekly.attrs["excluded_partial_boundary_weeks"], ["2015-01-04"]
        )

    def test_naive_forecast_repeats_last_observation(self):
        forecast = naive_forecast(weekly_values(), horizon=4)
        self.assertEqual(forecast.tolist(), [163.0] * 4)

    def test_seasonal_naive_forecast_uses_values_from_52_weeks_ago(self):
        series = weekly_values()
        forecast = seasonal_naive_forecast(series, horizon=4)
        self.assertEqual(forecast.tolist(), series.iloc[-52:-48].tolist())

    def test_moving_average_baseline_uses_trailing_four_weeks(self):
        series = weekly_values()
        forecast = moving_average_forecast(series, horizon=4, window=4)
        self.assertEqual(forecast.tolist(), [161.5] * 4)

    def test_forecast_horizon_and_future_dates_are_exact(self):
        series = weekly_values()
        dates = future_week_dates(series, horizon=4)
        self.assertEqual(len(naive_forecast(series, horizon=4)), 4)
        self.assertEqual(dates.tolist(), list(pd.date_range("2021-03-28", periods=4, freq="W-SUN")))

    def test_lag_features_use_only_prior_sales(self):
        series = weekly_values()
        features, target = make_xgboost_features(series)
        date = series.index[55]
        self.assertEqual(features.loc[date, "lag_1"], series.loc[series.index[54]])
        self.assertEqual(target.loc[date], series.loc[date])

    def test_rolling_features_are_shifted_before_rolling(self):
        series = weekly_values()
        features, _ = make_xgboost_features(series)
        date = series.index[55]
        expected = series.loc[series.index[51] : series.index[54]].mean()
        self.assertEqual(features.loc[date, "previous_4_week_mean"], expected)

    def test_future_target_change_does_not_change_features_available_before_it(self):
        original = weekly_values()
        changed = original.copy()
        changed_date = original.index[55]
        changed.loc[changed_date] += 100000
        original_features, _ = make_xgboost_features(original)
        changed_features, _ = make_xgboost_features(changed)
        pd.testing.assert_frame_equal(
            original_features.loc[:changed_date],
            changed_features.loc[:changed_date],
        )

    def test_recursive_forecast_feeds_previous_predictions_into_next_step(self):
        class LagPlusOneEstimator:
            def fit(self, features, target):
                self.training_target = target.copy()
                self.seen_lags = []
                return self

            def predict(self, features):
                lag_value = float(features.iloc[0]["lag_1"])
                self.seen_lags.append(lag_value)
                return np.array([lag_value + 1])

        estimator = LagPlusOneEstimator()
        result = recursive_xgboost_forecast(
            weekly_values(60),
            horizon=4,
            estimator_factory=lambda: estimator,
        )
        self.assertEqual(
            result.forecast.tolist(),
            [160.0, 161.0, 162.0, 163.0],
        )
        self.assertEqual(estimator.seen_lags, [159.0, 160.0, 161.0, 162.0])

    def test_recursive_forecast_fits_only_historical_targets(self):
        class TargetRecorder:
            def fit(self, features, target):
                self.feature_dates = features.index.copy()
                self.target_dates = target.index.copy()
                self.target_values = target.copy()
                return self

            def predict(self, features):
                return np.array([float(features.iloc[0]["lag_1"])])

        history = weekly_values(60)
        future_actuals = pd.Series(
            [999999.0, 888888.0, 777777.0, 666666.0],
            index=future_week_dates(history, 4),
        )
        estimator = TargetRecorder()
        result = recursive_xgboost_forecast(
            history,
            horizon=4,
            estimator_factory=lambda: estimator,
        )
        self.assertTrue(estimator.target_dates.max() <= history.index[-1])
        self.assertFalse(estimator.target_values.isin(future_actuals.values).any())
        self.assertTrue((result.dates > history.index[-1]).all())

    def test_backtesting_is_chronological_and_models_share_origins(self):
        predictions, folds = rolling_origin_backtest(
            weekly_values(),
            initial_train=52,
            horizon=4,
            step=4,
            models=("naive", "moving_average_4", "seasonal_naive"),
        )
        self.assertEqual(folds["fold"].nunique(), 3)
        self.assertTrue((folds["train_end"] < folds["forecast_start"]).all())
        self.assertTrue(
            predictions.groupby("fold")["forecast_origin"].nunique().eq(1).all()
        )
        self.assertTrue(
            predictions.groupby(["model", "fold"]).size().eq(4).all()
        )

    def test_error_metric_calculations(self):
        metrics = evaluate_forecast_metrics([100, 100], [110, 90])
        self.assertEqual(metrics["mae"], 10.0)
        self.assertEqual(metrics["rmse"], 10.0)
        self.assertEqual(metrics["wape_pct"], 10.0)
        self.assertEqual(metrics["bias_pct"], 0.0)

    def test_wape_is_undefined_when_all_actuals_are_zero(self):
        metrics = evaluate_forecast_metrics([0, 0], [1, -1])
        self.assertTrue(np.isnan(metrics["wape_pct"]))

    def test_bias_percentage_sign_indicates_over_or_underforecast(self):
        overforecast = evaluate_forecast_metrics([100, 100], [110, 110])
        underforecast = evaluate_forecast_metrics([100, 100], [90, 90])
        self.assertGreater(overforecast["bias_pct"], 0)
        self.assertLess(underforecast["bias_pct"], 0)

    def test_model_comparison_uses_pooled_metrics_and_simpler_tie_break(self):
        rows = []
        fold_rows = []
        actuals = [100.0, 200.0]
        forecasts = {
            "naive": [108.0, 192.0],
            "seasonal_naive": [108.5, 191.5],
            "xgboost": [120.0, 180.0],
        }
        for model, values in forecasts.items():
            for position, (actual, forecast) in enumerate(zip(actuals, values)):
                rows.append(
                    {
                        "model": model,
                        "fold": 1,
                        "target_date": pd.Timestamp("2024-01-07")
                        + pd.Timedelta(weeks=position),
                        "actual": actual,
                        "forecast": forecast,
                        "status": "ok",
                    }
                )
            fold_rows.append({"model": model, "fold": 1, "status": "ok"})
        metrics = compare_models(pd.DataFrame(rows), pd.DataFrame(fold_rows))
        selected, _ = select_best_model(metrics, wape_tolerance_pct_points=1.0)
        self.assertAlmostEqual(
            metrics.set_index("model").loc["naive", "wape_pct"],
            (16 / 300) * 100,
        )
        self.assertEqual(metrics.set_index("model").loc["xgboost", "status"], "complete")
        self.assertEqual(selected, "naive")

    def test_final_future_forecast_dates_follow_last_history_week(self):
        series = weekly_values()
        forecast = forecast_future(series, "naive", horizon=4)
        self.assertEqual(len(forecast), 4)
        self.assertTrue((forecast["date"] > series.index[-1]).all())
        self.assertEqual(
            forecast["date"].tolist(),
            list(pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=4, freq="W-SUN")),
        )

    def test_monthly_aggregation_uses_revenue_and_month_end_dates(self):
        transactions = pd.DataFrame(
            {
                "Order Date": pd.to_datetime(
                    ["2020-01-02", "2020-01-20", "2020-02-01", "2020-02-28"]
                ),
                "Sales": [10.0, 20.0, 30.0, 40.0],
            }
        )

        monthly = aggregate_monthly_sales(transactions)

        self.assertEqual(
            monthly.index.tolist(),
            list(pd.to_datetime(["2020-01-31", "2020-02-29"])),
        )
        self.assertEqual(monthly.tolist(), [30.0, 70.0])

    def test_monthly_features_are_based_only_on_prior_sales(self):
        series = monthly_values()
        original_features, _ = make_monthly_xgboost_features(series)
        changed = series.copy()
        changed_date = series.index[10]
        changed.loc[changed_date] += 100000
        changed_features, _ = make_monthly_xgboost_features(changed)

        pd.testing.assert_frame_equal(
            original_features.loc[:changed_date],
            changed_features.loc[:changed_date],
        )
        self.assertEqual(
            original_features.loc[changed_date, "previous_3_month_mean"],
            series.iloc[7:10].mean(),
        )

    def test_monthly_recursive_forecast_uses_only_training_and_prior_predictions(self):
        class LagPlusOneEstimator:
            def fit(self, features, target):
                self.target_dates = target.index.copy()
                self.target_values = target.copy()
                self.seen_lags = []
                return self

            def predict(self, features):
                lag_value = float(features.iloc[0]["lag_1"])
                self.seen_lags.append(lag_value)
                return np.array([lag_value + 1])

        training = monthly_values(12)
        future_actuals = [999999.0, 888888.0, 777777.0]
        estimator = LagPlusOneEstimator()
        result = recursive_monthly_xgboost_forecast(
            training,
            horizon=3,
            estimator_factory=lambda: estimator,
        )

        self.assertEqual(result.forecast.tolist(), [112.0, 113.0, 114.0])
        self.assertEqual(estimator.seen_lags, [111.0, 112.0, 113.0])
        self.assertLessEqual(estimator.target_dates.max(), training.index[-1])
        self.assertFalse(estimator.target_values.isin(future_actuals).any())
        self.assertTrue((result.dates > training.index[-1]).all())

    def test_monthly_future_dates_follow_last_observed_month(self):
        series = monthly_values()

        dates = monthly_future_dates(series, horizon=3)

        self.assertEqual(
            dates.tolist(),
            list(pd.date_range("2022-01-31", periods=3, freq="ME")),
        )
        self.assertTrue((dates > series.index[-1]).all())


if __name__ == "__main__":
    unittest.main()
