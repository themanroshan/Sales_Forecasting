import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sales_data import (
    calculate_average_order_to_ship_days,
    calculate_average_order_value,
    calculate_abc_product_segmentation,
    calculate_distinct_customers,
    calculate_distinct_orders,
    calculate_distinct_products,
    calculate_order_to_ship_days,
    calculate_total_sales,
    calculate_weekly_sales_anomalies,
    load_sales_data,
)


class SalesDataTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "Order ID": ["ORD-1", "ORD-1", "ORD-1", "ORD-2"],
                "Order Date": [
                    "01-02-2020",
                    "01-02-2020",
                    "01-02-2020",
                    "10-02-2020",
                ],
                "Ship Date": [
                    "03-02-2020",
                    "03-02-2020",
                    "03-02-2020",
                    "15-02-2020",
                ],
                "Customer ID": ["C-1", "C-1", "C-1", "C-2"],
                "Product ID": ["P-1", "P-2", "P-1", "P-2"],
                "Sales": [10.0, 20.0, 30.0, 40.0],
                "Category": ["Office Supplies"] * 4,
                "Region": ["West"] * 4,
                "Sub-Category": ["Paper"] * 4,
                "Postal Code": [None, 10001, 10001, 10002],
            }
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.csv_path = Path(self.temp_dir.name) / "train.csv"
        self.data.to_csv(self.csv_path, index=False)
        self.df, self.report = load_sales_data(self.csv_path)

    def test_loads_data_and_explicitly_parses_day_month_year(self):
        self.assertEqual(len(self.df), 4)
        self.assertEqual(self.df.loc[0, "Order Date"], pd.Timestamp("2020-02-01"))
        self.assertEqual(self.report.date_range, "2020-02-01 to 2020-02-10")

    def test_distinct_order_and_customer_counts(self):
        self.assertEqual(calculate_distinct_orders(self.df), 2)
        self.assertEqual(calculate_distinct_customers(self.df), 2)

    def test_average_order_value_uses_distinct_orders(self):
        self.assertEqual(calculate_total_sales(self.df), 100.0)
        self.assertEqual(calculate_average_order_value(self.df), 50.0)

    def test_counts_distinct_products(self):
        self.assertEqual(calculate_distinct_products(self.df), 2)

    def test_shipping_duration_counts_each_order_once(self):
        self.assertEqual(calculate_average_order_to_ship_days(self.df), 3.5)
        durations = calculate_order_to_ship_days(self.df)
        self.assertEqual(len(durations), 2)
        self.assertEqual(durations.index.nunique(), 2)

    def test_missing_postal_code_rows_are_retained(self):
        self.assertEqual(len(self.df), 4)
        self.assertEqual(self.df["Postal Code"].isna().sum(), 1)
        self.assertEqual(self.report.missing_values["Postal Code"], 1)

    def test_abc_segmentation_uses_product_revenue_and_cumulative_boundaries(self):
        products = pd.DataFrame(
            {
                "Product ID": ["P-1", "P-1", "P-2", "P-3", "P-4"],
                "Product Name": ["One", "One - alternate listing", "Two", "Three", "Four"],
                "Sales": [30.0, 20.0, 30.0, 15.0, 5.0],
            }
        )

        result = calculate_abc_product_segmentation(products)

        self.assertEqual(result["product_id"].tolist(), ["P-1", "P-2", "P-3", "P-4"])
        self.assertEqual(result["product_id"].nunique(), len(result))
        self.assertEqual(result["rank"].tolist(), [1, 2, 3, 4])
        self.assertEqual(result["segment"].tolist(), ["A", "A", "B", "C"])
        self.assertAlmostEqual(result.iloc[-1]["cumulative_contribution_pct"], 100.0)

    def test_dataset_anomaly_report_records_overlap_dates_and_direction(self):
        dataset_path = Path(__file__).resolve().parents[1] / "Data" / "train.csv"
        dataset, _ = load_sales_data(dataset_path)

        anomalies = calculate_weekly_sales_anomalies(dataset)

        isolation_count = int(anomalies["isolation_forest"].sum())
        z_score_count = int(anomalies["z_score"].sum())
        overlap_count = int(
            (anomalies["isolation_forest"] & anomalies["z_score"]).sum()
        )
        self.assertEqual((isolation_count, z_score_count, overlap_count), (11, 11, 6))
        self.assertEqual(len(anomalies), 16)
        self.assertFalse(
            anomalies["week_ending"].eq(pd.Timestamp("2015-01-04")).any()
        )
        self.assertIn("High", anomalies["direction"].tolist())
        self.assertIn("Low", anomalies["direction"].tolist())
        self.assertEqual(
            anomalies.loc[anomalies["week_ending"].eq(pd.Timestamp("2015-03-22")),
                          "detection_methods"].iloc[0],
            "Isolation Forest, Z-Score",
        )


if __name__ == "__main__":
    unittest.main()
