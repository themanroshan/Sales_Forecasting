"""Loading, validating, and summarizing the retail sales dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


DATE_FORMAT = "%d-%m-%Y"
REQUIRED_COLUMNS = (
    "Order ID",
    "Order Date",
    "Ship Date",
    "Customer ID",
    "Product ID",
    "Sales",
    "Category",
    "Region",
    "Sub-Category",
)
IDENTIFIER_COLUMNS = ("Order ID", "Customer ID", "Product ID")


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    columns: int
    date_range: str
    data_types: dict[str, str]
    distinct_orders: int
    distinct_customers: int
    missing_values: dict[str, int]
    duplicate_rows: int
    invalid_records: dict[str, int]
    invalid_record_count: int
    cleaning_decisions: tuple[str, ...]


def _valid_identifier(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def _distinct_identifier_count(df: pd.DataFrame, column: str) -> int:
    values = df.loc[_valid_identifier(df[column]), column].astype("string").str.strip()
    return int(values.nunique())


def _parse_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")


def build_data_quality_report(df: pd.DataFrame) -> DataQualityReport:
    """Summarize schema, missing data, and invalid records without dropping rows."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    order_dates = _parse_date(df["Order Date"])
    ship_dates = _parse_date(df["Ship Date"])
    sales = pd.to_numeric(df["Sales"], errors="coerce")
    finite_sales = pd.Series(
        np.isfinite(sales.to_numpy(dtype=float, na_value=np.nan)), index=df.index
    )

    invalid_masks = {
        "invalid_order_dates": order_dates.isna(),
        "invalid_ship_dates": ship_dates.isna(),
        "invalid_sales": sales.isna() | ~finite_sales | sales.lt(0),
        "missing_order_ids": ~_valid_identifier(df["Order ID"]),
        "missing_customer_ids": ~_valid_identifier(df["Customer ID"]),
        "missing_product_ids": ~_valid_identifier(df["Product ID"]),
        "ship_before_order": ship_dates.lt(order_dates).fillna(False),
    }

    identifier_valid = _valid_identifier(df["Order ID"])
    order_dates_per_order = order_dates[identifier_valid].groupby(
        df.loc[identifier_valid, "Order ID"]
    ).nunique()
    ship_dates_per_order = ship_dates[identifier_valid].groupby(
        df.loc[identifier_valid, "Order ID"]
    ).nunique()
    inconsistent_order_ids = set(
        order_dates_per_order[order_dates_per_order > 1].index
    ) | set(ship_dates_per_order[ship_dates_per_order > 1].index)
    if inconsistent_order_ids:
        invalid_masks["inconsistent_order_dates"] = df["Order ID"].isin(
            inconsistent_order_ids
        )

    invalid_counts = {
        name: int(mask.sum()) for name, mask in invalid_masks.items() if mask.any()
    }
    invalid_union = pd.Series(False, index=df.index)
    for mask in invalid_masks.values():
        invalid_union |= mask

    valid_order_dates = order_dates.dropna()
    if valid_order_dates.empty:
        date_range = "Unavailable"
    else:
        date_range = (
            f"{valid_order_dates.min():%Y-%m-%d} to "
            f"{valid_order_dates.max():%Y-%m-%d}"
        )

    missing_values = {
        column: int(count)
        for column, count in df.isna().sum().items()
    }
    return DataQualityReport(
        rows=len(df),
        columns=len(df.columns),
        date_range=date_range,
        data_types={column: str(dtype) for column, dtype in df.dtypes.items()},
        distinct_orders=_distinct_identifier_count(df, "Order ID"),
        distinct_customers=_distinct_identifier_count(df, "Customer ID"),
        missing_values=missing_values,
        duplicate_rows=int(df.duplicated().sum()),
        invalid_records=invalid_counts,
        invalid_record_count=int(invalid_union.sum()),
        cleaning_decisions=(
            "Dates are parsed explicitly as day-month-year.",
            "Rows are not dropped automatically; invalid records are reported.",
            "Missing Postal Code values are retained because postal codes are not "
            "required for the current sales analysis.",
            "No quantity, profit, discount, or inventory fields are inferred.",
        ),
    )


def load_sales_data(path: str | Path) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load CSV data, parse known fields, and return a non-destructive quality report."""
    df = pd.read_csv(Path(path))
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    df["Order Date"] = _parse_date(df["Order Date"])
    df["Ship Date"] = _parse_date(df["Ship Date"])
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    for column in IDENTIFIER_COLUMNS:
        df[column] = df[column].astype("string")
    return df, build_data_quality_report(df)


def calculate_total_sales(df: pd.DataFrame) -> float:
    return float(pd.to_numeric(df["Sales"], errors="coerce").sum())


def calculate_distinct_orders(df: pd.DataFrame) -> int:
    return _distinct_identifier_count(df, "Order ID")


def calculate_distinct_customers(df: pd.DataFrame) -> int:
    return _distinct_identifier_count(df, "Customer ID")


def calculate_average_order_value(df: pd.DataFrame) -> float:
    order_count = calculate_distinct_orders(df)
    return calculate_total_sales(df) / order_count if order_count else 0.0


def calculate_distinct_products(df: pd.DataFrame) -> int:
    return _distinct_identifier_count(df, "Product ID")


def calculate_order_to_ship_days(df: pd.DataFrame) -> pd.Series:
    """Return one valid order-to-ship duration per distinct order."""
    valid_rows = df.loc[
        _valid_identifier(df["Order ID"])
        & df["Order Date"].notna()
        & df["Ship Date"].notna()
        & df["Ship Date"].ge(df["Order Date"])
    ]
    if valid_rows.empty:
        return pd.Series(
            dtype=float,
            index=pd.Index([], name="Order ID"),
            name="Order-to-Ship Days",
        )

    order_groups = valid_rows.groupby("Order ID", sort=False).agg(
        order_dates=("Order Date", "nunique"),
        ship_dates=("Ship Date", "nunique"),
    )
    consistent_orders = order_groups.loc[
        order_groups["order_dates"].eq(1) & order_groups["ship_dates"].eq(1)
    ].index
    order_rows = valid_rows.loc[valid_rows["Order ID"].isin(consistent_orders)]
    order_rows = order_rows.drop_duplicates(subset=["Order ID"])
    durations = (order_rows["Ship Date"] - order_rows["Order Date"]).dt.days
    return pd.Series(
        durations.to_numpy(dtype=float),
        index=pd.Index(order_rows["Order ID"].astype("string"), name="Order ID"),
        name="Order-to-Ship Days",
    )


def calculate_average_order_to_ship_days(df: pd.DataFrame) -> float:
    """Average one shipping duration per order, ignoring inconsistent date groups."""
    durations = calculate_order_to_ship_days(df)
    return float(durations.mean()) if not durations.empty else float("nan")


def calculate_weekly_sales_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return flagged Sunday-ending weeks from the existing two anomaly methods."""
    required = ("Order Date", "Sales")
    missing_columns = [column for column in required if column not in df]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    dates = _parse_date(df["Order Date"])
    sales = pd.to_numeric(df["Sales"], errors="coerce")
    finite_sales = np.isfinite(sales.to_numpy(dtype=float, na_value=np.nan))
    invalid_sales = sales.isna() | ~pd.Series(finite_sales, index=df.index) | sales.lt(0)
    if dates.isna().any() or invalid_sales.any():
        raise ValueError("Anomaly detection requires valid dates and non-negative sales.")

    weekly_sales = sales.groupby(dates.dt.to_period("W-SUN").dt.end_time.dt.normalize()).sum()
    weekly_sales = weekly_sales.sort_index().astype(float)
    first_date = dates.min().normalize()
    last_date = dates.max().normalize()
    partial_weeks: set[pd.Timestamp] = set()
    first_period = first_date.to_period("W-SUN")
    last_period = last_date.to_period("W-SUN")
    if first_date > first_period.start_time.normalize():
        partial_weeks.add(first_period.end_time.normalize())
    if last_date < last_period.end_time.normalize():
        partial_weeks.add(last_period.end_time.normalize())
    if partial_weeks:
        weekly_sales = weekly_sales.loc[~weekly_sales.index.isin(partial_weeks)]
    if len(weekly_sales) < 20:
        raise ValueError("At least 20 weekly observations are required for Isolation Forest.")

    isolation_flags = IsolationForest(
        contamination=0.05,
        random_state=42,
    ).fit_predict(weekly_sales.to_numpy().reshape(-1, 1)) == -1
    standard_deviation = float(weekly_sales.std())
    z_scores = (
        (weekly_sales - weekly_sales.mean()) / standard_deviation
        if standard_deviation
        else pd.Series(0.0, index=weekly_sales.index)
    )
    z_score_flags = z_scores.abs().gt(2)
    anomaly_mask = isolation_flags | z_score_flags.to_numpy()

    report = pd.DataFrame(
        {
            "week_ending": weekly_sales.index,
            "sales": weekly_sales.to_numpy(),
            "isolation_forest": isolation_flags,
            "z_score": z_score_flags.to_numpy(),
            "z_score_value": z_scores.to_numpy(),
        }
    )
    report = report.loc[anomaly_mask].copy()
    report["direction"] = np.where(
        report["sales"] > weekly_sales.mean(), "High", "Low"
    )
    report["detection_methods"] = report.apply(
        lambda row: ", ".join(
            name
            for name, flagged in (
                ("Isolation Forest", row["isolation_forest"]),
                ("Z-Score", row["z_score"]),
            )
            if flagged
        ),
        axis=1,
    )
    return report.reset_index(drop=True)


def calculate_abc_product_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Rank products by revenue and assign conventional cumulative-share ABC bands."""
    required = ("Product ID", "Product Name", "Sales")
    missing_columns = [column for column in required if column not in df]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    sales = pd.to_numeric(df["Sales"], errors="coerce")
    finite_sales = np.isfinite(sales.to_numpy(dtype=float, na_value=np.nan))
    if sales.isna().any() or not finite_sales.all() or sales.lt(0).any():
        raise ValueError("Product segmentation requires finite, non-negative sales.")
    if not _valid_identifier(df["Product ID"]).all():
        raise ValueError("Product segmentation requires a valid Product ID on every row.")

    products = pd.DataFrame(
        {
            "product_id": df["Product ID"].astype("string").str.strip(),
            "product_name": df["Product Name"],
            "revenue": sales,
        }
    )
    products = (
        products.groupby("product_id", as_index=False, sort=False)
        .agg(product_name=("product_name", "first"), revenue=("revenue", "sum"))
        .sort_values(["revenue", "product_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_revenue = float(products["revenue"].sum())
    if total_revenue <= 0:
        raise ValueError("ABC segmentation requires positive total revenue.")

    products.insert(0, "rank", np.arange(1, len(products) + 1))
    products["revenue_contribution_pct"] = products["revenue"] / total_revenue * 100
    products["cumulative_contribution_pct"] = (
        products["revenue_contribution_pct"].cumsum()
    )
    contribution_before_product = (
        products["cumulative_contribution_pct"]
        - products["revenue_contribution_pct"]
    )
    products["segment"] = np.select(
        [contribution_before_product < 80, contribution_before_product < 95],
        ["A", "B"],
        default="C",
    )
    return products
