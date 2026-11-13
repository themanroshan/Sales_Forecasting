import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sales_data import (
    calculate_average_order_to_ship_days,
    calculate_average_order_value,
    calculate_abc_product_segmentation,
    calculate_distinct_customers,
    calculate_distinct_orders,
    calculate_distinct_products,
    calculate_total_sales,
    calculate_weekly_sales_anomalies,
    load_sales_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CHARTS_DIR = PROJECT_ROOT / "Charts"
FORECAST_OUTPUT_DIR = PROJECT_ROOT / "Forecasting_Output"

st.set_page_config(page_title="Sales Forecasting System", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0b1220;
    color: #e5edf7;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101c30 0%, #0d1626 100%);
    border-right: 1px solid #223249;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 8px;
    padding: 0.3rem 0.5rem;
    color: #c4d2e3;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #182b43;
}

[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #67c8f2;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.75rem;
    padding-bottom: 3rem;
}

[data-testid="stMetric"] {
    background: #111c2f;
    border: 1px solid #25364e;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
    padding: 12px 14px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #9aacc2 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #f0f6ff !important;
    font-size: clamp(1.1rem, 2vw, 1.5rem);
    white-space: nowrap;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #b7c6d9 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #25364e;
    border-radius: 12px;
    overflow: hidden;
}

h1, h2, h3, h4, h5, h6 {
    color: #e7f0fb;
}

[data-testid="stHeading"] h1 {
    font-size: clamp(1.75rem, 2.5vw, 2.15rem);
    font-weight: 600;
}

.hero-card {
    padding: 1rem 1.15rem;
    border-radius: 12px;
    background: #111c2f;
    border: 1px solid #25364e;
    margin-bottom: 0.5rem;
}

.project-header {
    margin-bottom: 1.25rem;
    padding: 1.25rem 1.4rem;
    border: 1px solid #25435b;
    border-left: 4px solid #67c8f2;
    border-radius: 14px;
    background: linear-gradient(110deg, #11263a 0%, #101a2b 68%, #111827 100%);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
}

.project-header h1 {
    margin: 0;
    color: #f0f7ff;
    font-size: clamp(1.45rem, 2.5vw, 2rem);
    font-weight: 600;
    line-height: 1.3;
}

div[data-testid="stPlotlyChart"] {
    overflow: hidden;
    border: 1px solid #25364e;
    border-radius: 14px;
    background: #111c2f;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #111c2f;
    border-color: #30425a;
}

[data-baseweb="select"] *,
[data-baseweb="input"] input {
    color: #e5edf7;
}

[data-testid="stCaption"],
[data-testid="stMarkdownContainer"] p {
    color: #a7b6c9;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1.25rem 1rem 2rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df, data_quality = load_sales_data(PROJECT_ROOT / "Data" / "train.csv")
df["Year"] = df["Order Date"].dt.year

with st.sidebar:
    st.caption("SALES FORECASTING")
    selected = st.radio(
        "Navigation",
        [
            "Sales Overview",
            "Forecast Explorer",
            "Anomaly Report",
            "Product Segments",
            "Executive Summary",
        ],
        label_visibility="collapsed",
    )

st.markdown(
    """
    <div class="project-header">
        <h1>📊 End-to-End Sales Forecasting &amp; Demand Intelligence System</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Data quality report"):
    report_col1, report_col2, report_col3 = st.columns(3)
    report_col1.metric("Rows / columns", f"{data_quality.rows:,} / {data_quality.columns}")
    report_col2.metric("Order date range", data_quality.date_range)
    report_col3.metric(
        "Distinct orders / customers",
        f"{data_quality.distinct_orders:,} / {data_quality.distinct_customers:,}",
    )
    st.write(f"Duplicate rows: {data_quality.duplicate_rows:,}")
    st.write(f"Invalid records: {data_quality.invalid_record_count:,}")
    st.write("Data types:", data_quality.data_types)
    missing_values = {
        column: count
        for column, count in data_quality.missing_values.items()
        if count
    }
    st.write("Missing values:", missing_values or "None")
    st.write("Invalid record counts:", data_quality.invalid_records or "None")
    st.markdown("**Cleaning decisions**")
    for decision in data_quality.cleaning_decisions:
        st.write(f"- {decision}")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if selected == "Sales Overview":

    st.title("Sales overview")
    st.caption("Explore sales trends and compare performance across the business.")

    filter_category, filter_region, filter_breakdown = st.columns(3)

    with filter_category:
        selected_category = st.selectbox(
            "Category", ["All categories", *sorted(df["Category"].unique())]
        )

    with filter_region:
        selected_region = st.selectbox(
            "Region", ["All regions", *sorted(df["Region"].unique())]
        )

    with filter_breakdown:
        breakdown = st.selectbox(
            "Compare by", ["Category", "Region", "Sub-Category"]
        )

    filtered_df = df.copy()
    if selected_category != "All categories":
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]
    if selected_region != "All regions":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${calculate_total_sales(filtered_df):,.0f}")
    col2.metric("Distinct Orders", f"{calculate_distinct_orders(filtered_df):,}")
    col3.metric("Distinct Customers", f"{calculate_distinct_customers(filtered_df):,}")
    col4.metric("Average Order Value", f"${calculate_average_order_value(filtered_df):,.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Distinct Products", f"{calculate_distinct_products(filtered_df):,}")
    col6.metric(
        "Average Order-to-Ship Days",
        f"{calculate_average_order_to_ship_days(filtered_df):.1f}",
    )
    col7.metric("Categories", filtered_df["Category"].nunique())
    col8.metric("Regions", filtered_df["Region"].nunique())

    if filtered_df.empty:
        st.info("No sales match the selected filters.")
    else:
        monthly_sales = (
            filtered_df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]
            .sum()
            .reset_index()
        )
        sales_by_dimension = (
            filtered_df.groupby(breakdown, as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=True)
        )

        st.subheader("Sales trends")
        trend_column, comparison_column = st.columns([1.5, 1])

        with trend_column:
            trend_chart = px.area(
                monthly_sales,
                x="Order Date",
                y="Sales",
                color_discrete_sequence=["#6aaed2"],
                labels={"Order Date": "Month", "Sales": "Sales"},
            )
            trend_chart.update_traces(
                line={"width": 2},
                fillcolor="rgba(106, 174, 210, 0.18)",
                hovertemplate="%{x|%b %Y}<br>Sales: $%{y:,.0f}<extra></extra>",
            )
            trend_chart.update_layout(
                template="plotly_dark",
                height=360,
                margin={"l": 8, "r": 8, "t": 16, "b": 8},
                hovermode="x unified",
                showlegend=False,
                paper_bgcolor="#111c2f",
                plot_bgcolor="#111c2f",
                font={"color": "#b7c6d9"},
                xaxis={"showgrid": False, "zerolinecolor": "#30425a"},
                yaxis={"title": None, "gridcolor": "#26364d", "zerolinecolor": "#30425a"},
            )
            st.plotly_chart(trend_chart, use_container_width=True)

        with comparison_column:
            comparison_chart = px.bar(
                sales_by_dimension,
                x="Sales",
                y=breakdown,
                orientation="h",
                color_discrete_sequence=["#6aaed2"],
                labels={"Sales": "Sales", breakdown: ""},
            )
            comparison_chart.update_traces(
                hovertemplate="%{y}<br>Sales: $%{x:,.0f}<extra></extra>"
            )
            comparison_chart.update_layout(
                template="plotly_dark",
                height=360,
                margin={"l": 8, "r": 8, "t": 16, "b": 8},
                showlegend=False,
                paper_bgcolor="#111c2f",
                plot_bgcolor="#111c2f",
                font={"color": "#b7c6d9"},
                xaxis={
                    "title": None,
                    "gridcolor": "#26364d",
                    "tickformat": "$~s",
                    "nticks": 4,
                    "tickangle": 0,
                    "zerolinecolor": "#30425a",
                },
                yaxis={"title": None, "showgrid": False, "zerolinecolor": "#30425a"},
            )
            st.plotly_chart(comparison_chart, use_container_width=True)

# --------------------------------------------------
# FORECAST EXPLORER
# --------------------------------------------------

elif selected == "Forecast Explorer":

    st.header("Forecast Explorer")
    metrics_path = FORECAST_OUTPUT_DIR / "monthly_model_metrics.csv"
    predictions_path = FORECAST_OUTPUT_DIR / "monthly_test_predictions.csv"
    future_path = FORECAST_OUTPUT_DIR / "monthly_future_forecast.csv"

    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        display_metrics = metrics[
            [
                "model",
                "status",
                "mae",
                "rmse",
                "wape_pct",
                "bias_pct",
                "evaluated_predictions",
            ]
        ].rename(
            columns={
                "model": "Model",
                "status": "Status",
                "mae": "MAE",
                "rmse": "RMSE",
                "wape_pct": "WAPE (%)",
                "bias_pct": "Bias (%)",
                "evaluated_predictions": "Test months",
            }
        )
        st.subheader("Monthly revenue model comparison")
        st.caption(
            "All models use the same chronological three-month holdout. "
            "Metrics are historical test results, not future-forecast accuracy."
        )
        st.dataframe(
            display_metrics.round(2),
            use_container_width=True,
            hide_index=True,
        )

        selected_models = metrics.loc[metrics["selected"].astype(bool), "model"]
        if not selected_models.empty:
            st.info(
                f"Lowest holdout WAPE among models successfully refit for the future "
                f"forecast: **{selected_models.iloc[0]}**. This small holdout does "
                "not establish production readiness."
            )
        failed_models = metrics.loc[metrics["status"].eq("failed")]
        if not failed_models.empty:
            with st.expander("Model execution issues"):
                st.dataframe(
                    failed_models[["model", "error"]],
                    use_container_width=True,
                    hide_index=True,
                )

        test_chart = CHARTS_DIR / "monthly_test_forecasts.png"
        future_chart = CHARTS_DIR / "monthly_future_forecast.png"
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Historical test predictions")
            st.caption("Overall dataset; the shaded period is the held-out test window.")
            if test_chart.is_file():
                st.image(test_chart, use_container_width=True)
            else:
                st.warning("Run the monthly forecasting command to generate this chart.")
        with chart_col2:
            st.subheader("Future revenue forecast")
            st.caption("Forecast dates follow the final observed month.")
            if future_chart.is_file():
                st.image(future_chart, use_container_width=True)
            else:
                st.warning("Run the monthly forecasting command to generate this chart.")

        if predictions_path.is_file():
            with st.expander("Historical test predictions"):
                st.dataframe(
                    pd.read_csv(predictions_path),
                    use_container_width=True,
                    hide_index=True,
                )
        if future_path.is_file():
            st.subheader("Future forecast values")
            st.dataframe(
                pd.read_csv(future_path),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.warning(
            "Monthly forecast results are not available yet. From the repository "
            "root, run `python -m sales_forecasting --data Data\\train.csv "
            "--output Forecasting_Output --charts Charts`."
        )


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------
elif selected == "Anomaly Report":

    st.header("Sales Anomaly Detection")
    anomalies = calculate_weekly_sales_anomalies(df)
    isolation_count = int(anomalies["isolation_forest"].sum())
    z_score_count = int(anomalies["z_score"].sum())
    overlap_count = int(
        (anomalies["isolation_forest"] & anomalies["z_score"]).sum()
    )
    st.caption(
        "Overall dataset; these weekly anomaly flags are not affected by dashboard filters."
    )
    count_col1, count_col2, count_col3 = st.columns(3)
    count_col1.metric("Isolation Forest weeks", isolation_count)
    count_col2.metric("Z-Score weeks", z_score_count)
    count_col3.metric("Weeks flagged by both", overlap_count)
    st.dataframe(
        anomalies[
            [
                "week_ending",
                "sales",
                "direction",
                "detection_methods",
            ]
        ].rename(
            columns={
                "week_ending": "Week ending",
                "sales": "Actual sales",
                "direction": "Unusually",
                "detection_methods": "Flagged by",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Isolation Forest")
        st.caption("Static chart for the overall dataset.")
        st.image(
            CHARTS_DIR / "isolation_forest_anomalies.png",
            use_container_width=True,
        )

    with col2:
        st.subheader("Z-Score Detection")
        st.caption("Static chart for the overall dataset.")
        st.image(CHARTS_DIR / "zscore_anomalies.png", use_container_width=True)

# --------------------------------------------------
# PRODUCT SEGMENTS
# --------------------------------------------------

elif selected == "Product Segments":

    st.header("Product revenue segmentation (ABC)")
    product_segments = calculate_abc_product_segmentation(df)
    segment_summary = product_segments.groupby("segment").agg(
        products=("product_id", "count"),
        revenue=("revenue", "sum"),
        revenue_contribution_pct=("revenue_contribution_pct", "sum"),
    )
    segment_col1, segment_col2, segment_col3 = st.columns(3)
    for column, segment in zip(
        (segment_col1, segment_col2, segment_col3), ("A", "B", "C")
    ):
        row = segment_summary.loc[segment]
        column.metric(
            f"Segment {segment}",
            f"{int(row['products']):,} products",
            f"{row['revenue_contribution_pct']:.2f}% of revenue",
        )
    st.caption(
        "Overall dataset, sorted by historical product revenue. A covers approximately "
        "the first 80% of cumulative revenue, B through approximately 95%, and C the remainder."
    )
    st.dataframe(
        product_segments.rename(
            columns={
                "rank": "Rank",
                "product_id": "Product ID",
                "product_name": "Product Name",
                "revenue": "Revenue",
                "revenue_contribution_pct": "Revenue contribution (%)",
                "cumulative_contribution_pct": "Cumulative contribution (%)",
                "segment": "ABC segment",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

elif selected == "Executive Summary":
    st.header("Executive Business Summary")

    st.markdown(
        """
        <div class="hero-card">
            <strong>Project Completed Successfully</strong><br>
            The forecasting system is now organized into a polished, decision-ready dashboard with business insights and model recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
        ## Key Findings

        - Technology generated the highest revenue in the historical dataset.
        - The shipping KPI measures average order-to-ship days, counted once per order.
        - Revenue forecasts are monthly and predict sales dollars, not quantity or inventory demand.
        - Product ABC groups describe historical revenue contribution, not product growth or demand.
        - Anomaly flags identify unusual weekly sales values; they do not explain their causes.

        ## Interpretation

        Compare models using the same final three-month holdout and review MAE, RMSE, WAPE, and bias.
        A short historical holdout is not evidence of production readiness. Re-run the analysis when
        new sales data becomes available.

        ## Limitations

        The dataset has no quantity, profit, discount, inventory, promotion, or external-market
        fields. Results describe historical revenue only and do not support causal or inventory claims.
        """)
