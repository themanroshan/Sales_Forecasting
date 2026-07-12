import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Sales Forecasting System", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #111827 100%);
    color: #f8fafc;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #111827 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.22);
}

[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(99, 102, 241, 0.32);
    border-radius: 18px;
    box-shadow: 0 10px 35px rgba(15, 23, 42, 0.25);
    padding: 14px 16px;
}

div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

h1, h2, h3, h4, h5, h6 {
    color: #8b5cf6;
}

.hero-card {
    padding: 1.3rem 1.4rem;
    border-radius: 24px;
    background: linear-gradient(120deg, rgba(37, 99, 235, 0.22), rgba(139, 92, 246, 0.30));
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
    margin-bottom: 1rem;
}

.pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    color: #e2e8f0;
    font-size: 0.84rem;
    margin-top: 0.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("Data/train.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Year"] = df["Order Date"].dt.year
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size: 1.1rem; font-weight: 700;">Sales Intelligence Hub</div>
            <div style="color: #cbd5e1; margin-top: 0.3rem;">AI-powered forecasting cockpit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        "Dashboard Menu",
        [
            "Sales Overview",
            "Forecast Explorer",
            "Anomaly Report",
            "Product Segments",
            "Executive Summary",
        ],
        icons=[
            "house",
            "graph-up",
            "exclamation-triangle",
            "boxes",
            "file-earmark-text",
        ],
        menu_icon="cpu",
        default_index=0,
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin: 0;">📊 End-to-End Sales Forecasting & Demand Intelligence System</h1>
        <p style="margin: 0.45rem 0 0; color: #e2e8f0;">A modern forecasting workspace blending analytics, forecasting accuracy, and business clarity.</p>
        <span class="pill">Developed by Roshan Manjal • Xylofy AI Internship Project</span>
    </div>
    """,
    unsafe_allow_html=True,
)
# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if selected == "Sales Overview":

    st.header("Dashboard Overview")

    # Filters

    colf1, colf2 = st.columns(2)

    with colf1:
        selected_category = st.multiselect(
            "Category", df["Category"].unique(), default=df["Category"].unique()
        )

    with colf2:
        selected_region = st.multiselect(
            "Region", df["Region"].unique(), default=df["Region"].unique()
        )

    filtered_df = df[
        (df["Category"].isin(selected_category)) & (df["Region"].isin(selected_region))
    ]

    # KPI Cards

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")

    col2.metric("Orders", len(filtered_df))

    col3.metric("Categories", filtered_df["Category"].nunique())

    col4.metric("Regions", filtered_df["Region"].nunique())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Category Sales Trend")

        st.image("Charts/Category_Sales_Trend.png", use_container_width=True)

    with col2:

        st.subheader("Regional Sales Trend")

        st.image("Charts/Regional_Sales_Trend.png", use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Monthly Sales Trend")

        st.image("Charts/Monthly_Trend.png", use_container_width=True)

    with col4:

        st.subheader("Average Shipping Time")

        st.image("Charts/Avg_Shipping_Time.png", use_container_width=True)

# --------------------------------------------------
# FORECAST EXPLORER
# --------------------------------------------------

elif selected == "Forecast Explorer":

    st.header("Forecast Explorer")

    comparison = pd.DataFrame(
        {
            "Model": ["SARIMA", "Prophet", "XGBoost"],
            "MAE": [18031.40, 20250.79, 14163.32],
            "RMSE": [19009.18, 22318.41, 19267.49],
            "MAPE": ["18.97%", "21.86%", "13.42%"],
        }
    )

    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="hero-card">
            <strong>Recommended Production Model:</strong> XGBoost<br>
            XGBoost achieved the lowest forecasting error and is recommended for production deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("SARIMA Forecast")

        st.image("Charts/Sarima_Forecast_vs_Actual.png", use_container_width=True)

    with col2:

        st.subheader("Prophet Forecast")

        st.image("Charts/Prophet_Forecast.png", use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Prophet Components")

        st.image("Charts/Prophet_Components.png", use_container_width=True)

    with col4:

        st.subheader("XGBoost Forecast")

        st.image("Charts/XGBoost_vs_actual.png", use_container_width=True)

    st.markdown("---")

    st.markdown("""
    ### Final Recommendation

    Based on model evaluation results:

    - SARIMA MAPE = 18.97%
    - Prophet MAPE = 21.86%
    - XGBoost MAPE = 13.42%

    XGBoost achieved the best forecasting performance and is recommended for deployment in production environments.
    """)


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------
elif selected == "Anomaly Report":

    st.header("Sales Anomaly Detection")

    st.markdown(
        """
        <div class="hero-card">
            Both Isolation Forest and Z-Score detected 11 anomalous sales periods, highlighting unusual demand behavior for deeper review.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Isolation Forest")
        st.image(
            "Charts/Isolation_Forest_Anamoly_Detection.png", use_container_width=True
        )

    with col2:
        st.subheader("Z-Score Detection")
        st.image("Charts/Z-Score_Anamoly_Detection.png", use_container_width=True)

# --------------------------------------------------
# PRODUCT SEGMENTS
# --------------------------------------------------

elif selected == "Product Segments":

    st.header("Product Demand Segmentation")

    col1, col2 = st.columns(2)

    with col1:
        st.image("Charts/Elbow_Method.png", use_container_width=True)

    with col2:
        st.image("Charts/Product_Demand.png", use_container_width=True)

    st.markdown(
        """
        <div class="hero-card">
            <strong>3 Product Demand Clusters Identified</strong>
            <ul style="margin: 0.4rem 0 0 1rem; color: #e2e8f0;">
                <li>Cluster 0 — High Value Premium Products</li>
                <li>Cluster 1 — Low Volume Stable Demand Products</li>
                <li>Cluster 2 — High Volume Growth Products</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cluster_df = pd.DataFrame(
        {
            "Cluster": [0, 1, 2],
            "Description": [
                "Premium High Value Products",
                "Stable Low Volume Products",
                "High Demand Growth Products",
            ],
        }
    )

    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

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

        - Technology generated the highest revenue.
        - West region showed the strongest growth.
        - Average shipping time was approximately 4 days.
        - Sales exhibited seasonal behavior.
        - XGBoost achieved the best forecasting accuracy.
        - 11 anomalies were detected.
        - 3 product demand clusters were identified.

        ## Business Recommendations

        1. Increase inventory for high-demand products.
        2. Use XGBoost for production forecasting.
        3. Align promotions with seasonal demand spikes.

        ## Risk

        Forecasts are based on historical data and future market conditions may differ.
        """)
