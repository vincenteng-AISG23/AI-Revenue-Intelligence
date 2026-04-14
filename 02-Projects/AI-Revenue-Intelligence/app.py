# ============================================================
# PROJECT:
# Enterprise AI Revenue Intelligence Platform
#
# FILE:
# app.py
#
# PURPOSE:
# Build a CEO / CIO-ready revenue intelligence dashboard with:
# 1. Executive KPI summary
# 2. Revenue analysis by business line
# 3. Revenue analysis by region
# 4. Profit analysis by product
# 5. Monthly revenue trend
# 6. Group-level revenue forecast
# 7. Business-line forecast accountability
# 8. Plant-level forecast accountability
# 9. Top-customer forecast table
# 10. Margin watchlist
#
# BUSINESS INTENT:
# This dashboard is designed to support senior leadership with:
# - group-level visibility
# - GM accountability by business line
# - plant accountability by country and site
# - customer focus for commercial action
# - margin discipline for profitability improvement
# ============================================================

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression


# ============================================================
# SECTION 1: PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Enterprise AI Revenue Intelligence Platform",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# SECTION 2: HELPER FUNCTIONS FOR COUNTRY / PLANT MAPPING
#
# PURPOSE:
# Enrich the dataset so the dashboard can support accountability
# at country and plant level.
#
# IMPORTANT:
# These are realistic simulated plant mappings for management
# reporting and forecasting use.
# ============================================================
def assign_country(region: str) -> str:
    """
    Map region field into country field.

    Parameters
    ----------
    region : str
        Original region value from the dataset.

    Returns
    -------
    str
        Country value.
    """
    if region == "China":
        return "China"
    if region == "Indonesia":
        return "Indonesia"
    if region == "Thailand":
        return "Thailand"
    if region == "Malaysia":
        return "Malaysia"
    if region == "Singapore":
        return "Singapore"
    return "Other"


def assign_plant(customer_id: str, country: str) -> str:
    """
    Assign a deterministic plant to each customer-country pair.

    WHY DETERMINISTIC:
    A customer should not jump randomly between plants every time
    the app reloads. This function uses a stable hash so the same
    customer always maps to the same plant.

    Parameters
    ----------
    customer_id : str
        Customer identifier.
    country : str
        Country identifier.

    Returns
    -------
    str
        Plant name.
    """
    plant_map = {
        "China": ["Suzhou", "Changchun", "Zhejiang"],
        "Indonesia": ["Surabaya", "Jakarta"],
        "Thailand": ["Chiangmai", "Bangkok"],
        "Malaysia": ["Johor", "Penang", "KL"],
        "Singapore": ["Singapore"],
    }

    plants = plant_map.get(country, ["Unknown"])

    stable_key = f"{customer_id}-{country}"
    stable_hash = hashlib.md5(stable_key.encode()).hexdigest()
    plant_index = int(stable_hash, 16) % len(plants)

    return plants[plant_index]


# ============================================================
# SECTION 3: LOAD DATA
#
# PURPOSE:
# Load transaction data and prepare the time-based and
# organization-based fields needed by the dashboard.
# ============================================================
@st.cache_data

def load_customer_segments() -> pd.DataFrame:
    """
    Load customer segmentation output.
    """
    return pd.read_csv("data/customer_segments.csv")

@st.cache_data
def load_recommendations() -> pd.DataFrame:
    """
    Load product recommendation output.
    """
    return pd.read_csv("data/product_recommendations.csv")


def load_data() -> pd.DataFrame:
    """
    Load and prepare the source transaction dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched transaction data.
    """
    df = pd.read_csv("data/sales_transactions.csv")

    # Convert order date for monthly analytics
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").astype(str)

    # Add country field
    df["Country"] = df["Region"].apply(assign_country)

    # Add plant field using stable mapping
    df["Plant"] = df.apply(
        lambda row: assign_plant(row["CustomerID"], row["Country"]),
        axis=1,
    )

    return df


# ============================================================
# SECTION 4: GENERIC FORECAST FUNCTION
#
# PURPOSE:
# Forecast monthly values using Linear Regression.
#
# NOTE:
# This is a baseline model suitable for first-phase predictive
# analytics. It can be upgraded later to more advanced methods.
# ============================================================
def generate_forecast_from_monthly(
    monthly_df: pd.DataFrame,
    value_column: str,
    forecast_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate forecast from monthly aggregated data.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly aggregated data.
    value_column : str
        Column name to forecast.
    forecast_months : int
        Number of future months to predict.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Historical monthly data and future forecast data.
    """
    historical_df = monthly_df.copy().reset_index(drop=True)
    historical_df["MonthIndex"] = np.arange(len(historical_df))

    X = historical_df[["MonthIndex"]]
    y = historical_df[value_column]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.arange(len(historical_df), len(historical_df) + forecast_months)
    future_df = pd.DataFrame({"MonthIndex": future_index})

    forecast_values = model.predict(future_df[["MonthIndex"]])

    forecast_df = pd.DataFrame(
        {
            "MonthIndex": future_index,
            "ForecastValue": forecast_values,
        }
    )

    return historical_df, forecast_df


# ============================================================
# SECTION 5: GROUP FORECAST
# ============================================================
def generate_group_forecast(df: pd.DataFrame, forecast_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate total group-level revenue forecast.
    """
    monthly = (
        df.groupby("YearMonth", as_index=False)["Revenue"]
        .sum()
        .sort_values("YearMonth")
    )

    return generate_forecast_from_monthly(
        monthly_df=monthly,
        value_column="Revenue",
        forecast_months=forecast_months,
    )


# ============================================================
# SECTION 6: BUSINESS LINE FORECAST ACCOUNTABILITY
# ============================================================
def create_business_line_forecast_table(df: pd.DataFrame, forecast_months: int = 3) -> pd.DataFrame:
    """
    Create business-line accountability table for leadership.

    Returns
    -------
    pd.DataFrame
        Forecast accountability table by business line.
    """
    result_rows: list[dict] = []

    for business_line in sorted(df["BusinessLine"].unique()):
        sub_df = df[df["BusinessLine"] == business_line].copy()

        monthly = (
            sub_df.groupby("YearMonth", as_index=False)
            .agg({"Revenue": "sum", "Profit": "sum"})
            .sort_values("YearMonth")
            .reset_index(drop=True)
        )

        if len(monthly) < 6:
            continue

        last_3_revenue = monthly["Revenue"].tail(3).sum()
        last_3_profit = monthly["Profit"].tail(3).sum()
        last_3_margin = (last_3_profit / last_3_revenue * 100) if last_3_revenue != 0 else 0

        _, revenue_forecast = generate_forecast_from_monthly(
            monthly[["YearMonth", "Revenue"]],
            "Revenue",
            forecast_months,
        )

        _, profit_forecast = generate_forecast_from_monthly(
            monthly[["YearMonth", "Profit"]],
            "Profit",
            forecast_months,
        )

        next_3_revenue = revenue_forecast["ForecastValue"].sum()
        next_3_profit = profit_forecast["ForecastValue"].sum()
        next_3_margin = (next_3_profit / next_3_revenue * 100) if next_3_revenue != 0 else 0

        growth_pct = (
            ((next_3_revenue - last_3_revenue) / last_3_revenue) * 100
            if last_3_revenue != 0 else 0
        )

        if next_3_margin < 20:
            ceo_comment = "Margin watch: improve pricing or reduce cost"
        elif growth_pct < 0:
            ceo_comment = "Revenue softening: GM action required"
        else:
            ceo_comment = "Stable outlook: continue execution discipline"

        result_rows.append(
            {
                "BusinessLine": business_line,
                "Last3M_Revenue": round(last_3_revenue, 2),
                "Next3M_ForecastRevenue": round(next_3_revenue, 2),
                "RevenueGrowthPct": round(growth_pct, 2),
                "Last3M_Profit": round(last_3_profit, 2),
                "Next3M_ForecastProfit": round(next_3_profit, 2),
                "Last3M_MarginPct": round(last_3_margin, 2),
                "Next3M_MarginPct": round(next_3_margin, 2),
                "CEO_Comment": ceo_comment,
            }
        )

    result_df = pd.DataFrame(result_rows)

    if not result_df.empty:
        result_df = result_df.sort_values(by="Next3M_ForecastRevenue", ascending=False)

    return result_df


# ============================================================
# SECTION 7: PLANT-LEVEL FORECAST ACCOUNTABILITY
#
# PURPOSE:
# Build plant-level management accountability by:
# Country -> Plant -> Business Line
# ============================================================
def create_plant_forecast_table(df: pd.DataFrame, forecast_months: int = 3) -> pd.DataFrame:
    """
    Create plant-level forecast accountability table.

    Returns
    -------
    pd.DataFrame
        Forecast accountability table by country, plant,
        and business line.
    """
    result_rows: list[dict] = []

    grouped = df.groupby(["Country", "Plant", "BusinessLine"])

    for (country, plant, business_line), sub_df in grouped:
        monthly = (
            sub_df.groupby("YearMonth", as_index=False)
            .agg({"Revenue": "sum", "Profit": "sum"})
            .sort_values("YearMonth")
            .reset_index(drop=True)
        )

        if len(monthly) < 6:
            continue

        last_3_revenue = monthly["Revenue"].tail(3).sum()
        last_3_profit = monthly["Profit"].tail(3).sum()
        last_3_margin = (last_3_profit / last_3_revenue * 100) if last_3_revenue != 0 else 0

        _, revenue_forecast = generate_forecast_from_monthly(
            monthly[["YearMonth", "Revenue"]],
            "Revenue",
            forecast_months,
        )

        _, profit_forecast = generate_forecast_from_monthly(
            monthly[["YearMonth", "Profit"]],
            "Profit",
            forecast_months,
        )

        next_3_revenue = revenue_forecast["ForecastValue"].sum()
        next_3_profit = profit_forecast["ForecastValue"].sum()
        next_3_margin = (next_3_profit / next_3_revenue * 100) if next_3_revenue != 0 else 0

        growth_pct = (
            ((next_3_revenue - last_3_revenue) / last_3_revenue) * 100
            if last_3_revenue != 0 else 0
        )

        if next_3_margin < 20:
            ceo_action = "Margin risk – cost or pricing action needed"
        elif growth_pct < 0:
            ceo_action = "Revenue decline – GM attention required"
        else:
            ceo_action = "Stable / growing"

        result_rows.append(
            {
                "Country": country,
                "Plant": plant,
                "BusinessLine": business_line,
                "Last3M_Revenue": round(last_3_revenue, 2),
                "Next3M_ForecastRevenue": round(next_3_revenue, 2),
                "RevenueGrowthPct": round(growth_pct, 2),
                "Last3M_Profit": round(last_3_profit, 2),
                "Next3M_ForecastProfit": round(next_3_profit, 2),
                "Last3M_MarginPct": round(last_3_margin, 2),
                "Next3M_MarginPct": round(next_3_margin, 2),
                "CEO_Action": ceo_action,
            }
        )

    result_df = pd.DataFrame(result_rows)

    if not result_df.empty:
        result_df = result_df.sort_values(by="Next3M_ForecastRevenue", ascending=False)

    return result_df


# ============================================================
# SECTION 8: TOP CUSTOMER FORECAST TABLE
# ============================================================
def create_top_customer_forecast_table(
    df: pd.DataFrame,
    top_n: int = 10,
    forecast_months: int = 3,
) -> pd.DataFrame:
    """
    Create forecast table for top customers.
    """
    top_customers = (
        df.groupby(["CustomerName", "BusinessLine"], as_index=False)["Revenue"]
        .sum()
        .sort_values(by="Revenue", ascending=False)
        .head(top_n)
    )

    result_rows: list[dict] = []

    for _, row in top_customers.iterrows():
        customer_name = row["CustomerName"]
        business_line = row["BusinessLine"]

        sub_df = df[
            (df["CustomerName"] == customer_name) &
            (df["BusinessLine"] == business_line)
        ].copy()

        monthly = (
            sub_df.groupby("YearMonth", as_index=False)
            .agg({"Revenue": "sum", "Profit": "sum"})
            .sort_values("YearMonth")
            .reset_index(drop=True)
        )

        if len(monthly) < 4:
            continue

        historical_revenue = monthly["Revenue"].sum()
        historical_profit = monthly["Profit"].sum()
        historical_margin = (
            (historical_profit / historical_revenue) * 100
            if historical_revenue != 0 else 0
        )

        _, revenue_forecast = generate_forecast_from_monthly(
            monthly[["YearMonth", "Revenue"]],
            "Revenue",
            forecast_months,
        )

        forecast_revenue = revenue_forecast["ForecastValue"].sum()

        recent_revenue = monthly["Revenue"].tail(min(3, len(monthly))).sum()

        if historical_margin < 15:
            risk_flag = "Low Margin"
        elif forecast_revenue < recent_revenue:
            risk_flag = "Revenue Risk"
        else:
            risk_flag = "Healthy"

        result_rows.append(
            {
                "CustomerName": customer_name,
                "BusinessLine": business_line,
                "HistoricalRevenue": round(historical_revenue, 2),
                "ForecastNext3MRevenue": round(forecast_revenue, 2),
                "HistoricalProfit": round(historical_profit, 2),
                "MarginPct": round(historical_margin, 2),
                "RiskFlag": risk_flag,
            }
        )

    result_df = pd.DataFrame(result_rows)

    if not result_df.empty:
        result_df = result_df.sort_values(by="ForecastNext3MRevenue", ascending=False)

    return result_df


# ============================================================
# SECTION 9: MARGIN WATCHLIST
# ============================================================
def create_margin_watchlist(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Create a low-margin watchlist for management review.
    """
    watchlist = (
        df.groupby("ProductName", as_index=False)
        .agg({"Revenue": "sum", "Profit": "sum"})
    )

    watchlist["MarginPct"] = np.where(
        watchlist["Revenue"] != 0,
        (watchlist["Profit"] / watchlist["Revenue"]) * 100,
        0,
    )

    watchlist = watchlist.sort_values(
        by=["MarginPct", "Revenue"],
        ascending=[True, False],
    ).head(top_n)

    watchlist["ManagementAction"] = np.where(
        watchlist["MarginPct"] < 20,
        "Review pricing / cost-down actions",
        "Monitor",
    )

    return watchlist


# ============================================================
# SECTION 10: LOAD SOURCE DATA
# ============================================================
df = load_data()

# Load customer segmentation
customer_segments_df = load_customer_segments()

# Merge segmentation into main dataset
df = df.merge(
    customer_segments_df[["CustomerID", "SegmentName"]],
    on="CustomerID",
    how="left"
)

# Load recommender_system
recommendations_df = load_recommendations()

# ============================================================
# SECTION 11: DASHBOARD TITLE
# ============================================================
st.title("📈 Enterprise AI Revenue Intelligence Platform")
st.caption(
    "Executive dashboard for revenue analysis, management accountability, "
    "and predictive business planning."
)

st.markdown(
    """
This dashboard gives senior leadership a structured view of:
- current commercial performance
- business-line accountability
- plant accountability by country and site
- customer concentration
- forecast outlook
- margin pressure points

It is designed to help the CEO, CIO, CFO, and GMs move from passive reporting
to action-oriented performance management.
"""
)


# ============================================================
# SECTION 12: EXECUTIVE KPIS
# ============================================================
total_revenue = df["Revenue"].sum()
total_profit = df["Profit"].sum()
total_orders = df["OrderID"].nunique()
total_customers = df["CustomerID"].nunique()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with kpi2:
    st.metric("Total Profit", f"${total_profit:,.0f}")

with kpi3:
    st.metric("Total Orders", f"{total_orders:,}")

with kpi4:
    st.metric("Total Customers", f"{total_customers:,}")

# ============================================================
# CUSTOMER SEGMENTATION INSIGHTS
# ============================================================
st.subheader("🤖 Customer Segmentation Insights")

# Segment distribution
segment_counts = (
    customer_segments_df["SegmentName"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["Segment", "CustomerCount"]

fig_segment = px.bar(
    segment_counts,
    x="Segment",
    y="CustomerCount",
    title="Customer Segment Distribution",
    text_auto=True,
)

st.plotly_chart(fig_segment, use_container_width=True)

st.subheader("⚠️ High Value Customers At Risk")

at_risk_customers = customer_segments_df[
    customer_segments_df["SegmentName"] == "High Value At Risk"
].sort_values(by="TotalRevenue", ascending=False)

st.dataframe(at_risk_customers.head(20), use_container_width=True)


# ============================================================
# SECTION 13: FILTERS
# ============================================================
st.subheader("Dashboard Filters")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_regions = st.multiselect(
        "Select Region(s)",
        options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique()),
    )

with filter_col2:
    selected_business_lines = st.multiselect(
        "Select Business Line(s)",
        options=sorted(df["BusinessLine"].unique()),
        default=sorted(df["BusinessLine"].unique()),
    )

with filter_col3:
    selected_countries = st.multiselect(
        "Select Country(s)",
        options=sorted(df["Country"].unique()),
        default=sorted(df["Country"].unique()),
    )

filtered_df = df[
    (df["Region"].isin(selected_regions)) &
    (df["BusinessLine"].isin(selected_business_lines)) &
    (df["Country"].isin(selected_countries))
].copy()


# ============================================================
# SECTION 14: FILTERED KPIS
# ============================================================
filtered_revenue = filtered_df["Revenue"].sum()
filtered_profit = filtered_df["Profit"].sum()
filtered_orders = filtered_df["OrderID"].nunique()
filtered_customers = filtered_df["CustomerID"].nunique()

st.subheader("Filtered Executive Summary")

fkpi1, fkpi2, fkpi3, fkpi4 = st.columns(4)

with fkpi1:
    st.metric("Filtered Revenue", f"${filtered_revenue:,.0f}")

with fkpi2:
    st.metric("Filtered Profit", f"${filtered_profit:,.0f}")

with fkpi3:
    st.metric("Filtered Orders", f"{filtered_orders:,}")

with fkpi4:
    st.metric("Filtered Customers", f"{filtered_customers:,}")


# ============================================================
# SECTION 15: REVENUE BY BUSINESS LINE
# ============================================================
revenue_by_business = (
    filtered_df.groupby("BusinessLine", as_index=False)["Revenue"]
    .sum()
    .sort_values(by="Revenue", ascending=False)
)

fig_business = px.bar(
    revenue_by_business,
    x="BusinessLine",
    y="Revenue",
    title="Revenue by Business Line",
    text_auto=".2s",
)

st.plotly_chart(fig_business, use_container_width=True)


# ============================================================
# SECTION 16: REVENUE BY REGION
# ============================================================
revenue_by_region = (
    filtered_df.groupby("Region", as_index=False)["Revenue"]
    .sum()
    .sort_values(by="Revenue", ascending=False)
)

fig_region = px.bar(
    revenue_by_region,
    x="Region",
    y="Revenue",
    title="Revenue by Region",
    text_auto=".2s",
)

st.plotly_chart(fig_region, use_container_width=True)


# ============================================================
# SECTION 17: PROFIT BY PRODUCT
# ============================================================
profit_by_product = (
    filtered_df.groupby("ProductName", as_index=False)["Profit"]
    .sum()
    .sort_values(by="Profit", ascending=False)
)

fig_profit = px.bar(
    profit_by_product,
    x="ProductName",
    y="Profit",
    title="Profit by Product",
    text_auto=".2s",
)

st.plotly_chart(fig_profit, use_container_width=True)


# ============================================================
# SECTION 18: MONTHLY REVENUE TREND
# ============================================================
monthly_revenue = (
    filtered_df.groupby("YearMonth", as_index=False)["Revenue"]
    .sum()
    .sort_values(by="YearMonth")
)

fig_monthly = px.line(
    monthly_revenue,
    x="YearMonth",
    y="Revenue",
    title="Monthly Revenue Trend",
    markers=True,
)

st.plotly_chart(fig_monthly, use_container_width=True)


# ============================================================
# SECTION 19: GROUP FORECAST
# ============================================================
st.subheader("📈 Group Revenue Forecast (Next 6 Months)")

group_monthly_data, group_forecast_data = generate_group_forecast(
    filtered_df,
    forecast_months=6,
)

group_forecast_fig = go.Figure()

group_forecast_fig.add_trace(
    go.Scatter(
        x=group_monthly_data["MonthIndex"],
        y=group_monthly_data["Revenue"],
        mode="lines+markers",
        name="Actual Revenue",
    )
)

group_forecast_fig.add_trace(
    go.Scatter(
        x=group_forecast_data["MonthIndex"],
        y=group_forecast_data["ForecastValue"],
        mode="lines+markers",
        name="Forecast Revenue",
        line=dict(dash="dash"),
    )
)

group_forecast_fig.update_layout(
    title="Group Revenue Forecast vs Actual",
    xaxis_title="Time Index",
    yaxis_title="Revenue",
)

st.plotly_chart(group_forecast_fig, use_container_width=True)


# ============================================================
# SECTION 20: BUSINESS LINE FORECAST ACCOUNTABILITY
# ============================================================
st.subheader("Business Line Forecast Accountability")

business_line_forecast_df = create_business_line_forecast_table(
    filtered_df,
    forecast_months=3,
)

st.dataframe(business_line_forecast_df, use_container_width=True)


# ============================================================
# SECTION 21: PLANT-LEVEL FORECAST ACCOUNTABILITY
# ============================================================
st.subheader("🏭 Plant-Level Forecast Accountability")

plant_forecast_df = create_plant_forecast_table(
    filtered_df,
    forecast_months=3,
)

st.dataframe(plant_forecast_df, use_container_width=True)


# ============================================================
# SECTION 22: TOP CUSTOMER FORECAST TABLE
# ============================================================
st.subheader("Top Customer Forecast Table")

top_customer_forecast_df = create_top_customer_forecast_table(
    filtered_df,
    top_n=10,
    forecast_months=3,
)

st.dataframe(top_customer_forecast_df, use_container_width=True)


# ============================================================
# SECTION 23: MARGIN WATCHLIST
# ============================================================
st.subheader("Margin Watchlist")

margin_watchlist_df = create_margin_watchlist(filtered_df, top_n=10)

st.dataframe(margin_watchlist_df, use_container_width=True)

# ============================================================
# PRODUCT RECOMMENDATION ENGINE
# ============================================================
st.subheader("🧠 AI Product Recommendation Engine")

st.markdown(
    """
This section shows cross-sell opportunities based on customer
purchase behavior. It identifies products that are frequently
bought together by the same customers.
"""
)

# Select product
selected_product = st.selectbox(
    "Select a product to see recommendations:",
    sorted(recommendations_df["BaseProduct"].unique())
)

# Filter recommendations
filtered_recs = recommendations_df[
    recommendations_df["BaseProduct"] == selected_product
].sort_values(by="CoPurchaseScore", ascending=False)

st.write(f"### Recommended products for: {selected_product}")
st.dataframe(filtered_recs, use_container_width=True)

# ============================================================
# SECTION 24: TRANSACTION PREVIEW
# ============================================================
st.subheader("Transaction Data Preview")
st.dataframe(filtered_df.head(50), use_container_width=True)


# ============================================================
# SECTION 25: EXECUTIVE INTERPRETATION
# ============================================================
st.markdown(
    """
---
### Executive Interpretation

This dashboard now supports leadership at multiple levels:

- **Group level** for total company outlook
- **Business-line level** for GM accountability
- **Plant level** for site-by-site performance pressure
- **Customer level** for commercial focus and retention action
- **Margin watchlist** for pricing, sourcing, and cost discipline

This makes the platform more than a reporting dashboard.
It becomes a management tool for revenue growth, profitability
improvement, and executive decision-making.
"""
)