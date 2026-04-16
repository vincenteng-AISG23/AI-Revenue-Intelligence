# ============================================================
# ENTERPRISE AI REVENUE INTELLIGENCE PLATFORM (CIO VERSION)
# Keeps the existing project file structure unchanged
#
# Expected existing files:
# - data/sales_transactions.csv
# - data/customer_segments.csv
# - data/product_recommendations.csv
# ============================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Enterprise AI Revenue Intelligence Platform",
    layout="wide",
)

# ============================================================
# FILE PATHS
# ============================================================
DATA_DIR = Path("data")
SALES_FILE = DATA_DIR / "sales_transactions.csv"
SEGMENTS_FILE = DATA_DIR / "customer_segments.csv"
RECOMMENDATIONS_FILE = DATA_DIR / "product_recommendations.csv"

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 800;
            margin-bottom: 0.2rem;
            color: #111827;
        }
        .sub-title {
            font-size: 17px;
            color: #4B5563;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 28px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.2rem;
        }
        .section-note {
            font-size: 15px;
            color: #6B7280;
            margin-bottom: 0.9rem;
        }
        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 14px;
        }
        section[data-testid="stSidebar"] {
            background-color: #F8FAFC;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def fmt_currency(value: float) -> str:
    return f"${value:,.0f}"


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_sales_data() -> pd.DataFrame:
    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {SALES_FILE}. Please keep using your existing working data folder."
        )
    return pd.read_csv(SALES_FILE)


@st.cache_data
def load_segments_data() -> pd.DataFrame | None:
    if SEGMENTS_FILE.exists():
        return pd.read_csv(SEGMENTS_FILE)
    return None


@st.cache_data
def load_recommendations_data() -> pd.DataFrame | None:
    if RECOMMENDATIONS_FILE.exists():
        return pd.read_csv(RECOMMENDATIONS_FILE)
    return None


# ============================================================
# PREPARE SALES DATA
# ============================================================
def prepare_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = df.copy()

    date_col = find_column(prepared_df, ["OrderDate", "Date", "InvoiceDate"])
    revenue_col = find_column(prepared_df, ["Revenue", "SalesAmount", "NetSales"])
    profit_col = find_column(prepared_df, ["Profit", "GrossProfit"])
    qty_col = find_column(prepared_df, ["Quantity", "Qty"])
    unit_price_col = find_column(prepared_df, ["UnitPrice", "Price"])
    discount_col = find_column(prepared_df, ["Discount", "DiscountPct"])
    customer_id_col = find_column(prepared_df, ["CustomerID", "CustomerId"])
    customer_name_col = find_column(prepared_df, ["CustomerName"])
    region_col = find_column(prepared_df, ["Region", "Country"])
    plant_col = find_column(prepared_df, ["Plant"])
    business_line_col = find_column(prepared_df, ["BusinessLine", "Business_Line"])
    sales_channel_col = find_column(prepared_df, ["SalesChannel", "Channel"])
    product_name_col = find_column(prepared_df, ["ProductName"])
    category_col = find_column(prepared_df, ["Category"])

    if date_col:
        prepared_df[date_col] = pd.to_datetime(prepared_df[date_col], errors="coerce")

    if revenue_col is None and qty_col and unit_price_col:
        prepared_df["Revenue"] = (
            prepared_df[qty_col].fillna(0) * prepared_df[unit_price_col].fillna(0)
        )
        if discount_col:
            discount_series = prepared_df[discount_col].fillna(0)
            if discount_series.max() > 1:
                discount_series = discount_series / 100.0
            prepared_df["Revenue"] = prepared_df["Revenue"] * (1 - discount_series)
        revenue_col = "Revenue"

    if profit_col is None and revenue_col:
        if "Cost" in prepared_df.columns:
            prepared_df["Profit"] = prepared_df[revenue_col] - prepared_df["Cost"]
        else:
            prepared_df["Profit"] = prepared_df[revenue_col] * 0.35
        profit_col = "Profit"

    rename_map = {}
    if date_col and date_col != "OrderDate":
        rename_map[date_col] = "OrderDate"
    if revenue_col and revenue_col != "Revenue":
        rename_map[revenue_col] = "Revenue"
    if profit_col and profit_col != "Profit":
        rename_map[profit_col] = "Profit"
    if customer_id_col and customer_id_col != "CustomerID":
        rename_map[customer_id_col] = "CustomerID"
    if customer_name_col and customer_name_col != "CustomerName":
        rename_map[customer_name_col] = "CustomerName"
    if region_col and region_col != "Country":
        rename_map[region_col] = "Country"
    if plant_col and plant_col != "Plant":
        rename_map[plant_col] = "Plant"
    if business_line_col and business_line_col != "BusinessLine":
        rename_map[business_line_col] = "BusinessLine"
    if sales_channel_col and sales_channel_col != "SalesChannel":
        rename_map[sales_channel_col] = "SalesChannel"
    if product_name_col and product_name_col != "ProductName":
        rename_map[product_name_col] = "ProductName"
    if category_col and category_col != "Category":
        rename_map[category_col] = "Category"

    prepared_df = prepared_df.rename(columns=rename_map)

    if "OrderDate" in prepared_df.columns:
        prepared_df["MonthStart"] = prepared_df["OrderDate"].dt.to_period("M").dt.to_timestamp()

    return prepared_df


# ============================================================
# FORECASTING
# ============================================================
def build_monthly_forecast(df: pd.DataFrame, forecast_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "MonthStart" not in df.columns or "Revenue" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    monthly_df = (
        df.groupby("MonthStart", as_index=False)["Revenue"]
        .sum()
        .sort_values("MonthStart")
        .reset_index(drop=True)
    )

    if monthly_df.empty or len(monthly_df) < 2:
        return monthly_df, pd.DataFrame()

    monthly_df["t"] = np.arange(len(monthly_df))

    model = LinearRegression()
    model.fit(monthly_df[["t"]], monthly_df["Revenue"])

    future_t = np.arange(len(monthly_df), len(monthly_df) + forecast_months)
    future_dates = pd.date_range(
        start=monthly_df["MonthStart"].max() + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    )

    forecast_values = model.predict(pd.DataFrame({"t": future_t}))
    forecast_values = np.maximum(forecast_values, 0)

    forecast_df = pd.DataFrame(
        {
            "MonthStart": future_dates,
            "ForecastRevenue": forecast_values,
        }
    )

    return monthly_df, forecast_df


# ============================================================
# CUSTOMER SEGMENT SUMMARY
# ============================================================
def summarize_customer_segments(segment_df: pd.DataFrame | None) -> pd.DataFrame:
    if segment_df is None or segment_df.empty:
        return pd.DataFrame()

    segment_col = find_column(segment_df, ["SegmentName"])
    revenue_col = find_column(segment_df, ["TotalRevenue", "Revenue"])
    profit_col = find_column(segment_df, ["TotalProfit", "Profit"])
    recency_col = find_column(segment_df, ["LastPurchaseDaysAgo", "AvgRecency"])

    if segment_col is None:
        return pd.DataFrame()

    summary = segment_df.groupby(segment_col).size().reset_index(name="Customers")

    if revenue_col:
        rev_sum = segment_df.groupby(segment_col)[revenue_col].sum().reset_index(name="SegmentRevenue")
        rev_avg = segment_df.groupby(segment_col)[revenue_col].mean().reset_index(name="AvgRevenuePerCustomer")
        summary = summary.merge(rev_sum, on=segment_col, how="left")
        summary = summary.merge(rev_avg, on=segment_col, how="left")

    if profit_col:
        prof_avg = segment_df.groupby(segment_col)[profit_col].mean().reset_index(name="AvgProfitPerCustomer")
        summary = summary.merge(prof_avg, on=segment_col, how="left")

    if recency_col:
        rec_avg = segment_df.groupby(segment_col)[recency_col].mean().reset_index(name="AvgRecencyDays")
        summary = summary.merge(rec_avg, on=segment_col, how="left")

    strategy_map = {
        "High Value Active": "Protect & Grow",
        "Low Engagement": "Reactivation Campaign",
        "High Value At Risk": "Immediate Sales Intervention",
    }

    priority_map = {
        "High Value Active": "Medium",
        "Low Engagement": "High",
        "High Value At Risk": "Critical",
    }

    summary["Strategy"] = summary[segment_col].map(strategy_map).fillna("Review")
    summary["Priority"] = summary[segment_col].map(priority_map).fillna("Medium")

    return summary.sort_values("Customers", ascending=False)


# ============================================================
# MAIN LOAD
# ============================================================
try:
    sales_raw = load_sales_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

sales_df = prepare_sales_data(sales_raw)
segments_df = load_segments_data()
recommendations_df = load_recommendations_data()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("📊 Executive Filters")

country_options = sorted(sales_df["Country"].dropna().unique()) if "Country" in sales_df.columns else []
product_options = sorted(sales_df["ProductName"].dropna().unique()) if "ProductName" in sales_df.columns else []
business_line_options = sorted(sales_df["BusinessLine"].dropna().unique()) if "BusinessLine" in sales_df.columns else []
channel_options = sorted(sales_df["SalesChannel"].dropna().unique()) if "SalesChannel" in sales_df.columns else []

selected_countries = st.sidebar.multiselect(
    "Country",
    options=country_options,
    default=country_options,
)

selected_products = st.sidebar.multiselect(
    "Product",
    options=product_options,
    default=product_options,
)

selected_business_lines = st.sidebar.multiselect(
    "Business Line",
    options=business_line_options,
    default=business_line_options,
)

selected_channels = st.sidebar.multiselect(
    "Sales Channel",
    options=channel_options,
    default=channel_options,
)

filtered_df = sales_df.copy()

if selected_countries and "Country" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

if selected_products and "ProductName" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["ProductName"].isin(selected_products)]

if selected_business_lines and "BusinessLine" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["BusinessLine"].isin(selected_business_lines)]

if selected_channels and "SalesChannel" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["SalesChannel"].isin(selected_channels)]

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">📈 Enterprise AI Revenue Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Executive decision platform for revenue visibility, forecasting, customer intelligence, and AI-driven commercial action across medical equipment, autonomous components, and smart-device manufacturing.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
This platform demonstrates how **AI/ML models are embedded into commercial decision-making**:

- Revenue and profitability visibility  
- Predictive forecasting for proactive planning  
- Customer segmentation for targeted intervention  
- AI-driven cross-sell recommendation logic  

Designed for **CIO-level strategy, accountability, and business impact**.
"""
)

# ============================================================
# NAVIGATION
# ============================================================
section = st.radio(
    "Select View",
    [
        "1. KPI Overview",
        "2. Revenue & Profit Analysis",
        "3. Forecasting & Planning",
        "4. Customer Intelligence",
        "5. AI Recommendation Engine",
    ],
    horizontal=True,
)

# ============================================================
# 1. KPI OVERVIEW
# ============================================================
if section == "1. KPI Overview":
    st.markdown('<div class="section-title">1️⃣ KPI Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Leadership summary of the current business scope across selected countries, products, business lines, and channels.</div>',
        unsafe_allow_html=True,
    )

    total_revenue = filtered_df["Revenue"].sum() if "Revenue" in filtered_df.columns else 0
    total_profit = filtered_df["Profit"].sum() if "Profit" in filtered_df.columns else 0
    margin_pct = safe_divide(total_profit, total_revenue) * 100
    total_orders = len(filtered_df)
    total_customers = filtered_df["CustomerID"].nunique() if "CustomerID" in filtered_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Revenue", fmt_currency(total_revenue))
    with c2:
        st.metric("Total Profit", fmt_currency(total_profit))
    with c3:
        st.metric("Profit Margin", f"{margin_pct:.2f}%")
    with c4:
        st.metric("Active Customers", f"{total_customers:,}")

    st.info("This screen provides a concise executive snapshot of commercial performance and portfolio scale.")

# ============================================================
# 2. REVENUE & PROFIT ANALYSIS
# ============================================================
elif section == "2. Revenue & Profit Analysis":
    st.markdown('<div class="section-title">2️⃣ Revenue & Profit Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Commercial analysis by business line, country, and product to identify revenue concentration, profitability, and margin risk.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Revenue Contribution by Business Line")
            if "BusinessLine" in filtered_df.columns:
                rev_bl = filtered_df.groupby("BusinessLine")["Revenue"].sum().reset_index()
                fig = px.bar(
                    rev_bl,
                    x="BusinessLine",
                    y="Revenue",
                    color="BusinessLine",
                    title="Revenue Contribution by Business Line",
                    labels={"BusinessLine": "Business Line", "Revenue": "Revenue (USD)"},
                    text_auto=".2s",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Revenue Contribution by Region")
            if "Country" in filtered_df.columns:
                rev_region = filtered_df.groupby("Country")["Revenue"].sum().reset_index()
                fig = px.bar(
                    rev_region,
                    x="Country",
                    y="Revenue",
                    color="Country",
                    title="Revenue Contribution by Region",
                    labels={"Country": "Country / Region", "Revenue": "Revenue (USD)"},
                    text_auto=".2s",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.subheader("Profit Contribution by Product")
            if {"ProductName", "Profit"}.issubset(filtered_df.columns):
                profit_prod = (
                    filtered_df.groupby("ProductName")["Profit"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                fig = px.bar(
                    profit_prod,
                    x="ProductName",
                    y="Profit",
                    color="ProductName",
                    title="Profit Contribution by Product",
                    labels={"ProductName": "Product", "Profit": "Profit (USD)"},
                    text_auto=".2s",
                )
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with col4:
        with st.container(border=True):
            st.subheader("Margin Risk and Action Watchlist")
            if {"BusinessLine", "ProductName", "Revenue", "Profit"}.issubset(filtered_df.columns):
                margin_df = (
                    filtered_df.groupby(["BusinessLine", "ProductName"], as_index=False)[["Revenue", "Profit"]]
                    .sum()
                )
                margin_df["MarginPct"] = np.where(
                    margin_df["Revenue"] > 0,
                    (margin_df["Profit"] / margin_df["Revenue"]) * 100,
                    0,
                )

                def risk_level(m: float) -> str:
                    if m < 30:
                        return "High Risk"
                    if m < 35:
                        return "Watch"
                    return "Healthy"

                def action_label(m: float) -> str:
                    if m < 30:
                        return "Cost / pricing review required"
                    if m < 35:
                        return "Monitor closely"
                    return "Healthy"

                margin_df["RiskLevel"] = margin_df["MarginPct"].apply(risk_level)
                margin_df["ManagementAction"] = margin_df["MarginPct"].apply(action_label)

                st.dataframe(
                    margin_df.sort_values("MarginPct", ascending=True).head(10),
                    use_container_width=True,
                )

    st.info("This screen helps leadership identify where commercial performance is concentrated and where profitability discipline is required.")

# ============================================================
# 3. FORECASTING & PLANNING
# ============================================================
elif section == "3. Forecasting & Planning":
    st.markdown('<div class="section-title">3️⃣ Forecasting & Planning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">ML-based revenue forecasting and business-line outlook to support commercial planning, accountability, and early intervention.</div>',
        unsafe_allow_html=True,
    )

    monthly_df, forecast_df = build_monthly_forecast(filtered_df, forecast_months=3)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Monthly Revenue Trend and Forecast")
            if not monthly_df.empty and not forecast_df.empty:
                history_df = monthly_df.copy()
                history_df["Series"] = "Historical Revenue"
                history_df = history_df.rename(columns={"Revenue": "Value"})

                future_df = forecast_df.copy()
                future_df["Series"] = "Forecast Revenue"
                future_df = future_df.rename(columns={"ForecastRevenue": "Value"})

                chart_df = pd.concat(
                    [
                        history_df[["MonthStart", "Value", "Series"]],
                        future_df[["MonthStart", "Value", "Series"]],
                    ],
                    ignore_index=True,
                )

                fig = px.line(
                    chart_df,
                    x="MonthStart",
                    y="Value",
                    color="Series",
                    markers=True,
                    title="Revenue Forecast (AI Model)",
                    labels={"MonthStart": "Month", "Value": "Revenue (USD)", "Series": "Series"},
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough monthly data to generate forecast.")

    with col2:
        with st.container(border=True):
            st.subheader("Business Line Performance Outlook")

            if (
                "BusinessLine" in filtered_df.columns
                and "MonthStart" in filtered_df.columns
                and "Revenue" in filtered_df.columns
                and "Profit" in filtered_df.columns
                and not forecast_df.empty
            ):
                latest_month = filtered_df["MonthStart"].max()
                last3_df = filtered_df[
                    filtered_df["MonthStart"] >= latest_month - pd.DateOffset(months=2)
                ].copy()

                hist = (
                    last3_df.groupby("BusinessLine", as_index=False)[["Revenue", "Profit"]]
                    .sum()
                    .rename(columns={"Revenue": "Last3MRevenue", "Profit": "Last3MProfit"})
                )

                total_hist = hist["Last3MRevenue"].sum()
                total_forecast = forecast_df["ForecastRevenue"].sum()

                hist["Next3MForecastRevenue"] = np.where(
                    total_hist > 0,
                    (hist["Last3MRevenue"] / total_hist) * total_forecast,
                    0,
                )

                hist["RevenueGrowthPct"] = np.where(
                    hist["Last3MRevenue"] > 0,
                    ((hist["Next3MForecastRevenue"] - hist["Last3MRevenue"]) / hist["Last3MRevenue"]) * 100,
                    0,
                )

                hist["Last3MMarginPct"] = np.where(
                    hist["Last3MRevenue"] > 0,
                    (hist["Last3MProfit"] / hist["Last3MRevenue"]) * 100,
                    0,
                )

                hist["Next3MForecastMarginPct"] = np.where(
                    hist["RevenueGrowthPct"] < 0,
                    hist["Last3MMarginPct"] - 1.0,
                    hist["Last3MMarginPct"] + 0.5,
                )

                def outlook_label(row: pd.Series) -> str:
                    if row["RevenueGrowthPct"] < -5 and row["Next3MForecastMarginPct"] < row["Last3MMarginPct"]:
                        return "High-risk business line"
                    if row["RevenueGrowthPct"] < 0:
                        return "Revenue softening"
                    if row["Next3MForecastMarginPct"] < row["Last3MMarginPct"]:
                        return "Margin pressure"
                    return "Stable / positive outlook"

                hist["Outlook"] = hist.apply(outlook_label, axis=1)

                st.dataframe(hist, use_container_width=True)
            else:
                st.warning("Business line outlook cannot be generated from current data.")

    forecast_value = forecast_df["ForecastRevenue"].sum() if not forecast_df.empty else 0
    st.info(
        f"The forecast model indicates an estimated {fmt_currency(forecast_value)} in projected revenue over the next 3 months based on recent trend patterns."
    )

# ============================================================
# 4. CUSTOMER INTELLIGENCE
# ============================================================
elif section == "4. Customer Intelligence":
    st.markdown('<div class="section-title">4️⃣ Customer Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Customer segmentation translated into prioritisation, risk visibility, and commercial intervention logic.</div>',
        unsafe_allow_html=True,
    )

    segment_summary_df = summarize_customer_segments(segments_df)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Customer Segment Distribution")
            if not segment_summary_df.empty and "SegmentName" in segment_summary_df.columns:
                fig = px.bar(
                    segment_summary_df,
                    x="SegmentName",
                    y="Customers",
                    color="SegmentName",
                    title="Customer Segment Distribution",
                    labels={"SegmentName": "Segment", "Customers": "Customer Count"},
                    text_auto=True,
                )
                fig.update_layout(height=430, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Customer segment data is not available.")

    with col2:
        with st.container(border=True):
            st.subheader("Segment Summary")
            if not segment_summary_df.empty:
                st.dataframe(segment_summary_df, use_container_width=True)
            else:
                st.warning("Segment summary is not available.")

    with st.container(border=True):
        st.subheader("High Value Customers at Risk")

        if segments_df is not None and not segments_df.empty:
            segment_col = find_column(segments_df, ["SegmentName"])
            revenue_col = find_column(segments_df, ["TotalRevenue", "Revenue"])
            recency_col = find_column(segments_df, ["LastPurchaseDaysAgo"])
            customer_id_col = find_column(segments_df, ["CustomerID"])
            customer_name_col = find_column(segments_df, ["CustomerName"])
            profit_col = find_column(segments_df, ["TotalProfit", "Profit"])
            orders_col = find_column(segments_df, ["TotalOrders", "Orders"])
            avg_order_col = find_column(segments_df, ["AvgOrderValue"])

            if segment_col and revenue_col and recency_col:
                at_risk_df = segments_df[
                    segments_df[segment_col].astype(str).str.contains("At Risk", case=False, na=False)
                ].copy()

                if not at_risk_df.empty:
                    at_risk_df["RiskScore"] = (
                        at_risk_df[recency_col].fillna(0) * 0.5
                        + at_risk_df[revenue_col].fillna(0) * -0.00001
                    )
                    at_risk_df["Action"] = "Sales Director to engage within 7 days"

                    cols = [
                        c for c in [
                            customer_id_col,
                            customer_name_col,
                            revenue_col,
                            profit_col,
                            orders_col,
                            avg_order_col,
                            recency_col,
                            segment_col,
                            "RiskScore",
                            "Action",
                        ] if c and c in at_risk_df.columns
                    ]

                    st.dataframe(
                        at_risk_df.sort_values("RiskScore", ascending=False).head(10)[cols],
                        use_container_width=True,
                    )
                else:
                    st.info("No high-value at-risk customers identified.")
            else:
                st.warning("Required segmentation columns are missing.")
        else:
            st.warning("customer_segments.csv is not available.")

# ============================================================
# 5. AI RECOMMENDATION ENGINE
# ============================================================
elif section == "5. AI Recommendation Engine":
    st.markdown('<div class="section-title">5️⃣ AI Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Recommendation logic translated into cross-sell opportunity, commercial targeting, and estimated revenue uplift.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Cross-Sell Revenue Opportunity")

        if recommendations_df is not None and not recommendations_df.empty:
            base_product_col = find_column(recommendations_df, ["BaseProduct"])
            reco_product_col = find_column(recommendations_df, ["RecommendedProduct"])
            score_col = find_column(recommendations_df, ["CoPurchaseScore", "Score"])

            if base_product_col and reco_product_col and score_col:
                selected_base_product = st.selectbox(
                    "Select Base Product",
                    sorted(recommendations_df[base_product_col].dropna().unique())
                )

                reco_df = recommendations_df[
                    recommendations_df[base_product_col] == selected_base_product
                ].copy()

                if {"ProductName", "UnitPrice"}.issubset(filtered_df.columns):
                    avg_price_df = (
                        filtered_df.groupby("ProductName", as_index=False)["UnitPrice"]
                        .mean()
                        .rename(columns={"ProductName": reco_product_col, "UnitPrice": "AvgUnitPrice"})
                    )
                    reco_df = reco_df.merge(avg_price_df, on=reco_product_col, how="left")
                else:
                    reco_df["AvgUnitPrice"] = 1000

                reco_df["EligibleCustomers"] = np.maximum((reco_df[score_col] / 10).round(), 5)

                reco_df["EstimatedRevenueUplift"] = (
                    reco_df[score_col].fillna(0)
                    * reco_df["AvgUnitPrice"].fillna(1000)
                    * reco_df["EligibleCustomers"].fillna(5)
                    * 0.08
                )

                reco_df["RecommendedAction"] = "Launch targeted cross-sell campaign"

                reco_df = reco_df.sort_values("EstimatedRevenueUplift", ascending=False).head(5)

                col1, col2 = st.columns([1.2, 1])

                with col1:
                    fig = px.bar(
                        reco_df,
                        x=reco_product_col,
                        y="EstimatedRevenueUplift",
                        color=reco_product_col,
                        title="Cross-Sell Revenue Opportunity",
                        labels={
                            reco_product_col: "Recommended Product",
                            "EstimatedRevenueUplift": "Estimated Revenue Uplift (USD)"
                        },
                        text_auto=".2s",
                    )
                    fig.update_layout(height=420, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.dataframe(
                        reco_df[
                            [
                                base_product_col,
                                reco_product_col,
                                score_col,
                                "AvgUnitPrice",
                                "EligibleCustomers",
                                "EstimatedRevenueUplift",
                                "RecommendedAction",
                            ]
                        ],
                        use_container_width=True,
                    )

                uplift_total = reco_df["EstimatedRevenueUplift"].sum()
                st.info(
                    f"For the selected base product, the current recommendation logic indicates an estimated cross-sell opportunity of approximately {fmt_currency(uplift_total)}."
                )
            else:
                st.warning("Recommendation file exists but required columns are missing.")
        else:
            st.warning("product_recommendations.csv is not available.")