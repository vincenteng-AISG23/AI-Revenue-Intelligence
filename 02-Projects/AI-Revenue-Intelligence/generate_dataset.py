# ============================================================
# PROJECT:
# Enterprise AI Revenue Intelligence Platform
#
# FILE:
# generate_dataset.py
#
# PURPOSE:
# Generate a realistic synthetic sales transaction dataset for
# an advanced manufacturing business with 3 business lines:
#
# 1. Medical equipment for hospital operating theatres
# 2. PCB / control / sensor parts for autonomous cars and robots
# 3. Electronic and mechanical parts for smart home devices,
#    drones, and other connected products
#
# WHY THIS FILE MATTERS:
# A strong AI project starts with well-structured data.
# This dataset is designed to support:
# - Revenue analysis and executive reporting
# - Sales forecasting
# - CRM intelligence and customer segmentation
# - Product recommendation logic
#
# OUTPUT:
# data/sales_transactions.csv
#
# DESIGN PRINCIPLES:
# - Industry-relevant structure
# - Business-friendly fields
# - Clear code for new programmers
# - Reusable for multiple AI/ML use cases
# ============================================================

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# SECTION 1: REPRODUCIBILITY SETTINGS
#
# PURPOSE:
# Fix the random seed so the script generates the same style of
# results each time. This helps with testing, debugging, and
# demonstration consistency.
# ============================================================
random.seed(42)
np.random.seed(42)


# ============================================================
# SECTION 2: MASTER BUSINESS REFERENCE DATA
#
# PURPOSE:
# Define the core business reference lists used across the
# dataset generation process.
#
# THESE INCLUDE:
# - Regions
# - Sales channels
# - Customer segments
# - Industries
# - Product catalog across 3 manufacturing business lines
# ============================================================

REGIONS = [
    "Singapore",
    "Malaysia",
    "Thailand",
    "China",
    "Indonesia",
]

SALES_CHANNELS = [
    "Direct Sales",
    "Distributor",
    "Online",
    "Retail",
]

CUSTOMER_SEGMENTS = [
    "Enterprise",
    "SME",
    "Retail",
]

INDUSTRIES = [
    "Hospital Systems",
    "Medical Distributors",
    "Automotive OEM",
    "Robotics",
    "Smart Home",
    "Consumer Electronics",
    "Industrial Automation",
    "Technology",
]

PRODUCT_CATALOG = [
    # --------------------------------------------------------
    # BUSINESS LINE 1: MEDICAL EQUIPMENT
    # These are high-value products typically sold to hospital
    # groups, healthcare distributors, and institutional buyers.
    # --------------------------------------------------------
    {
        "ProductID": "M001",
        "ProductName": "Surgical Light System",
        "Category": "Medical Equipment",
        "BusinessLine": "Medical",
        "UnitPrice": 12000,
        "BaseCost": 7000,
    },
    {
        "ProductID": "M002",
        "ProductName": "Operating Table",
        "Category": "Medical Equipment",
        "BusinessLine": "Medical",
        "UnitPrice": 18000,
        "BaseCost": 11000,
    },
    {
        "ProductID": "M003",
        "ProductName": "Anesthesia Machine",
        "Category": "Medical Equipment",
        "BusinessLine": "Medical",
        "UnitPrice": 25000,
        "BaseCost": 15000,
    },
    {
        "ProductID": "M004",
        "ProductName": "Patient Monitoring Module",
        "Category": "Medical Equipment",
        "BusinessLine": "Medical",
        "UnitPrice": 8500,
        "BaseCost": 5000,
    },

    # --------------------------------------------------------
    # BUSINESS LINE 2: AUTONOMOUS / ROBOTICS COMPONENTS
    # These are niche electronic parts and sub-systems used in
    # autonomous vehicles, robotics, and advanced automation.
    # --------------------------------------------------------
    {
        "ProductID": "A001",
        "ProductName": "Advanced PCB Board",
        "Category": "Electronics",
        "BusinessLine": "Autonomous",
        "UnitPrice": 450,
        "BaseCost": 250,
    },
    {
        "ProductID": "A002",
        "ProductName": "Control Unit Module",
        "Category": "Electronics",
        "BusinessLine": "Autonomous",
        "UnitPrice": 780,
        "BaseCost": 420,
    },
    {
        "ProductID": "A003",
        "ProductName": "Sensor Processing Unit",
        "Category": "Electronics",
        "BusinessLine": "Autonomous",
        "UnitPrice": 620,
        "BaseCost": 350,
    },
    {
        "ProductID": "A004",
        "ProductName": "Robotics Motion Controller",
        "Category": "Electronics",
        "BusinessLine": "Autonomous",
        "UnitPrice": 950,
        "BaseCost": 540,
    },

    # --------------------------------------------------------
    # BUSINESS LINE 3: SMART DEVICES / DRONES / CONNECTED PARTS
    # These are lower-priced, higher-volume components for
    # smart homes, drones, and consumer / connected devices.
    # --------------------------------------------------------
    {
        "ProductID": "S001",
        "ProductName": "Smart Home Controller",
        "Category": "IoT Devices",
        "BusinessLine": "Smart Devices",
        "UnitPrice": 180,
        "BaseCost": 90,
    },
    {
        "ProductID": "S002",
        "ProductName": "Drone Navigation Module",
        "Category": "IoT Devices",
        "BusinessLine": "Smart Devices",
        "UnitPrice": 320,
        "BaseCost": 170,
    },
    {
        "ProductID": "S003",
        "ProductName": "Wireless Sensor Node",
        "Category": "IoT Devices",
        "BusinessLine": "Smart Devices",
        "UnitPrice": 95,
        "BaseCost": 45,
    },
    {
        "ProductID": "S004",
        "ProductName": "Smart Lock Control Board",
        "Category": "IoT Devices",
        "BusinessLine": "Smart Devices",
        "UnitPrice": 140,
        "BaseCost": 70,
    },
]


# ============================================================
# SECTION 3: CUSTOMER MASTER GENERATION
#
# PURPOSE:
# Create a synthetic customer master table with realistic
# business attributes.
#
# WHY THIS MATTERS:
# Customer-level attributes are important for:
# - CRM analytics
# - customer segmentation
# - churn / inactivity logic
# - revenue analysis by segment and industry
# ============================================================
def generate_customers(num_customers: int = 220) -> pd.DataFrame:
    """
    Generate a synthetic customer master dataset.

    Parameters
    ----------
    num_customers : int
        Number of customer records to generate.

    Returns
    -------
    pd.DataFrame
        Customer master data with business attributes.
    """
    customer_rows: list[dict] = []

    for i in range(1, num_customers + 1):
        customer_id = f"C{i:04d}"
        customer_name = f"Customer {i:03d}"

        region = random.choice(REGIONS)

        customer_segment = random.choices(
            population=CUSTOMER_SEGMENTS,
            weights=[0.35, 0.40, 0.25],
            k=1,
        )[0]

        industry = random.choice(INDUSTRIES)

        customer_tenure_months = random.randint(1, 96)
        last_purchase_days_ago = random.randint(1, 180)

        customer_rows.append(
            {
                "CustomerID": customer_id,
                "CustomerName": customer_name,
                "Region": region,
                "CustomerSegment": customer_segment,
                "Industry": industry,
                "CustomerTenureMonths": customer_tenure_months,
                "LastPurchaseDaysAgo": last_purchase_days_ago,
            }
        )

    return pd.DataFrame(customer_rows)


# ============================================================
# SECTION 4: DATE HELPER
#
# PURPOSE:
# Generate a random order date between a chosen start and end
# date. Time-based data is required for forecasting and trend
# analysis.
# ============================================================
def random_order_date(start_date: datetime, end_date: datetime) -> datetime:
    """
    Generate a random date between two datetime boundaries.

    Parameters
    ----------
    start_date : datetime
        Earliest possible order date.
    end_date : datetime
        Latest possible order date.

    Returns
    -------
    datetime
        A random date within the chosen range.
    """
    number_of_days = (end_date - start_date).days
    random_days = random.randint(0, number_of_days)
    return start_date + timedelta(days=random_days)


# ============================================================
# SECTION 5: BUSINESS RULE HELPERS
#
# PURPOSE:
# Add more realistic commercial behavior into the dataset.
#
# EXAMPLES:
# - Medical equipment usually sells in lower quantities
# - Smart devices usually sell in higher quantities
# - Enterprise customers may get larger discounts
# - Distributors may also receive discount treatment
# ============================================================
def determine_quantity(business_line: str) -> int:
    """
    Determine a realistic order quantity based on business line.

    Parameters
    ----------
    business_line : str
        Product business line.

    Returns
    -------
    int
        Simulated quantity purchased.
    """
    if business_line == "Medical":
        return int(np.random.poisson(lam=1) + 1)
    if business_line == "Autonomous":
        return int(np.random.poisson(lam=4) + 1)
    return int(np.random.poisson(lam=7) + 1)


def determine_discount(customer_segment: str, sales_channel: str, business_line: str) -> float:
    """
    Determine a realistic discount rate using basic business logic.

    Parameters
    ----------
    customer_segment : str
        Customer type such as Enterprise, SME, or Retail.
    sales_channel : str
        Sales route used for the order.
    business_line : str
        Product business line.

    Returns
    -------
    float
        Discount percentage represented as a decimal.
        Example: 0.10 means 10%.
    """
    if business_line == "Medical":
        if customer_segment == "Enterprise":
            return round(random.uniform(0.08, 0.18), 2)
        return round(random.uniform(0.03, 0.10), 2)

    if business_line == "Autonomous":
        if sales_channel == "Distributor":
            return round(random.uniform(0.05, 0.15), 2)
        return round(random.uniform(0.02, 0.10), 2)

    if customer_segment == "Retail":
        return round(random.uniform(0.00, 0.08), 2)

    return round(random.uniform(0.03, 0.12), 2)


# ============================================================
# SECTION 6: SALES TRANSACTION GENERATION
#
# PURPOSE:
# Create transaction-level sales records that combine:
# - customer information
# - product information
# - commercial logic
# - financial calculations
#
# OUTPUT FIELDS SUPPORT:
# - executive reporting
# - forecasting
# - CRM analysis
# - recommendation logic
# ============================================================
def generate_sales_transactions(
    customers_df: pd.DataFrame,
    num_transactions: int = 5000,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Generate synthetic sales transactions.

    Parameters
    ----------
    customers_df : pd.DataFrame
        Customer master data.
    num_transactions : int
        Number of transaction lines to generate.
    start_date : str
        Start of the transaction period in YYYY-MM-DD format.
    end_date : str
        End of the transaction period in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Transaction-level dataset.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    transaction_rows: list[dict] = []

    for i in range(1, num_transactions + 1):
        order_id = f"O{i:06d}"
        order_date = random_order_date(start_dt, end_dt)

        customer = customers_df.sample(1).iloc[0]
        product = random.choice(PRODUCT_CATALOG)

        sales_channel = random.choice(SALES_CHANNELS)
        quantity = determine_quantity(product["BusinessLine"])
        discount = determine_discount(
            customer_segment=customer["CustomerSegment"],
            sales_channel=sales_channel,
            business_line=product["BusinessLine"],
        )

        unit_price = product["UnitPrice"]
        base_cost = product["BaseCost"]

        gross_revenue = unit_price * quantity
        net_revenue = gross_revenue * (1 - discount)
        total_cost = base_cost * quantity
        total_profit = net_revenue - total_cost

        transaction_rows.append(
            {
                "OrderID": order_id,
                "OrderDate": order_date.date(),
                "CustomerID": customer["CustomerID"],
                "CustomerName": customer["CustomerName"],
                "Region": customer["Region"],
                "SalesChannel": sales_channel,
                "ProductID": product["ProductID"],
                "ProductName": product["ProductName"],
                "Category": product["Category"],
                "BusinessLine": product["BusinessLine"],
                "UnitPrice": unit_price,
                "Quantity": quantity,
                "Discount": discount,
                "Revenue": round(net_revenue, 2),
                "Cost": round(total_cost, 2),
                "Profit": round(total_profit, 2),
                "CustomerTenureMonths": customer["CustomerTenureMonths"],
                "CustomerSegment": customer["CustomerSegment"],
                "Industry": customer["Industry"],
                "LastPurchaseDaysAgo": customer["LastPurchaseDaysAgo"],
            }
        )

    return pd.DataFrame(transaction_rows)


# ============================================================
# SECTION 7: MAIN PROGRAM EXECUTION
#
# PURPOSE:
# Create the output folder if needed, generate the customer
# master and transaction dataset, save the final CSV file,
# and print a preview for verification.
# ============================================================
def main() -> None:
    """
    Main entry point for dataset generation.
    """
    os.makedirs("data", exist_ok=True)

    customers_df = generate_customers(num_customers=220)

    sales_df = generate_sales_transactions(
        customers_df=customers_df,
        num_transactions=5000,
        start_date="2023-01-01",
        end_date="2025-12-31",
    )

    output_path = "data/sales_transactions.csv"
    sales_df.to_csv(output_path, index=False)

    print("=" * 60)
    print("Synthetic revenue intelligence dataset generated successfully.")
    print(f"Output file: {output_path}")
    print(f"Number of rows: {len(sales_df):,}")
    print(f"Number of columns: {len(sales_df.columns)}")
    print("=" * 60)
    print("\nSample data preview:")
    print(sales_df.head())


# ============================================================
# SECTION 8: SCRIPT ENTRY POINT
#
# PURPOSE:
# Run the main function only when this file is executed
# directly, not when imported by another file.
# ============================================================
if __name__ == "__main__":
    main()