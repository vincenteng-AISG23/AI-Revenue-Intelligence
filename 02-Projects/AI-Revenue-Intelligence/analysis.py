# ============================================================
# PROJECT:
# Enterprise AI Revenue Intelligence Platform
#
# FILE:
# analysis.py
#
# PURPOSE:
# Perform executive-level analysis on sales data
# to generate business insights for CIO / CEO / CFO
#
# THIS SCRIPT ANSWERS:
# - Revenue by Business Line
# - Revenue by Region
# - Top Customers
# - Product Profitability
#
# OUTPUT:
# Console-based insights (later we upgrade to dashboard)
# ============================================================

import pandas as pd


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

def load_data():
    """
    Load the generated dataset from CSV file
    """
    df = pd.read_csv("data/sales_transactions.csv")
    print("✅ Data loaded successfully\n")
    return df


# ============================================================
# STEP 2: BASIC OVERVIEW
# ============================================================

def data_overview(df):
    """
    Display basic dataset information
    """
    print("===== DATA OVERVIEW =====")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nTotal Rows:", len(df))
    print("\n")


# ============================================================
# STEP 3: REVENUE BY BUSINESS LINE
# ============================================================

def revenue_by_business(df):
    """
    Analyze revenue contribution by business line
    """
    print("===== REVENUE BY BUSINESS LINE =====")

    result = df.groupby("BusinessLine")["Revenue"].sum().sort_values(ascending=False)

    print(result)
    print("\n")


# ============================================================
# STEP 4: REVENUE BY REGION
# ============================================================

def revenue_by_region(df):
    """
    Analyze which regions generate most revenue
    """
    print("===== REVENUE BY REGION =====")

    result = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)

    print(result)
    print("\n")


# ============================================================
# STEP 5: TOP CUSTOMERS
# ============================================================

def top_customers(df):
    """
    Identify top revenue generating customers
    """
    print("===== TOP 10 CUSTOMERS =====")

    result = df.groupby("CustomerName")["Revenue"].sum().sort_values(ascending=False).head(10)

    print(result)
    print("\n")


# ============================================================
# STEP 6: PRODUCT PROFITABILITY
# ============================================================

def product_profitability(df):
    """
    Identify most profitable products
    """
    print("===== PRODUCT PROFITABILITY =====")

    result = df.groupby("ProductName")["Profit"].sum().sort_values(ascending=False)

    print(result)
    print("\n")


# ============================================================
# STEP 7: MAIN EXECUTION
# ============================================================

def main():
    df = load_data()

    data_overview(df)

    revenue_by_business(df)
    revenue_by_region(df)
    top_customers(df)
    product_profitability(df)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()