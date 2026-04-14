# ============================================================
# PROJECT:
# Enterprise AI Revenue Intelligence Platform
#
# FILE:
# customer_segmentation.py
#
# PURPOSE:
# Build an AI-based customer segmentation model using K-Means
# clustering.
#
# BUSINESS VALUE:
# This script helps management understand different types of
# customers based on commercial behavior, so that sales and CRM
# teams can prioritize actions more effectively.
#
# WHAT THIS SCRIPT DOES:
# 1. Load transaction data
# 2. Aggregate customer-level commercial metrics
# 3. Prepare features for clustering
# 4. Standardize the data
# 5. Train a K-Means clustering model
# 6. Assign customer segments
# 7. Print summary output
#
# IMPORTANT LEARNING POINT:
# Clustering is unsupervised learning. That means we do not tell
# the model what label to predict. Instead, the model finds
# hidden patterns in customer behavior.
# ============================================================

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 1: LOAD DATA
#
# PURPOSE:
# Load the transaction dataset used in the revenue intelligence
# project.
# ============================================================
def load_data() -> pd.DataFrame:
    """
    Load the transaction-level dataset.

    Returns
    -------
    pd.DataFrame
        Raw transaction data with OrderDate converted to datetime.
    """
    df = pd.read_csv("data/sales_transactions.csv")
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    return df


# ============================================================
# STEP 2: BUILD CUSTOMER-LEVEL FEATURE TABLE
#
# PURPOSE:
# Aggregate transaction data into customer-level behavior
# metrics suitable for segmentation.
#
# FEATURES CREATED:
# - TotalRevenue
# - TotalProfit
# - TotalOrders
# - AvgOrderValue
# - LastPurchaseDaysAgo
# ============================================================
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a customer-level feature table for segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level dataset.

    Returns
    -------
    pd.DataFrame
        Customer-level feature table.
    """
    latest_date = df["OrderDate"].max()

    customer_features = (
        df.groupby(["CustomerID", "CustomerName"], as_index=False)
        .agg(
            TotalRevenue=("Revenue", "sum"),
            TotalProfit=("Profit", "sum"),
            TotalOrders=("OrderID", "nunique"),
            AvgOrderValue=("Revenue", "mean"),
            LastPurchaseDate=("OrderDate", "max"),
        )
    )

    customer_features["LastPurchaseDaysAgo"] = (
        latest_date - customer_features["LastPurchaseDate"]
    ).dt.days

    return customer_features


# ============================================================
# STEP 3: RUN K-MEANS CLUSTERING
#
# PURPOSE:
# Use standardized numeric features to group customers into
# behavioral segments.
#
# WHY STANDARDIZATION MATTERS:
# Revenue and profit can be much larger numbers than order count
# or recency. StandardScaler ensures that one feature does not
# dominate the clustering just because of scale.
# ============================================================
def run_customer_segmentation(customer_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Train a K-Means clustering model and assign customer segments.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level feature table.
    n_clusters : int
        Number of clusters to create.

    Returns
    -------
    pd.DataFrame
        Customer feature table with assigned cluster labels.
    """
    feature_columns = [
        "TotalRevenue",
        "TotalProfit",
        "TotalOrders",
        "AvgOrderValue",
        "LastPurchaseDaysAgo",
    ]

    X = customer_df[feature_columns].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_df["Cluster"] = model.fit_predict(X_scaled)

    return customer_df


# ============================================================
# STEP 4: ASSIGN BUSINESS-FRIENDLY SEGMENT NAMES
#
# PURPOSE:
# Cluster numbers like 0, 1, 2, 3 are not useful for executives.
# We will translate clusters into business-friendly labels based
# on average customer behavior.
# ============================================================
def assign_segment_names(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric cluster labels into business-friendly segment
    names using commercial logic.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer feature table with cluster assignments.

    Returns
    -------
    pd.DataFrame
        Customer table with SegmentName column added.
    """
    cluster_summary = (
        customer_df.groupby("Cluster", as_index=False)
        .agg(
            AvgRevenue=("TotalRevenue", "mean"),
            AvgProfit=("TotalProfit", "mean"),
            AvgOrders=("TotalOrders", "mean"),
            AvgRecency=("LastPurchaseDaysAgo", "mean"),
        )
    )

    segment_name_map = {}

    for _, row in cluster_summary.iterrows():
        cluster_id = row["Cluster"]
        avg_revenue = row["AvgRevenue"]
        avg_profit = row["AvgProfit"]
        avg_orders = row["AvgOrders"]
        avg_recency = row["AvgRecency"]

        if avg_revenue > customer_df["TotalRevenue"].median() and avg_profit > customer_df["TotalProfit"].median():
            if avg_recency <= customer_df["LastPurchaseDaysAgo"].median():
                segment_name_map[cluster_id] = "High Value Active"
            else:
                segment_name_map[cluster_id] = "High Value At Risk"
        elif avg_orders > customer_df["TotalOrders"].median():
            segment_name_map[cluster_id] = "Frequent Buyers"
        else:
            segment_name_map[cluster_id] = "Low Engagement"

    customer_df["SegmentName"] = customer_df["Cluster"].map(segment_name_map)

    return customer_df


# ============================================================
# STEP 5: PRINT RESULTS
#
# PURPOSE:
# Show the final segmentation output and management summary.
# ============================================================
def print_segmentation_results(customer_df: pd.DataFrame) -> None:
    """
    Print segmentation results to console.
    """
    print("\n" + "=" * 70)
    print("CUSTOMER SEGMENTATION RESULTS")
    print("=" * 70)

    print("\nSegment distribution:")
    print(customer_df["SegmentName"].value_counts())

    print("\nCustomer sample preview:")
    print(
        customer_df[
            [
                "CustomerID",
                "CustomerName",
                "TotalRevenue",
                "TotalProfit",
                "TotalOrders",
                "AvgOrderValue",
                "LastPurchaseDaysAgo",
                "Cluster",
                "SegmentName",
            ]
        ]
        .sort_values(by="TotalRevenue", ascending=False)
        .head(15)
    )

    print("\nSegment summary:")
    segment_summary = (
        customer_df.groupby("SegmentName", as_index=False)
        .agg(
            Customers=("CustomerID", "count"),
            AvgRevenue=("TotalRevenue", "mean"),
            AvgProfit=("TotalProfit", "mean"),
            AvgOrders=("TotalOrders", "mean"),
            AvgRecency=("LastPurchaseDaysAgo", "mean"),
        )
        .sort_values(by="AvgRevenue", ascending=False)
    )
    print(segment_summary)


# ============================================================
# STEP 6: SAVE OUTPUT
#
# PURPOSE:
# Save the segmented customer table so it can later be used in
# the Streamlit dashboard.
# ============================================================
def save_output(customer_df: pd.DataFrame) -> None:
    """
    Save segmented customer data to CSV file.
    """
    customer_df.to_csv("data/customer_segments.csv", index=False)
    print("\nSaved file: data/customer_segments.csv")


# ============================================================
# STEP 7: MAIN PROGRAM FLOW
# ============================================================
def main() -> None:
    """
    Main execution flow for customer segmentation.
    """
    df = load_data()
    customer_df = build_customer_features(df)
    customer_df = run_customer_segmentation(customer_df, n_clusters=4)
    customer_df = assign_segment_names(customer_df)
    print_segmentation_results(customer_df)
    save_output(customer_df)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()