# ============================================================
# PROJECT:
# Enterprise AI Revenue Intelligence Platform
#
# FILE:
# recommender_system.py
#
# PURPOSE:
# Build a simple product recommender system based on customer
# purchase behavior.
#
# BUSINESS VALUE:
# This model identifies which products are frequently purchased
# by the same customers. It supports:
# - cross-sell strategy
# - upsell opportunities
# - account planning
# - CRM recommendations
#
# MODEL APPROACH:
# Co-occurrence recommendation logic
#
# HOW IT WORKS:
# 1. Load transaction data
# 2. Build a customer-product purchase matrix
# 3. Identify which products are commonly bought together
# 4. Create top product-to-product recommendations
# 5. Save recommendation output for dashboard use
# ============================================================

from __future__ import annotations

import pandas as pd


# ============================================================
# STEP 1: LOAD DATA
#
# PURPOSE:
# Load the transaction-level sales dataset used in the AI
# Revenue Intelligence project.
# ============================================================
def load_data() -> pd.DataFrame:
    """
    Load transaction dataset.

    Returns
    -------
    pd.DataFrame
        Transaction-level sales data.
    """
    df = pd.read_csv("data/sales_transactions.csv")
    return df


# ============================================================
# STEP 2: BUILD CUSTOMER-PRODUCT MATRIX
#
# PURPOSE:
# Create a binary matrix showing whether each customer has bought
# each product.
#
# EXAMPLE:
# Customer A -> bought Product X = 1
# Customer A -> did not buy Product Y = 0
# ============================================================
def build_customer_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a customer-product purchase matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level dataset.

    Returns
    -------
    pd.DataFrame
        Binary customer-product matrix.
    """
    customer_product = (
        df.groupby(["CustomerID", "ProductName"])
        .size()
        .unstack(fill_value=0)
    )

    # Convert counts into binary values:
    # 1 means product purchased at least once
    # 0 means never purchased
    customer_product = (customer_product > 0).astype(int)

    return customer_product


# ============================================================
# STEP 3: BUILD PRODUCT CO-OCCURRENCE MATRIX
#
# PURPOSE:
# Measure how often products are purchased by the same customer.
#
# BUSINESS INTERPRETATION:
# If two products are frequently bought by the same customers,
# they may be good cross-sell candidates.
# ============================================================
def build_product_cooccurrence_matrix(customer_product: pd.DataFrame) -> pd.DataFrame:
    """
    Build product co-occurrence matrix.

    Parameters
    ----------
    customer_product : pd.DataFrame
        Binary customer-product matrix.

    Returns
    -------
    pd.DataFrame
        Product-to-product co-occurrence matrix.
    """
    cooccurrence_matrix = customer_product.T.dot(customer_product)

    return cooccurrence_matrix


# ============================================================
# STEP 4: GENERATE PRODUCT RECOMMENDATIONS
#
# PURPOSE:
# For each product, identify the top related products based on
# co-occurrence scores.
#
# IMPORTANT:
# The product itself will have the highest score, so we remove
# self-recommendation.
# ============================================================
def generate_recommendations(
    cooccurrence_matrix: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Generate top product recommendations for each product.

    Parameters
    ----------
    cooccurrence_matrix : pd.DataFrame
        Product-to-product co-occurrence matrix.
    top_n : int
        Number of recommendations to keep per product.

    Returns
    -------
    pd.DataFrame
        Product recommendation table.
    """
    recommendations = []

    for product in cooccurrence_matrix.index:
        product_scores = cooccurrence_matrix.loc[product].copy()

        # Remove self-match
        product_scores = product_scores.drop(labels=[product])

        # Sort from strongest to weakest relationship
        top_related = product_scores.sort_values(ascending=False).head(top_n)

        for related_product, score in top_related.items():
            recommendations.append(
                {
                    "BaseProduct": product,
                    "RecommendedProduct": related_product,
                    "CoPurchaseScore": int(score),
                }
            )

    recommendations_df = pd.DataFrame(recommendations)

    return recommendations_df


# ============================================================
# STEP 5: PRINT SAMPLE RESULTS
#
# PURPOSE:
# Display a preview so the user can inspect whether the
# recommendations make business sense.
# ============================================================
def print_recommendation_results(recommendations_df: pd.DataFrame) -> None:
    """
    Print recommendation results to the terminal.
    """
    print("\n" + "=" * 70)
    print("PRODUCT RECOMMENDATION RESULTS")
    print("=" * 70)

    print("\nSample recommendations:")
    print(recommendations_df.head(20))


# ============================================================
# STEP 6: SAVE OUTPUT
#
# PURPOSE:
# Save recommendation results into CSV so they can be loaded into
# the Streamlit dashboard later.
# ============================================================
def save_output(recommendations_df: pd.DataFrame) -> None:
    """
    Save recommendations to CSV file.
    """
    recommendations_df.to_csv("data/product_recommendations.csv", index=False)
    print("\nSaved file: data/product_recommendations.csv")


# ============================================================
# STEP 7: MAIN PROGRAM FLOW
# ============================================================
def main() -> None:
    """
    Main flow for recommender system execution.
    """
    df = load_data()
    customer_product = build_customer_product_matrix(df)
    cooccurrence_matrix = build_product_cooccurrence_matrix(customer_product)
    recommendations_df = generate_recommendations(cooccurrence_matrix, top_n=3)

    print_recommendation_results(recommendations_df)
    save_output(recommendations_df)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()