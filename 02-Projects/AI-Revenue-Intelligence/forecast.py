# ============================================================
# PROJECT:
# Enterprise AI Revenue Intelligence Platform
#
# FILE:
# forecast.py
#
# PURPOSE:
# Build a simple time-series forecasting model for revenue
#
# MODEL TYPE:
# Linear Regression (intro ML model for forecasting)
#
# BUSINESS VALUE:
# - Predict future revenue trends
# - Support planning and budgeting
# - Identify growth patterns
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("data/sales_transactions.csv")

df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["YearMonth"] = df["OrderDate"].dt.to_period("M")


# ============================================================
# STEP 2: AGGREGATE MONTHLY REVENUE
# ============================================================

monthly = df.groupby("YearMonth")["Revenue"].sum().reset_index()

# Convert YearMonth to numeric index for ML model
monthly["MonthIndex"] = np.arange(len(monthly))


# ============================================================
# STEP 3: TRAIN MODEL
# ============================================================

X = monthly[["MonthIndex"]]
y = monthly["Revenue"]

model = LinearRegression()
model.fit(X, y)


# ============================================================
# STEP 4: FORECAST NEXT 6 MONTHS
# ============================================================

future_months = 6

future_index = np.arange(len(monthly), len(monthly) + future_months)
future_index = future_index.reshape(-1, 1)

forecast = model.predict(future_index)


# ============================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================

plt.figure(figsize=(10, 5))

# Actual
plt.plot(monthly["MonthIndex"], y, label="Actual Revenue")

# Forecast
plt.plot(future_index, forecast, label="Forecast Revenue", linestyle="--")

plt.title("Revenue Forecast (Next 6 Months)")
plt.xlabel("Time")
plt.ylabel("Revenue")
plt.legend()

plt.show()


# ============================================================
# STEP 6: PRINT FORECAST VALUES
# ============================================================

print("===== FORECAST NEXT 6 MONTHS =====")

for i, value in enumerate(forecast):
    print(f"Month {i+1}: {value:,.2f}")