import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load sales data
folder = Path(__file__).parent
df = pd.read_csv(folder / "data" / "monthly_sales.csv")

df["Month"] = pd.to_datetime(df["Month"])
sales = df["Total Sales"].to_numpy()


# Create model features (same as Java model)
def get_features(month_index):
    x = np.zeros(13)

    x[0] = 1
    x[1] = month_index / 12.0

    month = ((month_index - 1) % 12) + 1

    if month >= 2:
        x[month] = 1

    return x


# Train using first 36 months
X_train = np.array([
    get_features(i) for i in range(1, 37)
])

y_train = sales[:36]

coefficients = np.linalg.lstsq(
    X_train, y_train, rcond=None
)[0]


# Predict months 37 to 48
X_test = np.array([
    get_features(i) for i in range(37, 49)
])

predictions = X_test @ coefficients
actual = sales[36:48]


# Evaluate model
mae = np.mean(np.abs(actual - predictions))
rmse = np.sqrt(np.mean((actual - predictions) ** 2))

print("MODEL EVALUATION")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


# GRAPH 1: Actual vs Predicted Sales
plt.figure(figsize=(12, 6))

plt.plot(
    df["Month"].iloc[36:48],
    actual,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    df["Month"].iloc[36:48],
    predictions,
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig(folder / "actual_vs_predicted.png")
plt.show()


# Predict next 6 months
X_future = np.array([
    get_features(i) for i in range(49, 55)
])

future_sales = X_future @ coefficients

future_dates = pd.date_range(
    start=df["Month"].iloc[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)


# Display future predictions
print("\nFUTURE SALES FORECAST")

for date, value in zip(future_dates, future_sales):
    print(date.strftime("%Y-%m"), ":", round(value, 2))


# GRAPH 2: Future Sales Forecast
plt.figure(figsize=(12, 6))

plt.plot(
    df["Month"],
    sales,
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_sales,
    marker="o",
    linestyle="--",
    label="Forecasted Sales"
)

plt.title("Future Sales Forecast - Next 6 Months")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig(folder / "future_sales_forecast.png")
plt.show()

print("\nGRAPHS CREATED SUCCESSFULLY!")
