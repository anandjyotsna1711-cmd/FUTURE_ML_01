
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SalesVision | Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    [data-testid="stMetric"] {
        background-color: #1c2230;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #303849;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
folder = Path(__file__).parent
file_path = folder / "data" / "monthly_sales.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month").reset_index(drop=True)
    return df

df = load_data()
sales = df["Total Sales"].to_numpy(dtype=float)

# ---------------- MODEL FEATURES ----------------
def get_features(month_index):
    x = np.zeros(13)
    x[0] = 1
    x[1] = month_index / 12.0

    month = ((month_index - 1) % 12) + 1
    if month >= 2:
        x[month] = 1

    return x

def get_baseline_features(month_index):
    month = 2 * np.pi * ((month_index - 1) % 12) / 12
    return np.array([
        1,
        month_index / 12.0,
        np.sin(month),
        np.cos(month)
    ])

# ---------------- MODEL EVALUATION ----------------
X_train = np.array([get_features(i) for i in range(1, 37)])
y_train = sales[:36]

X_test = np.array([get_features(i) for i in range(37, 49)])
actual = sales[36:48]

coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
predictions = X_test @ coefficients

mae = np.mean(np.abs(actual - predictions))
rmse = np.sqrt(np.mean((actual - predictions) ** 2))

# Baseline model for comparison
baseline_train = np.array([
    get_baseline_features(i) for i in range(1, 37)
])
baseline_test = np.array([
    get_baseline_features(i) for i in range(37, 49)
])

baseline_coefficients = np.linalg.lstsq(
    baseline_train, y_train, rcond=None
)[0]

baseline_predictions = baseline_test @ baseline_coefficients

baseline_mae = np.mean(np.abs(actual - baseline_predictions))
baseline_rmse = np.sqrt(np.mean((actual - baseline_predictions) ** 2))

# ---------------- FUTURE FORECAST ----------------
full_X = np.array([
    get_features(i) for i in range(1, len(sales) + 1)
])

full_coefficients = np.linalg.lstsq(
    full_X, sales, rcond=None
)[0]

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

horizon = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=3,
    max_value=12,
    value=6
)

st.sidebar.markdown("---")
st.sidebar.caption("SalesVision | Machine Learning Dashboard")
st.sidebar.caption("Java + Python + Streamlit")

# ---------------- HEADER ----------------
st.title("📊 SalesVision")
st.markdown(
    "### Retail Sales Forecasting & Analytics Engine"
)
st.caption(
    "An interactive machine learning dashboard for historical "
    "sales analysis, model evaluation and future forecasting."
)

st.markdown("---")


# ---------------- PROFESSIONAL KPI CARDS ----------------

# Calculate next 6 months of forecast revenue
kpi_indices = np.arange(49, 55)
kpi_X = np.array([get_features(i) for i in kpi_indices])
kpi_forecast = kpi_X @ full_coefficients

forecast_revenue = kpi_forecast.sum()
average_forecast = kpi_forecast.mean()

# Dashboard heading
st.markdown("## Business Performance Overview")
st.caption("Key indicators based on historical sales and machine learning predictions.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Historical Revenue",
        value=f"${sales.sum():,.0f}",
        help="Total sales across all 48 historical months."
    )

with col2:
    st.metric(
        label="Average Monthly Revenue",
        value=f"${sales.mean():,.0f}",
        help="Average monthly sales across the dataset."
    )

with col3:
    st.metric(
        label="6-Month Forecast Revenue",
        value=f"${forecast_revenue:,.0f}",
        help="Total predicted sales for the next six months."
    )

with col4:
    st.metric(
        label="Model MAE",
        value=f"${mae:,.2f}",
        help="Mean Absolute Error on the 12-month test set. Lower is better."
    )

st.markdown("---")

# ---------------- TABS ----------------
overview, diagnostics, forecast_tab = st.tabs([
    "📈 Sales Overview",
    "🤖 Model Diagnostics",
    "🔮 Future Forecast"
])

# ---------------- OVERVIEW ----------------
with overview:
    st.subheader("Historical Monthly Sales")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=sales,
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="#38bdf8", width=3)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=430,
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Sales Data")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

# ---------------- DIAGNOSTICS ----------------
with diagnostics:
    st.subheader("Actual vs Predicted Sales")

    test_dates = df["Month"].iloc[36:48]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=actual,
        name="Actual Sales",
        mode="lines+markers"
    ))

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=predictions,
        name="Predicted Sales",
        mode="lines+markers"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Seasonal Regression",
            "Month-Based Regression"
        ],
        "MAE": [
            baseline_mae,
            mae
        ],
        "RMSE": [
            baseline_rmse,
            rmse
        ]
    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        hide_index=True
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison["Model"],
        y=comparison["MAE"],
        name="MAE"
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Model MAE Comparison (Lower is Better)",
        yaxis_title="MAE",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FUTURE FORECAST ----------------
with forecast_tab:
    st.subheader(f"Next {horizon} Months Forecast")

    future_indices = np.arange(
        len(sales) + 1,
        len(sales) + horizon + 1
    )

    future_X = np.array([
        get_features(i) for i in future_indices
    ])

    future_sales = future_X @ full_coefficients

    future_dates = pd.date_range(
        start=df["Month"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Predicted Sales": np.round(future_sales, 2)
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=sales,
        name="Historical Sales",
        mode="lines",
        line=dict(color="#38bdf8", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_sales,
        name="Forecasted Sales",
        mode="lines+markers",
        line=dict(color="#a78bfa", width=3, dash="dash")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        title="Historical Sales and Future Forecast",
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Schedule")

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Total Predicted Sales",
        f"${future_sales.sum():,.2f}"
    )

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Forecast CSV",
        data=csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "SalesVision | Built with Python, Streamlit, NumPy, Pandas and Plotly"
)
