# SalesVision – Monthly Sales Forecasting System

## Overview
SalesVision is a machine learning-based sales forecasting project that analyzes historical monthly sales data, evaluates prediction performance, and forecasts future sales.

The project combines a Java-based forecasting model with an interactive Streamlit dashboard to make sales trends and predictions easier to understand.

## Features
- Historical monthly sales analysis
- Machine learning-based sales prediction
- Actual vs. predicted sales comparison
- Model evaluation using MAE and RMSE
- Six-month future sales forecasting
- Interactive dashboard with Plotly visualizations
- Downloadable forecast results

## Technologies Used
- **Java 17** – Forecasting model
- **Apache Maven** – Java project management
- **Python** – Data analysis and visualization
- **NumPy & Pandas** – Data processing and model calculations
- **Streamlit** – Interactive dashboard
- **Plotly** – Interactive charts
- **Apache Commons CSV** – CSV data handling

## Project Structure

```text
FUTURE_ML_01/
│
├── data/
│   └── monthly_sales.csv
│
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── future/
│                   ├── App.java
│                   └── SalesForecaster.java
│
├── app.py
├── visualize_sales.py
├── actual_vs_predicted.png
├── future_sales_forecast.png
├── pom.xml
└── README.md
```

## Dataset
The project uses
