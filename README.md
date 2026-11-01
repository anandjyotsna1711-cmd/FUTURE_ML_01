# SalesVision – Monthly Sales Forecasting System

## 1. Overview

SalesVision is a machine learning-based sales forecasting system designed to analyze historical monthly sales data, identify trends and seasonal patterns, evaluate prediction performance, and forecast future sales.

The project combines a Java-based forecasting model with Python data processing and an interactive Streamlit dashboard to present predictions in a clear, business-friendly format.

## 2. Project Objectives

- Analyze historical monthly sales data.
- Clean and prepare time-based data.
- Identify sales trends and seasonal patterns.
- Develop a regression-based forecasting model.
- Evaluate model performance using MAE and RMSE.
- Predict future monthly sales.
- Present forecasting results through visualizations and an interactive dashboard.

## 3. Features

- Historical monthly sales analysis.
- Data cleaning and missing-value handling.
- Time-based feature engineering.
- Regression-based sales forecasting.
- Baseline model comparison.
- Model evaluation using MAE and RMSE.
- Six-month future sales prediction.
- Actual vs. predicted sales visualization.
- Future sales forecast visualization.
- Interactive Streamlit dashboard.
- Business-friendly KPI cards and charts.
- Downloadable forecast results.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Java 17 | Forecasting model |
| Apache Maven | Java project management |
| Python | Data processing and visualization |
| NumPy | Numerical computations and regression |
| Pandas | Data cleaning and manipulation |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| Matplotlib | Data visualization |
| Apache Commons CSV | CSV data handling |

## 5. Project Structure

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

## 6. Dataset Description

The project uses monthly sales data containing 48 observations from January 2014 to December 2017.

The dataset contains the following columns:

- **Month:** Date representing the sales month.
- **Total Sales:** Total sales recorded for that month.

The first 36 months are used for model training, while the remaining 12 months are used for testing.

## 7. Data Preprocessing

The data preparation process includes:

- Loading the historical sales dataset.
- Converting dates into a consistent datetime format.
- Converting sales values into numeric format.
- Removing records with invalid dates.
- Sorting records chronologically.
- Checking for missing sales values.
- Handling missing sales values using time-based interpolation when required.

These steps help ensure that the dataset is suitable for time-based forecasting.

## 8. Feature Engineering and Model Development

The forecasting model uses regression with time-based and seasonal features.

The features include:

- Time trend to capture long-term sales movement.
- Monthly indicators to represent seasonal variations.
- A baseline seasonal regression model for comparison.

The model is trained on the first 36 months and
