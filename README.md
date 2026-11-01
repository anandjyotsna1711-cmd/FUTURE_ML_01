# Sales Forecasting System

## Project Overview
The Sales Forecasting System is a machine learning project developed to analyze historical monthly sales data and predict future sales trends.

It uses a Linear Regression model with time-based and seasonal features to forecast sales for the next six months.

## Technologies Used
- Java
- Python
- Apache Maven
- Apache Commons CSV
- NumPy
- Pandas
- Matplotlib

## Dataset
The project uses a monthly sales dataset containing 48 months of historical sales records from January 2014 to December 2017.

## Model Methodology
1. Load and process historical sales data.
2. Extract time-based and monthly seasonal features.
3. Train a Multiple Linear Regression model using the first 36 months.
4. Evaluate predictions against the remaining 12 months.
5. Forecast sales for the next six months.

## Model Evaluation

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 10,793.89 |
| Root Mean Squared Error (RMSE) | 14,557.85 |

The improved model reduced MAE by approximately 16.69% and RMSE by approximately 17.78% compared with the initial model on the same test period.

## Visualizations

The project generates two visualizations:

1. **Actual vs Predicted Sales** – compares actual sales with model predictions.
2. **Future Sales Forecast** – displays historical sales alongside the six-month forecast.

## Project Structure

```text
sales-forecasting/
│
├── data/
│   └── monthly_sales.csv
│
├── src/
│   └── main/
│       └── java/
│           └── com/future/
│               └── SalesForecaster.java
│
├── actual_vs_predicted.png
├── future_sales_forecast.png
├── visualize_sales.py
├── pom.xml
└── README.md
```

## Future Enhancements
- Experiment with advanced forecasting algorithms.
- Incorporate larger datasets.
- Improve model accuracy through cross-validation.
- Develop an interactive sales analytics dashboard.

## Conclusion
This project demonstrates how historical sales data can be used to identify trends, evaluate forecasting performance, and estimate future sales using machine learning.
