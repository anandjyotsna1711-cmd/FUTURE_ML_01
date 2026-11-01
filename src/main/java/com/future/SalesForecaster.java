
package com.future;

import java.io.FileReader;
import java.io.Reader;
import java.time.YearMonth;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class SalesForecaster {

    static double[] coefficients;

    public static double[] getFeatures(double[] row) {

        double[] x = new double[13];

        x[0] = 1;
        x[1] = row[0] / 12.0;

        int month = ((int) row[0] - 1) % 12 + 1;

        if (month >= 2) {
            x[month] = 1;
        }

        return x;
    }

    public static void train(List<double[]> data) {

        int n = 36;
        int features = 13;

        double[][] matrix = new double[features][features + 1];

        for (int i = 0; i < n; i++) {

            double[] row = data.get(i);
            double[] x = getFeatures(row);
            double y = row[3];

            for (int j = 0; j < features; j++) {

                for (int k = 0; k < features; k++) {
                    matrix[j][k] += x[j] * x[k];
                }

                matrix[j][features] += x[j] * y;
            }
        }

        for (int i = 0; i < features; i++) {

            int pivot = i;

            for (int j = i + 1; j < features; j++) {
                if (Math.abs(matrix[j][i]) >
                    Math.abs(matrix[pivot][i])) {
                    pivot = j;
                }
            }

            double[] temp = matrix[i];
            matrix[i] = matrix[pivot];
            matrix[pivot] = temp;

            double divisor = matrix[i][i];

            if (Math.abs(divisor) < 1e-10) {
                throw new ArithmeticException("Matrix is singular");
            }

            for (int j = 0; j <= features; j++) {
                matrix[i][j] /= divisor;
            }

            for (int k = 0; k < features; k++) {

                if (k == i) continue;

                double factor = matrix[k][i];

                for (int j = 0; j <= features; j++) {
                    matrix[k][j] -= factor * matrix[i][j];
                }
            }
        }

        coefficients = new double[features];

        for (int i = 0; i < features; i++) {
            coefficients[i] = matrix[i][features];
        }
    }

    public static double predict(double[] row) {

        double[] x = getFeatures(row);

        double result = 0;

        for (int i = 0; i < coefficients.length; i++) {
            result += coefficients[i] * x[i];
        }

        return result;
    }

    public static void main(String[] args) throws Exception {

        List<double[]> data = new ArrayList<>();

        String file = "data/monthly_sales.csv";

        try (Reader reader = new FileReader(file);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {

            int index = 0;

            for (CSVRecord record : parser) {

                YearMonth date =
                        YearMonth.parse(record.get("Month"));

                double sales =
                        Double.parseDouble(record.get("Total Sales"));

                data.add(new double[]{
                    index + 1, 0, 0, sales
                });

                index++;
            }
        }

        if (data.size() < 48) {
            throw new IllegalArgumentException(
                "Dataset must contain at least 48 months."
            );
        }

        train(data);

        System.out.println("IMPROVED MODEL TRAINED SUCCESSFULLY!");

        System.out.println("\nRegression Coefficients:");

        for (int i = 0; i < coefficients.length; i++) {
            System.out.printf("b%d = %.4f%n", i, coefficients[i]);
        }

        System.out.println("\nTesting predictions:");

        double totalAbsoluteError = 0;
        double totalSquaredError = 0;

        int testCount = 0;

        for (int i = 36; i < 48; i++) {

            double[] row = data.get(i);

            double actual = row[3];
            double predicted = predict(row);

            double error = actual - predicted;

            totalAbsoluteError += Math.abs(error);
            totalSquaredError += error * error;

            testCount++;

            System.out.printf(
                "Month %d | Actual: %.2f | Predicted: %.2f | Error: %.2f%n",
                (int) row[0], actual, predicted, error
            );
        }

        double mae = totalAbsoluteError / testCount;
        double rmse = Math.sqrt(totalSquaredError / testCount);

        System.out.println("\nIMPROVED MODEL EVALUATION");

        System.out.printf("MAE: %.2f%n", mae);
        System.out.printf("RMSE: %.2f%n", rmse);

        System.out.println("\nFUTURE SALES FORECAST");

        for (int i = 48; i < 54; i++) {

            double[] futureRow = {
                i + 1, 0, 0, 0
            };

            double forecast = predict(futureRow);

            System.out.printf(
                "Month %d | Forecasted Sales: %.2f%n",
                i + 1, forecast
            );
        }
    }
}
