
package com.future;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.time.LocalDate;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class App {

    public static void main(String[] args) throws Exception {

        String file = "data/Sample - Superstore.csv";

        Map<YearMonth, Double> monthlySales = new TreeMap<>();

        DateTimeFormatter formatter =
                DateTimeFormatter.ofPattern("M/d/yyyy");

        try (Reader reader = new FileReader(file);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {

            for (CSVRecord record : parser) {

                LocalDate date = LocalDate.parse(
                        record.get("Order Date"), formatter);

                YearMonth month = YearMonth.from(date);

                double sales =
                        Double.parseDouble(record.get("Sales"));

                monthlySales.put(
                        month,
                        monthlySales.getOrDefault(month, 0.0) + sales
                );
            }
        }

        try (FileWriter writer =
                     new FileWriter("data/monthly_sales.csv")) {

            writer.write("Month,Total Sales\n");

            for (Map.Entry<YearMonth, Double> entry
                    : monthlySales.entrySet()) {

                writer.write(
                        entry.getKey() + "," + entry.getValue() + "\n"
                );

                System.out.println(
                        entry.getKey() + " : " + entry.getValue()
                );
            }
        }

        System.out.println("\nMonthly sales saved successfully!");
    }
}
