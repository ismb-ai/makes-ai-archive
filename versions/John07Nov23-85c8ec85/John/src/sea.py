# Seasonality Detection in Time Series with Exponential Smoothing
#
# Igor Mol <igor.mol@makes.ai>
#
# 	The provided Python code implements a "SeasonalityDetector" class for 
# identifying and analyzing seasonality patterns in a time series dataset. The 
# detector applies exponential smoothing to the data, detects seasonality patterns
# within specified windows, and then prints a table and plots the detected patterns.
# The main function demonstrates the usage of the SeasonalityDetector class with
# a sample dataset. This code is valuable for understanding and visualizing re-
# curring patterns in time series data. 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define a class for Seasonality Detector
class SeasonalityDetector:
    def __init__(self, data, window_size=2, smoothing_factor=0.2):
        # Initialize the SeasonalityDetector with input data, window size, and smoothing factor
        self.data = data
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.detected_seasonality = []  # List to store detected seasonality patterns

    def preprocess_data(self):
        # Preprocess the input data
        # Sort data based on 'TIME_SCALE' column
        self.data = self.data.sort_values(by='TIME_SCALE')
        # Drop rows with missing values in the 'COUNT' column
        self.data = self.data.dropna(subset=['COUNT'])
        # If there are still missing values, fill them with the mean of the 'COUNT' column
        if self.data['COUNT'].isnull().sum() > 0:
            self.data['COUNT'].fillna(self.data['COUNT'].mean(), inplace=True)

    def exponential_smoothing(self, series):
        # Apply exponential smoothing to a time series
        smoothed_series = [series[0]]  # Assuming the first value remains the same

        for i in range(1, len(series)):
            smoothed_value = self.smoothing_factor * series[i] + (1 - self.smoothing_factor) * smoothed_series[-1]
            smoothed_series.append(smoothed_value)

        return smoothed_series

    def detect_seasonality(self):
        # Detect seasonality patterns in the data
        for i in range(len(self.data) - self.window_size + 1):
            window_data = self.data.iloc[i:i + self.window_size]
            y = window_data['COUNT'].values

            # Skip the window if it contains NaN values
            if np.isnan(y).any():
                continue

            # Apply exponential smoothing to the current window
            y_smoothed = self.exponential_smoothing(y)

            # Calculate the seasonal pattern and append it to the list
            seasonal_pattern = np.mean(y - y_smoothed)
            self.detected_seasonality.append(seasonal_pattern)

    def print_seasonality_table(self):
        # Print the results table with window index and detected seasonality
        results_table = pd.DataFrame({
            'Window Index': range(len(self.detected_seasonality)),
            'Detected Seasonality': self.detected_seasonality
        })

        print(tabulate(results_table, headers='keys', tablefmt='fancy_grid'))

    def plot_seasonality(self):
        # Plot the detected seasonality patterns
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        plt.plot(self.detected_seasonality, label='Detected Seasonality', linestyle='dashed', color='red')

        plt.xlabel('Window Index')
        plt.ylabel('Detected Seasonality')
        plt.title('Seasonality Detection in Time Series')
        plt.legend()
        plt.show()

# Define the main function
def main():
    # Replace 'new_data' with your actual DataFrame
    data = new_data
    
    # Create an instance of SeasonalityDetector
    seasonality_detector = SeasonalityDetector(data)
    # Preprocess the data
    seasonality_detector.preprocess_data()
    # Detect seasonality patterns
    seasonality_detector.detect_seasonality()
    # Print the results table
    seasonality_detector.print_seasonality_table()
    # Plot the detected seasonality patterns
    seasonality_detector.plot_seasonality()

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
