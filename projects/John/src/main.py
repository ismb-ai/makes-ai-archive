import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

def load_data(file_path):
    # Load CSV file into the data DataFrame
    data = pd.read_csv(file_path)
    return data

# Define a class for Time Series Modeling
class TimeSeriesModel:
    def __init__(self, data, window_size=10, months_to_predict=12, randomness_factor=0.1, smoothing_factor=1):
        # Initialize the TimeSeriesModel with input data, window size, months to predict, randomness factor, and smoothing factor
        self.data = data
        self.window_size = window_size
        self.months_to_predict = months_to_predict
        self.randomness_factor = randomness_factor
        self.smoothing_factor = smoothing_factor
        self.predictions = []  # List to store model predictions
        self.future_extrapolation = []  # List to store future extrapolation time points

    def preprocess_data(self):
        # Preprocess the input data
        self.data = self.data.sort_values(by='TIME_SCALE')
        self.data = self.data.dropna(subset=['COUNT'])
        if self.data['COUNT'].isnull().sum() > 0:
            self.data['COUNT'].fillna(self.data['COUNT'].mean(), inplace=True)

    def train_model(self, X, y):
        # Train a linear Support Vector Machine (SVM) model
        model = LinearSVR(random_state=0, tol=1e-5, dual=False, loss='squared_epsilon_insensitive')
        model.fit(X, y)
        return model

    def generate_predictions(self, X, model):
        # Generate predictions using the trained model
        return model.predict(X)

    def exponential_smoothing(self, series):
        # Apply exponential smoothing to a time series
        smoothed_series = [series[0]]  # Assuming the first value remains the same

        for i in range(1, len(series)):
            smoothed_value = self.smoothing_factor * series[i] + (1 - self.smoothing_factor) * smoothed_series[-1]
            smoothed_series.append(smoothed_value)

        return smoothed_series

    def sliding_window_predictions(self):
        # Perform sliding window predictions
        for i in range(len(self.data) - self.window_size + 1):
            window_data = self.data.iloc[i:i + self.window_size]
            X = window_data['TIME_SCALE'].values.reshape(-1, 1)
            y = window_data['COUNT'].values

            # Skip the window if it contains NaN values
            if np.isnan(y).any():
                continue

            # Apply exponential smoothing to the current window
            y_smoothed = self.exponential_smoothing(y)

            # Train a linear SVM model and generate predictions
            model = self.train_model(X, y_smoothed)

            # If there are more data points, extrapolate to the next time point
            if i + self.window_size < len(self.data):
                next_time_point = self.data.iloc[i + self.window_size]['TIME_SCALE']
                self.future_extrapolation.append(next_time_point)
                prediction = self.generate_predictions([[next_time_point]], model)
                # Add randomness to the prediction
                self.predictions.append(prediction[0] + np.random.normal(scale=self.randomness_factor))

    def extrapolate_future(self):
        # Extrapolate future data points
        for _ in range(self.months_to_predict):
            X = self.data['TIME_SCALE'].values.reshape(-1, 1)
            y = self.data['COUNT'].values

            # Skip if there are NaN values
            if np.isnan(y).any():
                continue

            # Train a linear SVM model and generate predictions for the next time point
            model = self.train_model(X, y)
            next_time_point = self.future_extrapolation[-1] + 1
            self.future_extrapolation.append(next_time_point)
            prediction = self.generate_predictions([[next_time_point]], model)
            # Add randomness to the prediction
            self.predictions.append(prediction[0] + np.random.normal(scale=self.randomness_factor))

    def plot_results(self):
        # Plot the original data, model predictions, and future extrapolation
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        combined_time = np.concatenate((self.data['TIME_SCALE'], self.future_extrapolation))
        combined_predictions = np.concatenate(([np.nan] * len(self.data['TIME_SCALE']), self.predictions))

        sns.lineplot(x=self.data['TIME_SCALE'], y=self.data['COUNT'], label='Original Data', color='blue')
        sns.lineplot(x=combined_time, y=combined_predictions,
                     label='Model Prediction and Future Extrapolation', linestyle='dashed', color='green', marker='o')

        plt.xlabel('Time Scale')
        plt.ylabel('Count')
        plt.title(f'Time Series Modeling with Sliding Window and {self.months_to_predict} Months Future Extrapolation (Linear SVM)')
        plt.legend()
        plt.show()

# Define the main function
def main():
    # File path for CSV
    file_path = "/Users/igormol/Desktop/John/JZ.csv"

    # Load data
    data = load_data(file_path)
    
    # Create an instance of TimeSeriesModel
    time_series_model = TimeSeriesModel(data)  # Adjust the smoothing factor as needed
    # Preprocess the data
    time_series_model.preprocess_data()
    # Perform sliding window predictions
    time_series_model.sliding_window_predictions()
    # Extrapolate future data points
    time_series_model.extrapolate_future()
    # Plot the results
    time_series_model.plot_results()

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
