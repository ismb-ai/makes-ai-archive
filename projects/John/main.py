# Time Series Analysis: Integrating Data Cleaning with Z-Score Outlier Detection 
# and Support Vector Regression Modeling
#
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Data cleaning function using Z-score
def z_score_outlier_detection(data_frame, column, threshold=1):
    z_scores = np.abs((data_frame[column] - data_frame[column].mean()) / data_frame[column].std())
    outliers = z_scores > threshold
    data_frame.loc[outliers, column] = np.nan
    return data_frame

# Class to model time series
class TimeSeriesModel:
    def __init__(self, data, window_size=1, months_to_predict=12, randomness_factor=1):
        # Initialize with input data and parameters
        self.data = data
        self.window_size = window_size
        self.months_to_predict = months_to_predict
        self.randomness_factor = randomness_factor
        self.predictions = []  
        self.future_extrapolation = []

    def preprocess_data(self):
        # Sort and clean the input data
        self.data = self.data.sort_values(by='TIME_SCALE')
        self.data = self.data.dropna(subset=['COUNT'])
        if self.data['COUNT'].isnull().sum() > 0:
            self.data['COUNT'].fillna(self.data['COUNT'].mean(), inplace=True)

    def train_model(self, X, y):
        # Train an SVM model with radial basis function (RBF) kernel
        model = SVR(kernel='rbf', gamma='scale')
        model.fit(X, y)
        return model

    def generate_predictions(self, X, model):
        # Generate predictions using the trained model
        return model.predict(X)

    def sliding_window_predictions(self):
        # Perform sliding window predictions
        for i in range(len(self.data) - self.window_size + 1):
            window_data = self.data.iloc[i:i + self.window_size]
            X = window_data['TIME_SCALE'].values.reshape(-1, 1)
            y = window_data['COUNT'].values

            # Skip if there are NaN values in the target variable
            if np.isnan(y).any():
                continue

            # Train model and make predictions
            model = self.train_model(X, y)
            if i + self.window_size < len(self.data):
                next_time_point = self.data.iloc[i + self.window_size]['TIME_SCALE']
                self.future_extrapolation.append(next_time_point)
                prediction = self.generate_predictions([[next_time_point]], model)
                self.predictions.append(prediction[0])

    def extrapolate_future(self):
        # Extrapolate future values
        for _ in range(self.months_to_predict):
            X = self.data['TIME_SCALE'].values.reshape(-1, 1)
            y = self.data['COUNT'].values

            # Skip if there are NaN values in the target variable
            if np.isnan(y).any():
                continue

            # Train model and make predictions with added randomness
            model = self.train_model(X, y)
            next_time_point = self.future_extrapolation[-1] + 1
            self.future_extrapolation.append(next_time_point)
            prediction = self.generate_predictions([[next_time_point]], model)
            self.predictions.append(prediction[0] + np.random.normal(scale=self.randomness_factor))

    def calculate_metrics(self):
        # Calculate evaluation metrics
        start_index = self.window_size
        end_index = start_index + len(self.predictions)
        y_true = self.data['COUNT'].iloc[start_index:end_index].values
        non_nan_indices = ~np.isnan(self.predictions)
        y_pred = np.array(self.predictions)[non_nan_indices]
        mae = mean_absolute_error(y_true, y_pred)
        covariance_matrix = np.cov(y_true, y_pred)
        std_deviation = np.std(y_true - y_pred)
        return mae, covariance_matrix, std_deviation

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
        plt.title(f'Time Series Modeling with Sliding Window and {self.months_to_predict} Months Future Extrapolation (SVM)')
        plt.legend()
        plt.show()

    def display_results_table(self, mae, covariance_matrix, std_deviation):
        # Display results table
        results_table = pd.DataFrame({
            'Metric': ['Mean Absolute Error (MAE)', 'Covariance Matrix', 'Standard Deviation'],
            'Value': [mae, covariance_matrix, std_deviation]
        })
        print(tabulate(results_table, headers='keys', tablefmt='fancy_grid'))

# Main function
def main():
    # Assuming 'data' is the original DataFrame
    original_data = data.copy()
    
    # Data cleaning
    cleaned_data = z_score_outlier_detection(original_data, 'COUNT')
    
    # Time series modeling
    time_series_model = TimeSeriesModel(cleaned_data)
    time_series_model.preprocess_data()
    time_series_model.sliding_window_predictions()
    time_series_model.extrapolate_future()
    mae, covariance_matrix, std_deviation = time_series_model.calculate_metrics()
    time_series_model.plot_results()
    time_series_model.display_results_table(mae, covariance_matrix, std_deviation)

if __name__ == "__main__":
    main()
