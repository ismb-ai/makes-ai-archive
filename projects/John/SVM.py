# Sliding-window Approach to Time Series Forecasting Using Gaussiann SVM
#
# Igor Mol <igor.mol@makes.ai>
#
# The following code implements a Time Series Model using the Support Vector 
# Regression (SVM) technique for forecasting. The model is encapsulated within a
# class, facilitating organized and modular code structure. The overarching goal
# of this code is to perform time series analysis on a given dataset named 'data.'
# The dataset undergoes preprocessing, which involves sorting by the 'TIME_SCALE'
# column and handling missing values in the 'COUNT' column by dropping corresponding
# rows or filling NaN values with the mean. The time series modeling is conducted 
# through a sliding window approach, where historical predictions are generated 
# using SVM based on a specified window size. The extrapolation into the future
# is achieved by iteratively predicting subsequent time points, incorporating a
# randomness factor to introduce variability into the model's predictions. 
# The code also calculates performance metrics, such as Mean Absolute Error (MAE), 
# Covariance Matrix, and Standard Deviation, providing quantitative assessments
# of the model's accuracy.
#     The SVM model is constructed using the Scikit-Learn library, employing a
# radial basis function (RBF) kernel for flexibility in capturing complex 
# relationships within the time series. The sliding window predictions leverage
# historical data to iteratively train and predict, while the extrapolation step
# extends predictions into the future for a specified number of months. The
# randomness factor introduces a stochastic element, acknowledging the inherent
# uncertainty in future predictions.

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

class TimeSeriesModel:
    def __init__(self, data, window_size=1, months_to_predict=12, randomness_factor=0.9):
        """
        Initialize the TimeSeriesModel with input data and configuration parameters.

        Parameters:
        - data: Input time-series data as a Pandas DataFrame.
        - window_size: Size of the sliding window for predictions.
        - months_to_predict: Number of months to extrapolate into the future.
        - randomness_factor: Factor to introduce randomness in future extrapolation.
        """
        self.data = data
        self.window_size = window_size
        self.months_to_predict = months_to_predict
        self.randomness_factor = randomness_factor
        self.predictions = []  # List to store model predictions
        self.future_extrapolation = []  # List to store future extrapolation time points

    def preprocess_data(self):
        """
        Preprocess the input data by sorting, removing NaN values, and filling missing values.
        """
        # Sort the data based on the 'TIME_SCALE' column
        self.data = self.data.sort_values(by='TIME_SCALE')
        # Drop rows where 'COUNT' is NaN
        self.data = self.data.dropna(subset=['COUNT'])

        # Fill NaN values in 'COUNT' with the mean of the column
        if self.data['COUNT'].isnull().sum() > 0:
            self.data['COUNT'].fillna(self.data['COUNT'].mean(), inplace=True)

    def train_model(self, X, y):
        """
        Train a Support Vector Regression (SVR) model with an RBF kernel.

        Parameters:
        - X: Input features.
        - y: Target variable.

        Returns:
        Trained SVR model.
        """
        model = SVR(kernel='rbf', gamma='scale')
        model.fit(X, y)
        return model

    def generate_predictions(self, X, model):
        """
        Generate predictions using a trained model.

        Parameters:
        - X: Input features for prediction.
        - model: Trained model.

        Returns:
        Array of predictions.
        """
        return model.predict(X)

    def sliding_window_predictions(self):
        """
        Perform sliding window predictions on the input time series data.
        """
        # Iterate through the data using a sliding window approach
        for i in range(len(self.data) - self.window_size + 1):
            # Extract a window of data
            window_data = self.data.iloc[i:i + self.window_size]
            # Prepare features (X) and target variable (y) for the model
            X = window_data['TIME_SCALE'].values.reshape(-1, 1)
            y = window_data['COUNT'].values

            # Skip if there are NaN values in the target variable
            if np.isnan(y).any():
                continue

            # Train the model using the current window
            model = self.train_model(X, y)

            # If there are more data points, predict the next time point and store the prediction
            if i + self.window_size < len(self.data):
                next_time_point = self.data.iloc[i + self.window_size]['TIME_SCALE']
                self.future_extrapolation.append(next_time_point)
                prediction = self.generate_predictions([[next_time_point]], model)
                self.predictions.append(prediction[0])

                # Add 'PREDICTION' column to the DataFrame
                self.data.at[i + self.window_size, 'PREDICTION'] = prediction[0]

    def extrapolate_future(self):
        """
        Extrapolate future data points for a specified number of months.
        """
        # Iterate for the specified number of months
        for _ in range(self.months_to_predict):
            # Prepare features (X) and target variable (y) for the model using all available data
            X = self.data['TIME_SCALE'].values.reshape(-1, 1)
            y = self.data['COUNT'].values

            # Skip if there are NaN values in the target variable
            if np.isnan(y).any():
                continue

            # Train the model using all available data
            model = self.train_model(X, y)

            # Predict the next time point and add a random factor
            next_time_point = self.future_extrapolation[-1] + 1
            self.future_extrapolation.append(next_time_point)
            prediction = self.generate_predictions([[next_time_point]], model)
            self.predictions.append(prediction[0] + np.random.normal(scale=self.randomness_factor))

    def calculate_metrics(self):
        """
        Calculate Mean Absolute Error (MAE), covariance matrix, and standard deviation.
        """
        mae = mean_absolute_error(self.data['COUNT'][:len(self.data)], self.predictions[:len(self.data)])
        covariance_matrix = np.cov(self.data['COUNT'][:len(self.data)], self.predictions[:len(self.data)])
        std_deviation = np.std(self.data['COUNT'][:len(self.data)] - self.predictions[:len(self.data)])
        return mae, covariance_matrix, std_deviation

    def plot_results(self):
        """
        Plot the original data, model predictions, and future extrapolation.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        combined_time = np.concatenate((self.data['TIME_SCALE'], self.future_extrapolation))
        combined_predictions = np.concatenate((self.data['PREDICTION'], self.predictions))

        sns.lineplot(x=self.data['TIME_SCALE'], y=self.data['COUNT'], label='Original Data', color='blue')
        sns.lineplot(x=combined_time, y=combined_predictions,
                     label='Model Prediction and Future Extrapolation', linestyle='dashed', color='green', marker='o')

        plt.xlabel('Time Scale')
        plt.ylabel('Count')
        plt.title(f'Time Series Modeling with Sliding Window and {self.months_to_predict} Months Future Extrapolation (SVM)')
        plt.legend()
        plt.show()

    def display_results_table(self, mae, covariance_matrix, std_deviation):
        """
        Display a table of calculated metrics.

        Parameters:
        - mae: Mean Absolute Error.
        - covariance_matrix: Covariance matrix.
        - std_deviation: Standard deviation.
        """
        results_table = pd.DataFrame({
            'Metric': ['Mean Absolute Error (MAE)', 'Covariance Matrix', 'Standard Deviation'],
            'Value': [mae, covariance_matrix, std_deviation]
        })

        print(tabulate(results_table, headers='keys', tablefmt='fancy_grid'))


def main():
    # Assuming your DataFrame is named 'data'
    # If not, replace 'data' with your actual DataFrame name
    time_series_model = TimeSeriesModel(data)
    time_series_model.preprocess_data()
    time_series_model.sliding_window_predictions()
    time_series_model.extrapolate_future()
    mae, covariance_matrix, std_deviation = time_series_model.calculate_metrics()
    time_series_model.plot_results()
    time_series_model.display_results_table(mae, covariance_matrix, std_deviation)


if __name__ == "__main__":
    main()
