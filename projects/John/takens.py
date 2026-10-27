# Takens Embedding Theorem Approach to Time Series Modelling with Guassian SVM
#
# Igor Mol <igor.mol@makes.ai>
#
# This code implements a time series forecasting model using Support Vector 
# Regression (SVM) with a radial basis function (RBF) kernel. The primary 
# technique used for the time series analysis is based on Takens' theorem, which
# involves creating lagged embeddings of the time series data. The code defines 
# a class called TimeSeriesModel, which encapsulates the functionality for data 
# preprocessing, model training, prediction generation, exponential smoothing, 
# lagged embedding, sliding window predictions, future extrapolation, and result
# visualization.
#     The TimeSeriesModel class is initialized with various parameters such as 
# the input data, window size, months to predict, randomness factor, and smoothing 
# factor. The preprocess_data method sorts the data based on the time scale and 
# handles missing values by filling them with the mean. The lagged embedding is 
# achieved using the lagged_embedding method, and exponential smoothing is applied 
# with the exponential_smoothing method.
#     The sliding window predictions are generated using the sliding_window_predictions
# method, where SVM is trained on lagged embeddings of the smoothed time series within
# a sliding window. The future extrapolation is performed using the extrapolate_future
# method, extending the time series predictions into the future. Finally, the plot_results
# method visualizes the original data, model predictions, and future extrapolation. 
# Additionally, the display_results_table method prints relevant metrics such as 
# Mean Absolute Error (MAE), Covariance Matrix, and Standard Deviation.
#     In the main function, a sample data frame (new_data) is used to create an 
# instance of the TimeSeriesModel class, preprocess the data, generate sliding 
# window predictions, perform future extrapolation, and visualize the results. 
# The script is designed to be adaptable to different datasets by replacing the
# placeholder 'new_data' with an actual DataFrame containing time series data.

import pandas as pd
import numpy as np
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

class TimeSeriesModel:
    def __init__(self, data, window_size=5, months_to_predict=12, randomness_factor=0, smoothing_factor=1):
        self.data = data
        self.window_size = window_size
        self.months_to_predict = months_to_predict
        self.randomness_factor = randomness_factor
        self.smoothing_factor = smoothing_factor
        self.predictions = []
        self.future_extrapolation = []

    def preprocess_data(self):
        self.data = self.data.sort_values(by='TIME_SCALE')
        self.data = self.data.dropna(subset=['COUNT'])
        if self.data['COUNT'].isnull().sum() > 0:
            self.data['COUNT'].fillna(self.data['COUNT'].mean(), inplace=True)

    def train_model(self, X, y):
        model = SVR(kernel='rbf', gamma='scale')
        model.fit(X, y)
        return model

    def generate_predictions(self, X, model):
        return model.predict(X)

    def exponential_smoothing(self, series):
        smoothed_series = [series[0]]
        for i in range(1, len(series)):
            smoothed_value = self.smoothing_factor * series[i] + (1 - self.smoothing_factor) * smoothed_series[-1]
            smoothed_series.append(smoothed_value)
        return smoothed_series

    def lagged_embedding(self, series, embedding_dimension):
        embedded_series = []
        for i in range(len(series) - embedding_dimension + 1):
            embedding = series[i:i + embedding_dimension]
            embedded_series.append(embedding)
        return np.array(embedded_series)

    def sliding_window_predictions(self, embedding_dimension=3):
        for i in range(len(self.data) - self.window_size - embedding_dimension + 1):
            window_data = self.data.iloc[i:i + self.window_size + embedding_dimension]
            X = window_data['TIME_SCALE'].values.reshape(-1, 1)
            y = window_data['COUNT'].values

            if np.isnan(y).any():
                continue

            y_smoothed = self.exponential_smoothing(y)
            X_embedded = self.lagged_embedding(y_smoothed, embedding_dimension)

            y_smoothed = y_smoothed[embedding_dimension - 1:]

            model = self.train_model(X_embedded[:-1], y_smoothed)

            if i + self.window_size + embedding_dimension < len(self.data):
                next_time_point = self.data.iloc[i + self.window_size + embedding_dimension]['TIME_SCALE']
                self.future_extrapolation.append(next_time_point)
                next_point_embedded = self.lagged_embedding([next_time_point], embedding_dimension)

                if next_point_embedded.size > 0:
                    prediction = self.generate_predictions(next_point_embedded.reshape(1, -1), model)
                    self.predictions.append(prediction[0] + np.random.normal(scale=self.randomness_factor))
                else:
                    self.predictions.append(np.nan)

    def extrapolate_future(self):
        for _ in range(self.months_to_predict):
            X = self.data['TIME_SCALE'].values.reshape(-1, 1)
            y = self.data['COUNT'].values

            if np.isnan(y).any():
                continue

            model = self.train_model(X, y)
            next_time_point = self.future_extrapolation[-1] + 1
            self.future_extrapolation.append(next_time_point)
            prediction = self.generate_predictions([[next_time_point]], model)
            self.predictions.append(prediction[0] + np.random.normal(scale=self.randomness_factor))

    def plot_results(self):
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
        results_table = pd.DataFrame({
            'Metric': ['Mean Absolute Error (MAE)', 'Covariance Matrix', 'Standard Deviation'],
            'Value': [mae, covariance_matrix, std_deviation]
        })

        print(tabulate(results_table, headers='keys', tablefmt='fancy_grid'))

def main():
    data = new_data  # Replace 'new_data' with your actual DataFrame
    time_series_model = TimeSeriesModel(data)
    time_series_model.preprocess_data()
    time_series_model.sliding_window_predictions()
    time_series_model.extrapolate_future()
    time_series_model.plot_results()

if __name__ == "__main__":
    main()
