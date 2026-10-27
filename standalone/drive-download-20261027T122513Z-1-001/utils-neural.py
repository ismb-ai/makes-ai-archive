# utils.py
#
# This program is organized into a class-based structure, primarily designed for
# processing time series data. The main class is TimeSeriesProcessor, and it encapsulates
# functionality related to loading, preprocessing, and transforming time series data.
# The class constructor initializes various parameters such as the file path, time
# period, and dimensionality for lag embeddings. The class methods include load_data,
# responsible for loading and preprocessing the input data from either a CSV or 
# XLSX file, and to_vector, which creates lag embeddings for each item in the time
# series data. The to_takens method generates lag embeddings for a given DataFrame,
# and the preprocess method serves as the main entry point for data processing.
# 	Within the to_vector method, the code checks whether relative returns need
# to be calculated for the time series data. If enabled, it computes relative returns
# using the relative_returns method. This method calculates the relative change 
# between consecutive values in a vector, handling cases where the previous value
# is zero to avoid division by zero errors. The lag embeddings are then created using
# the to_takens method. The processed data, including lag embeddings and initial 
# values, is stored within the class instance.


import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

class TimeSeriesProcessor:
    def __init__(self, filepath, period, excel=False, dim=5, relative_returns=True):
        # Initialize the TimeSeriesProcessor with input parameters
        self.filepath = filepath
        self.period = period
        self.excel = excel
        self.dim = dim
        self.relative_returns = relative_returns
        self.df_items = None
        self.df_lag = None
        self.initial_values = None

    def load_data(self):
        # Load data from a CSV or XLSX file and preprocess it
        if self.excel:
            df = pd.read_excel(self.filepath)
            print("Loading xlsx ... OK!")
        else:
            df = pd.read_csv(self.filepath)
            print("Loading csv ... OK!")

        print("Estimating column(s) representing 'date'...")
        df['date'] = pd.to_datetime(df['date'])
        self.df_items = {}

        # Iterate over unique items and preprocess the data
        for item in df['item'].unique():
            print("Choosing item ", item)
            temp = df.loc[df['item'] == item, :].drop(columns=["item"])
            temp.set_index('date', inplace=True)
            temp = pd.DataFrame(temp.resample(self.period)['value'].sum())

            for i in range(len(temp)):
                if temp['value'][i] == 0:
                    print("Found zero for item:", item, " at i:", i)
                    temp['value'][i] = 1
            self.df_items[item] = temp

    def relative_returns(self, values):
        # Calculate relative returns for a given vector of values
        n = len(values)
        returns = [math.nan]

        for i in range(1, n):
            if values[i - 1] != 0:
                value = (values[i] - values[i - 1]) / abs(values[i - 1])
                returns.append(value)
            else:
                raise ValueError(f"Coordinate {i} is zero")

        return {"returns": returns, "initial_value": values[0]}

    def to_vector(self):
        # Create lag embeddings for each item in df_items
        X = {}

        if self.relative_returns:
            self.initial_values = {}

            for item in self.df_items.keys():
                temp = self.df_items[item]
                returns = self.relative_returns(temp['value'])
                self.df_items[item]['value'] = returns['returns']
                self.initial_values[item] = returns['initial_value']
                self.df_items[item] = self.df_items[item].iloc[1:, :]
                X[item] = self.to_takens(self.df_items[item])

            return [X, self.initial_values]

        else:
            for item in self.df_items.keys():
                X[item] = self.to_takens(self.df_items[item])

            return X

    def to_takens(self, df):
        # Create lag embeddings for a given DataFrame
        values = df['value']
        X = {'date': df.index}
        i = 0

        while i <= self.dim - 1:
            X.update({'Lag ' + str(i): values.shift(i)})
            i += 1

        X = pd.DataFrame(X)
        X = X.iloc[(self.dim - 1):, :].reset_index(drop=True)
        return X

    def preprocess(self):
        # Main preprocessing function that calls other methods
        self.load_data()

        if self.relative_returns:
            self.df_lag, self.initial_values = self.to_vector()
            print("Using relative_returns = True... Saving variable initial_values.")
            return [self.df_lag, self.initial_values]

        else:
            self.df_lag = self.to_vector()
            return self.df_lag

    def train_lstm_models(self):
        # Train LSTM models for each item in the dataset
        lstm_models = {}

        for item in self.df_lag.keys():
            print(f"Training LSTM for item: {item}")
            data = self.df_lag[item]
            X, y = data.drop(columns=['date']), data['Lag 0']

            # Reshape the data for LSTM input (samples, time steps, features)
            X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))

            # Define and compile the LSTM model
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(1, X.shape[1])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            # Fit the model
            model.fit(X_reshaped, y, epochs=50, batch_size=16, verbose=0)

            lstm_models[item] = model

        return lstm_models

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a TimeSeries Dataset')
    parser.add_argument('-FILEPATH', type=str)
    parser.add_argument('-period', type=str)
    parser.add_argument('-EXCEL', type=bool, default=False)
    parser.add_argument('-dim', type=int)
    parser.add_argument('-relative_returns', type=bool)

    args = parser.parse_args()
    print(args)

    processor = TimeSeriesProcessor(args.FILEPATH, args.period, args.EXCEL, args.dim, args.relative_returns)
    processor.preprocess()
    lstm_models = processor.train_lstm_models()
