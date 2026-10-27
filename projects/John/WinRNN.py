import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Lambda
from tensorflow.keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error

def set_seaborn_style():
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)

def extract_features_target(data):
    features = data['TIME_SCALE'].values
    target = data['COUNT'].values
    return features, target

def create_sliding_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def build_rnn_model(window_size):
    model = Sequential()
    model.add(SimpleRNN(units=128, input_shape=(window_size, 1), activation='relu', return_sequences=True))
    model.add(SimpleRNN(units=100, activation='relu', return_sequences=True))
    model.add(SimpleRNN(units=100, activation='relu', return_sequences=True))  # Return sequences for exponential smoothing
    model.add(Lambda(lambda x: K.exp(-0.1) * x))  # Exponential smoothing layer
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X, y, epochs=500, batch_size=8, verbose=2):
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

def extrapolate_predictions(model, features, target, window_size, future_time_steps):
    X = create_sliding_windows(features, window_size)
    y = target[window_size - 1:]

    model.fit(X.reshape((X.shape[0], window_size, 1)), y, epochs=500, batch_size=8, verbose=2)

    full_features = np.arange(features[0] - window_size + 1, features[-1] + 1 + future_time_steps)
    full_windows = create_sliding_windows(full_features, window_size)
    full_windows = full_windows.reshape((full_windows.shape[0], window_size, 1))

    full_y_pred = model.predict(full_windows)
    return full_y_pred

def calculate_metrics(target, predictions):
    # Flatten the predictions array
    predictions = predictions.flatten()

    mae = mean_absolute_error(target, predictions[:len(target)])
    mse_values = [mean_squared_error(target[i:], predictions[:len(target) - i]) for i in range(len(target))]
    return mae, mse_values


def plot_results(time_indices, full_y_pred, target, mse_values, window_size):
    plt.figure(figsize=(12, 6))

    # Ensure lengths match by truncating or padding the arrays
    min_len = min(len(time_indices), len(full_y_pred.flatten()), len(target))
    time_indices = time_indices[:min_len]
    full_y_pred = full_y_pred.flatten()[:min_len]
    target = target[:min_len]

    sns.lineplot(x=time_indices, y=full_y_pred, label='Predicted')
    sns.lineplot(x=np.arange(len(target)), y=target, label='Original', linestyle='--')
    plt.xlabel('Time Index')
    plt.ylabel('COUNT')
    plt.title('Predicted vs Original for the Entire Time Interval')
    plt.legend()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=np.arange(len(target)), y=mse_values, label='MSE Trend', color='red', linestyle='dashed')
    plt.xlabel('Time Index')
    plt.ylabel('MSE')
    plt.title('MSE Trend over Time')
    plt.legend()


def main():
    set_seaborn_style()

    # Load time series data from CSV file
    file_path = "/Users/igormol/Desktop/John/KITPL1A.csv"
    data = pd.read_csv(file_path)

    features, target = extract_features_target(data)

    window_size = 3
    model = build_rnn_model(window_size)

    future_time_steps = 3
    full_y_pred = extrapolate_predictions(model, features, target, window_size, future_time_steps)

    time_indices = np.arange(features[0], features[0] + len(full_y_pred))

    mae, mse_values = calculate_metrics(target, full_y_pred)

    plot_results(time_indices, full_y_pred, target, mse_values, window_size)

    print(f'MAE: {mae}')

if __name__ == "__main__":
    main()
