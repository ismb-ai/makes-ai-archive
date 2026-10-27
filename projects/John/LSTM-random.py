# Hybrid Deep Learning Approach with Randomness
#
# Igor Mol <igor.mol@makes.ai>
#
# The provided code is a comprehensive Python script for time series forecasting
# using a recurrent neural network (RNN) with Long Short-Term Memory (LSTM) lay-
# ers. It follows a structured approach, encapsulating functionality within fun-
# ctions and a class for better organization. The first set of functions revol-
# ves around data handling: load_data loads a CSV file into a Pandas DataFrame,
# while preprocess_data extracts the relevant features and target variables, re-
# serves 80% of the data for training, and reshapes the input data for the RNN.
#     The subsequent functions are focused on constructing and training the deep
# learning model. build_model creates a Sequential model with an architecture 
# comprising multiple LSTM layers and a dense output layer. The train_model fun-
# ction utilizes the training data to fit the model, specifying the number of 
# epochs and batch size. The training history is recorded for later analysis. The
# make_predictions function employs the trained model to predict the entire time
# interval of the original data and extrapolate future values. It returns the 
# model predictions for both the historical data and the future time range.
#     Moving on to evaluation, the calculate_mae function computes the Mean 
# Absolute Error (MAE) between the true target values and the predictions, pro-
# viding a quantitative measure of model performance. Finally, the plot_results 
# function generates a visually informative plot, displaying the original time 
# series, model predictions, and extrapolated future predictions. It also inclu-
# des a subplot illustrating the Mean Squared Error (MSE) over training epochs,
# offering insights into the convergence and performance of the neural network.
# This structured and modular approach enhances code readability and facilitates
# reuse in different scenarios.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data(file_path):
    # Load CSV file into the data DataFrame
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Extracting the features and target variable
    X = data['TIME_SCALE'].values
    y = data['COUNT'].values

    # Use 80% of the data for training
    split_index = int(0.8 * len(X))
    X_train = X[:split_index]
    y_train = y[:split_index]

    # Reshape the input data for RNN
    X_train = X_train.reshape((X_train.shape[0], 1, 1))

    return X_train, y_train, X, y

def build_model():
    # Building a deeper and more complex RNN model
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', input_shape=(1, 1), return_sequences=True))
    model.add(LSTM(units=200, activation='relu', return_sequences=True))
    model.add(LSTM(units=300, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, epochs=300, batch_size=1):
    # Training the model on 80% of the time interval
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    return history

def make_predictions(model, X, noise_factor=5):
    # Use the entire time interval for predictions
    X = X.reshape((X.shape[0], 1, 1))
    predictions = model.predict(X)

    # Add more random fluctuations or white noise to current predictions
    noise = noise_factor * np.random.normal(size=predictions.shape)
    predictions_with_noise = predictions + noise

    # Extrapolate to predict the future 12 months ahead
    future_time_range = np.arange(max(X), max(X) + 13, 1)  # Extend by 12 months
    future_X = future_time_range.reshape((len(future_time_range), 1, 1))
    future_predictions = model.predict(future_X)

    # Add more random fluctuations to future predictions as well
    future_noise = noise_factor * np.random.normal(size=future_predictions.shape)
    future_predictions_with_noise = future_predictions + future_noise

    return predictions_with_noise.flatten(), future_time_range, future_predictions_with_noise.flatten()

def calculate_mae(y_true, predictions):
    # Calculate MAE
    mae = mean_absolute_error(y_true, predictions.flatten())
    print(f'Mean Absolute Error (MAE): {mae}')

def plot_results(data, predictions, future_time_range, future_predictions, history):
    # Plotting the results with improved quality
    plt.figure(figsize=(12, 8))

    # Plot Original Data and Predictions
    plt.subplot(2, 1, 1)
    plt.plot(data['TIME_SCALE'], data['COUNT'], label='Original Data', marker='o', linestyle='-')
    plt.plot(data['TIME_SCALE'], predictions, label='Model Prediction', linestyle='dashed', color='orange')
    plt.plot(future_time_range, future_predictions, label='Future Prediction (12 months)', linestyle='dashed', color='green')
    plt.title('RNN Model vs Original Data (Extrapolation)')
    plt.xlabel('Time')
    plt.ylabel('COUNT')
    plt.legend()
    plt.grid(True)

    # Plot MSE over training epochs
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Mean Squared Error (MSE)')
    plt.title('Mean Squared Error (MSE) Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# File path for CSV
# file_path = "/Users/igormol/Desktop/John/John.csv"

# Load data
# data = load_data(file_path)

# Preprocess data
X_train, y_train, X, y = preprocess_data(data)

# Build model
model = build_model()

# Train model
history = train_model(model, X_train, y_train)

# Make predictions with more random fluctuations
noise_factor = 7
predictions, future_time_range, future_predictions = make_predictions(model, X, noise_factor)

# Calculate MAE for predictions with noise
calculate_mae(y, predictions)

# Plot results with predictions including more random fluctuations
plot_results(data, predictions, future_time_range, future_predictions, history)
