# Forecast Web App using LSTM neural network.
#
# Igor Mol <igor.mol@makes.ai>

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# This function loads data from a CSV file:
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# This function aims to preprocess data for training the model:
def preprocess_data(data):
    X = data['TIME_SCALE'].values
    y = data['COUNT'].values

    # Split the data into training and testing sets (80-20 split)
    split_index = int(0.8 * len(X))
    X_train = X[:split_index]
    y_train = y[:split_index]

    # Reshape the training data for LSTM model input
    X_train = X_train.reshape((X_train.shape[0], 1, 1))

    return X_train, y_train, X, y

# Function to build the LSTM model:
def build_model():
    # Create a Sequential model
    model = Sequential()

    # Add the first LSTM layer with 100 units, ReLU activation, and input shape (1, 1)
    model.add(LSTM(units=100, activation='relu', input_shape=(1, 1), return_sequences=True))

    # Add the second LSTM layer with 200 units and ReLU activation
    model.add(LSTM(units=200, activation='relu', return_sequences=True))

    # Add the third LSTM layer with 300 units and ReLU activation
    model.add(LSTM(units=300, activation='relu', return_sequences=True))

    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.2))

    # Add the fourth LSTM layer with 200 units and ReLU activation
    model.add(LSTM(units=200, activation='relu'))

    # Add another dropout layer
    model.add(Dropout(0.2))

    # Add a Dense output layer with 1 unit (regression task)
    model.add(Dense(units=1))

    # Compile the model using Adam optimizer and Mean Squared Error (MSE) loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the compiled LSTM model
    return model

# Trains the provided LSTM model on the input training data.
#
# Parameters:
# - model: The LSTM model to be trained.
# - X_train: The input features for training.
# - y_train: The target values for training.
# - epochs: Number of training epochs (default is 50).
# - batch_size: Size of batches used for training (default is 1).
#
# Returns:
# - history: Training history containing loss values and metrics.
def train_model(model, X_train, y_train, epochs=50, batch_size=1):
    # Use the fit method to train the model on the provided data
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    # Return the training history
    return history


# Function to make predictions using the trained model
#
# Parameters:
# - model: The trained model for making predictions.
# - X: The input data for which predictions are made.
# - smoothing_factor: Factor for smoothing predictions (default is 0.1).
# - randomness_factor: Factor for introducing randomness (default is 0.1).
#
# Returns:
# - smoothed_predictions: Smoothed predictions based on the input data.
# - future_time_range: Time range for future predictions.
# - smoothed_future_predictions: Smoothed predictions for future time range.
def make_predictions(model, X, smoothing_factor=0.1, randomness_factor=0.1):
    # Reshape input data for compatibility with the model
    X = X.reshape((X.shape[0], 1, 1))
    
    # Use the trained model to make initial predictions
    predictions = model.predict(X)

    # Initialize list to store smoothed predictions
    smoothed_predictions = [predictions[0]]

    # Apply smoothing to the predictions to reduce noise
    for i in range(1, len(predictions)):
        noise = np.random.normal(scale=randomness_factor)
        smoothed_value = smoothing_factor * (predictions[i] + noise) + (1 - smoothing_factor) * smoothed_predictions[-1]
        smoothed_predictions.append(smoothed_value)

    # Generate future time range for predictions
    future_time_range = np.arange(max(X), max(X) + 13, 1)
    future_X = future_time_range.reshape((len(future_time_range), 1, 1))
    
    # Make predictions for the future time range using the model
    future_predictions = model.predict(future_X)

    # Initialize list to store smoothed future predictions
    smoothed_future_predictions = [smoothed_predictions[-1]]

    # Apply smoothing to the future predictions
    for i in range(1, len(future_predictions)):
        noise = np.random.normal(scale=randomness_factor)
        smoothed_value = smoothing_factor * (future_predictions[i] + noise) + (1 - smoothing_factor) * smoothed_future_predictions[-1]
        smoothed_future_predictions.append(smoothed_value)

    # Return the smoothed predictions, future time range, and smoothed future predictions
    return np.array(smoothed_predictions), future_time_range, np.array(smoothed_future_predictions)
    
# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions.flatten())
    st.write(f'Mean Absolute Error (MAE): {mae}')

# Function to plot the results
def plot_results(data, predictions, future_time_range, future_predictions, history):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot original data, model predictions, and future predictions
    axs[0].plot(data['TIME_SCALE'], data['COUNT'], label='Original Data', marker='o', linestyle='-')
    axs[0].plot(data['TIME_SCALE'], predictions, label='Model Prediction', linestyle='dashed', color='orange')
    axs[0].plot(future_time_range, future_predictions, label='Future Prediction (12 months)', linestyle='dashed', color='green')
    axs[0].set_title('RNN Model vs Original Data (Extrapolation)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('COUNT')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Mean Squared Error (MSE) over training epochs
    axs[1].plot(history.history['loss'], label='Mean Squared Error (MSE)')
    axs[1].set_title('Mean Squared Error (MSE) Over Training Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MSE')
    axs[1].legend()
    axs[1].grid(True)

    # Display the plots in the Streamlit app
    plt.tight_layout()
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    st.title('Sales Forecast Web App')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data, preprocess, build and train the model
        data = load_data(uploaded_file)
        X_train, y_train, X, y = preprocess_data(data)
        model = build_model()
        history = train_model(model, X_train, y_train)
        
        # Make predictions and display results
        predictions, future_time_range, future_predictions = make_predictions(model, X, randomness_factor=0.1)
        calculate_mae(y, predictions)
        plot_results(data, predictions, future_time_range, future_predictions, history)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
