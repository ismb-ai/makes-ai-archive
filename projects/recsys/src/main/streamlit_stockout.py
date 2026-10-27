# Web App to Detect Stock-outs using Autoencoder
#
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import streamlit as st

# Simple routine to load data from a CSV file and return it as a DataFrame.
def load_data(file_path):
    return pd.read_csv(file_path)

# A routine to drop missing values from the DataFrame 'df'.
def drop_missing_values(df):
    return df.dropna()

# The function prepare_training_data organizes the training data for a machine 
# learning model. It selects relevant columns ('Quantity' and 'Price') and  
# scales the features. In a next version, we should modify this function in
# order to deduce the meaning of each column from their names using a NLP.
#
# Parameters:
# - df: The input DataFrame containing relevant columns for training.
#
# Returns:
# - X_scaled: Scaled features ready for training the machine learning model.
def prepare_training_data(df):
    # Define the columns to be used for training
    columns_for_training = ['Quantity', 'Price']
    
    # Extract the selected columns as feature matrix X
    X = df[columns_for_training].values
    
    # Initialize a StandardScaler to scale the features
    scaler = StandardScaler()
    
    # Scale the features using the fitted scaler
    X_scaled = scaler.fit_transform(X)
    
    # Return the scaled features for training
    return X_scaled

# The function identify_stock_outs identifies stock-out and creates
# a binary label column 'StockOut' indicating whether a stock-out occurred.
#
# Parameters:
# - df: The input DataFrame containing information about stock quantities.
#
# Returns:
# - stock_outs: A DataFrame containing rows with stock-out instances.
def identify_stock_outs(df):
    # Filter rows where the 'Quantity' column is equal to 0
    stock_outs = df[df['Quantity'] == 0]
    
    # Create a new binary column 'StockOut' based on the stock-out condition
    df['StockOut'] = np.where(df['Quantity'] == 0, 1, 0)
    
    # Return the DataFrame containing stock-out instances
    return stock_outs

# Function to split data into training and testing sets
def split_data(X_scaled, y):
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# The function build_autoencoder builds a autoencoder neural network.
# An autoencoder is a type of neural network used for dimensionality reduction,
# and it consists of an encoder and a decoder.
#
# Parameters:
# - input_dim: The dimensionality of the input data.
# - encoding_dim: The dimensionality of the encoding space (latent space).
#
# Returns:
# - autoencoder: A Keras model representing the autoencoder.
def build_autoencoder(input_dim, encoding_dim):
    # Define the input layer with the specified input dimension
    input_layer = Input(shape=(input_dim,))
    
    # Create the encoder layer with 'encoding_dim' neurons and ReLU activation
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    
    # Create the decoder layer with 'input_dim' neurons and sigmoid activation
    decoder = Dense(input_dim, activation="sigmoid")(encoder)
    
    # Create the autoencoder model, connecting the input and decoder layers
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Compile the autoencoder using Adam optimizer and Mean Squared Error (MSE) loss
    autoencoder.compile(optimizer="adam", loss="mse")
    
    # Return the compiled autoencoder model
    return autoencoder

# train_autoencoder
# Function to train the autoencoder model on non-stock-out instances
# Trains the provided autoencoder model on non-stock-out instances.
#
# Parameters:
# - autoencoder: The autoencoder model to be trained.
# - X_train_non_stock_out: The input data for training, representing non-stock-out instances.
def train_autoencoder(autoencoder, X_train_non_stock_out):
    # Train the autoencoder on non-stock-out instances with specified parameters
    autoencoder.fit(X_train_non_stock_out, X_train_non_stock_out, epochs=5, batch_size=32, shuffle=True)

# reconstruct_data
# Function to reconstruct data using the trained autoencoder
# Reconstructs input data using the trained autoencoder.
#
# Parameters:
# - autoencoder: The trained autoencoder model.
# - X_scaled: The input data to be reconstructed.
# Returns: reconstructed_data: The reconstructed data.
def reconstruct_data(autoencoder, X_scaled):
    # Use the trained autoencoder to reconstruct the input data
    return autoencoder.predict(X_scaled)

# calculate_reconstruction_error
# Function to calculate the reconstruction error for each sample
# Calculates the mean absolute reconstruction error for each sample.
#
# Parameters:
# - X_scaled: The original scaled data.
# - reconstructed_data: The data reconstructed by the autoencoder.
# Returns: reconstruction_error: Mean absolute reconstruction error for each sample.
def calculate_reconstruction_error(X_scaled, reconstructed_data):
    # Calculate the mean absolute reconstruction error for each sample
    return np.mean(np.abs(X_scaled - reconstructed_data), axis=1)

# detect_stock_outs_with_threshold
# This function detects stock-outs in a DataFrame based on a specified threshold
# for reconstruction errors. It is designed for anomaly detection using an
# autoencoder model, where reconstruction errors measure deviations from normality.
#
# Parameters:
# - df: The input DataFrame containing relevant information.
# - reconstruction_error: A column representing reconstruction errors from an autoencoder.
# - threshold: The specified threshold for identifying stock-outs.
#
# Returns:
# - predicted_stock_outs: Rows from the DataFrame where reconstruction errors
#   are greater than the specified threshold, indicating potential stock-outs.
def detect_stock_outs_with_threshold(df, reconstruction_error, threshold):
    # Select rows where the reconstruction error exceeds the specified threshold
    predicted_stock_outs = df[reconstruction_error > threshold]
    
    # Return the DataFrame containing potential stock-out instances
    return predicted_stock_outs

# Main function to create the Streamlit web app
def main():
    st.title("Stock-Out Detection App")
    
    # Allow user to upload a CSV file
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        # Load data and preprocess
        df = load_data(file)
        df = drop_missing_values(df)
        X_scaled = prepare_training_data(df)
        stock_outs = identify_stock_outs(df)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(X_scaled, df['StockOut'].values)

        # Build and train the autoencoder model
        input_dim = X_train.shape[1]
        encoding_dim = 2
        autoencoder = build_autoencoder(input_dim, encoding_dim)

        # Select non-stock-out instances for training the autoencoder
        non_stock_out_indices = df[df['StockOut'] == 0].index
        X_train_non_stock_out = X_train[y_train == 0]
        train_autoencoder(autoencoder, X_train_non_stock_out)

        # Use the trained autoencoder to reconstruct the input data
        reconstructed_data = reconstruct_data(autoencoder, X_scaled)

        # Calculate the reconstruction error for each sample
        reconstruction_error = calculate_reconstruction_error(X_scaled, reconstructed_data)

        # Set a threshold for anomaly detection and detect stock-outs
        threshold = 2.0
        detected_stock_outs = detect_stock_outs_with_threshold(df, reconstruction_error, threshold)

        # Display detected stock-outs
        st.write("Detected Stock-Outs:")
        st.dataframe(detected_stock_outs)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
