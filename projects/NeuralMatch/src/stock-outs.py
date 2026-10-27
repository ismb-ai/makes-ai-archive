import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load the data
file_path = "/Users/igormol/Downloads/online_retail_II.csv"
df = pd.read_csv(file_path)

# Drop rows with missing values for simplicity
df = df.dropna()

# Select relevant columns for training the autoencoder
columns_for_training = ['Quantity', 'Price']

# Extract features (X) and normalize data
X = df[columns_for_training].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Identify stock-out situations (e.g., Quantity = 0)
stock_outs = df[df['Quantity'] == 0]

# Create a binary label indicating stock-out (1) or not (0)
df['StockOut'] = np.where(df['Quantity'] == 0, 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['StockOut'].values, test_size=0.2, random_state=42)

# Define the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 2  # Number of neurons in the encoding layer

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer="adam", loss="mse")

# Train the autoencoder on non-stock-out instances
non_stock_out_indices = df[df['StockOut'] == 0].index
X_train_non_stock_out = X_train[y_train == 0]
autoencoder.fit(X_train_non_stock_out, X_train_non_stock_out, epochs=50, batch_size=32, shuffle=True)

# Use the trained autoencoder to reconstruct the input data
reconstructed_data = autoencoder.predict(X_scaled)

# Calculate the reconstruction error for each sample
reconstruction_error = np.mean(np.abs(X_scaled - reconstructed_data), axis=1)

# Set a threshold for anomaly detection (you may adjust this threshold based on your data)
threshold = 2.0

# Identify stock-outs based on the reconstruction error
predicted_stock_outs = df[reconstruction_error > threshold]

# Display detected stock-outs
print("Detected Stock-Outs:")
print(predicted_stock_outs)

