# client_product_recsys.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This program contain the core functions of the recommender system, besides the
# routine that recommends products to clients within the webapp's framework.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.cluster import KMeans
import numpy as np
from tabulate import tabulate

# Function to load and process the recommender system's dataframe
def load_data(file_path):
    # Read the data from the CSV file located at the specified file_path
    df = pd.read_csv(file_path)

    # Group the DataFrame by 'Product_Name' and replace 'Product_ID' with the first occurrence in each group
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')

    # Define substrings and brands to be excluded from the data
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']

    # Remove rows where 'Product_Name' contains any of the excluded substrings and 'Brand_Name' is not in excluded brands
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]

    # Return the filtered DataFrame
    return df

# Define the interaction matrix that keeps tracks of the interaction's strength
# between clients and products. Each entry of this matrix measures the number
# of interactions of an item by a client.
def create_interaction_matrix(data_frame):
    # Group the DataFrame by 'Client_ID' and 'Product_ID', counting the occurrences of each pair
    # The result is a DataFrame where each row represents a client and each column represents a product,
    # with the values indicating the number of interactions (purchases, views, etc.) between the client and the product.
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size()

    # Unstack the grouped DataFrame to reshape it into a matrix format
    # Missing values (i.e., no interaction between a client and a product) are filled with 0.
    return interaction_matrix.unstack(fill_value=0)

# Function that constructs the autoencoder's neural network
def build_autoencoder_model(input_size):
    # Initialize a Sequential model
    model = Sequential()

    # Add the first fully connected layer with 128 neurons and ReLU activation function
    # The input shape is specified as (input_size,), where input_size is the number of features
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))

    # Apply dropout regularization with a rate of 0.5 to prevent overfitting
    model.add(Dropout(0.5))

    # Add another fully connected layer with 64 neurons and ReLU activation function
    model.add(Dense(64, activation='relu'))

    # Add another fully connected layer with 128 neurons and ReLU activation function
    model.add(Dense(128, activation='relu'))

    # Apply dropout regularization with a rate of 0.5 to prevent overfitting
    model.add(Dropout(0.5))

    # Add the output layer with the same number of neurons as the input size and sigmoid activation function
    model.add(Dense(input_size, activation='sigmoid'))

    # Compile the model with the Adam optimizer and mean squared error loss function
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the compiled model
    return model

# Perform the autoenconder's training
def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    # Train the autoencoder model on the training data (X_train)
    # The autoencoder is trained to reconstruct the input data, so X_train is used as both input and target output
    # Training is performed for the specified number of epochs using mini-batches of size batch_size
    # The data is shuffled during training to prevent the model from memorizing the order of the data
    # Validation data (X_test) is used to monitor the model's performance on unseen data during training
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

# Return the mode's embedding
def get_embeddings(encoder_model, X):
    # Generate embeddings using the encoder model
    return encoder_model.predict(X)

# Routine that recommends products to clients, as usual in a recommender system.
# Notice that in product_client_recsys.py, we implement a reciprocal function,
# that recommends clients to products instead.
def recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    # Get the index of the specified client_id in the interaction_matrix index
    client_index = interaction_matrix.index.get_loc(client_id)

    # Extract the embedding vector of the specified client
    client_embedding = embeddings[client_index]

    # Calculate cosine similarity between client's embedding and all product embeddings
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))

    # Filter out invalid indices (products already purchased)
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]

    # Get indices of top k similar products
    product_indices = similarities.argsort()[::-1][valid_indices][:k]

    # Get the products already purchased by the client
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values

    # Select top k recommended products that the client has not already purchased
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]

    # Return the list of recommended product IDs
    return recommended_products
