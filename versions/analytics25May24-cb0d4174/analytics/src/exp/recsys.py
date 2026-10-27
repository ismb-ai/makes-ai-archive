# recsys.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains functions concerning our recommender system's engine.
# An Autoenconder neural network is used to recommend clients to products and,
# conversely, products to clients. The kNN algorithm is employed for performing
# client and product similarity analysis.

# In sec. 1, we declare the functions that initializes and preprocess the auto-
# encoder architecture and the interaction matrix. In sec. 2, we define the
# routines of the recommender system. In sec. 3, we construct the similarity
# matrix and the corresponding functions of the latter.

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Section 1: Initialization and preprocessing routines

# Creates an interaction matrix from a data-frame containing client-product interactions.
def create_interaction_matrix(data_frame):
    # Group the DataFrame by 'Client_ID' and 'Product_ID', and count the occurrences of each unique pair.
    # The result is reshaped into a matrix where rows represent clients and columns represent products.
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

    # Return the interaction matrix.
    return interaction_matrix

# Function to build our autoenconder architecture
def build_autoencoder_model(input_size):
    # Initialize a Sequential model using a list of layers.
    model = Sequential([
        # Add a fully connected (Dense) layer with 128 neurons and ReLU activation function.
        # The input shape is determined by the input_size parameter.
        Dense(128, activation='relu', input_shape=(input_size,)),

        # Add a Dropout layer with a dropout rate of 0.5 to help prevent overfitting.
        Dropout(0.5),

        # Add another fully connected (Dense) layer with 64 neurons and ReLU activation function.
        Dense(64, activation='relu'),

        # Add another fully connected (Dense) layer with 128 neurons and ReLU activation function.
        Dense(128, activation='relu'),

        # Add another Dropout layer with a dropout rate of 0.5.
        Dropout(0.5),

        # Add a fully connected (Dense) layer with the same size as the input.
        # Use the sigmoid activation function to squash the output between 0 and 1.
        Dense(input_size, activation='sigmoid')
    ])

    # Compile the model using the Adam optimizer and mean squared error loss function.
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the compiled model.
    return model

# Trains an autoencoder model using the provided training data and validation data.
def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    # The number of training iterations is determined by the 'epochs' parameter.
    # The data is divided into batches of size 'batch_size' during training.
    # The 'shuffle' parameter shuffles the training data before each epoch to prevent the model from memorizing the order.
    # Validation data (X_test) is provided to evaluate the model's performance during training.
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

# Extracts embeddings (latent representations) from input data using a pre-trained encoder model.
def get_embeddings(encoder_model, X):
    # Use the encoder model to predict the latent representations (embeddings) of the input data X.
    embeddings = encoder_model.predict(X)
    # Return the embeddings.
    return embeddings

# Section 2: Recommender system for products and Clients

# Recommends products to a client based on their embeddings and interaction history.
def recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    # Get the index of the client in the interaction matrix.
    client_index = interaction_matrix.index.get_loc(client_id)
    # Retrieve the embedding of the client from the embeddings array.
    client_embedding = embeddings[client_index]

    # Calculate the cosine similarity between the client's embedding and all product embeddings.
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))

    # Find the indices of products sorted by their similarity to the client's embedding.
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]

    # Get the client's purchase history.
    client_purchases = interaction_matrix.loc[client_id]
    # Extract the IDs of products the client has already purchased.
    purchased_product_ids = client_purchases[client_purchases > 0].index.values

    # Select top k recommended products that the client has not already purchased.
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]

    # Return the recommended product IDs.
    return recommended_products

# Recommends clients to a given product based on their embeddings and interaction history.
def recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5):
    # Get the index of the product in the interaction matrix.
    product_index = interaction_matrix.columns.get_loc(product_id)
    # Extract the embedding of the product from the embeddings array.
    product_embedding = embeddings[:, product_index]

    # Transpose the embeddings matrix so that each row represents a client embedding.
    transposed_embeddings = embeddings.T

    # Calculate the similarity between the product embedding and each client embedding.
    recommendations = []
    for client_embedding in transposed_embeddings:
        similarity = np.dot(client_embedding, product_embedding)
        recommendations.append(similarity)

    # Convert recommendations list to numpy array for sorting.
    recommendations = np.array(recommendations)

    # Find the indices of clients sorted by their similarity to the product's embedding.
    valid_indices = np.where(recommendations.argsort()[::-1] < len(interaction_matrix.columns))[0]
    client_indices = recommendations.argsort()[::-1][valid_indices][:k]

    # Retrieve the IDs of recommended clients from the interaction matrix.
    recommended_clients = interaction_matrix.index[client_indices]

    # Return the recommended client IDs.
    return recommended_clients

# Section 3: Similarity Analysis

# Calculates the similarity matrix between products based on the interaction matrix.
def instantiate_similarity_matrix(interaction_matrix):
    similarity_matrix = cosine_similarity(interaction_matrix)

    return similarity_matrix

# Retrieves similar products to a given product based on a pre-calculated similarity matrix.
def get_similar_products(similarity_matrix, interaction_matrix, product_id, k=5):
    # Get the index of the product in the interaction matrix.
    product_index = interaction_matrix.columns.get_loc(product_id)

    # Sort the similarity scores of the given product with all other products in descending order
    # and select the top k+1 (excluding itself).
    similar_indices = similarity_matrix[product_index].argsort()[:-k-1:-1]

    # Retrieve the IDs of similar products from the interaction matrix and convert them to a list.
    similar_products = interaction_matrix.columns[similar_indices].tolist()[1:]

    # Return the list of similar product IDs.
    return similar_products

# Recommends products for a given product based on similarity scores calculated using a similarity matrix.
def recommend_products_for_product(similarity_matrix, interaction_matrix, product_id, k=5):
    # Get a list of similar products to the given product using the provided similarity matrix.
    similar_products = get_similar_products(similarity_matrix, interaction_matrix, product_id, k)

    # Calculate the recall at k, which measures the proportion of relevant items (purchased by users) among recommended items.
    relevant_items = interaction_matrix[interaction_matrix[product_id] > 0].index.tolist()
    recommended_relevant = sum(1 for item in similar_products if item in relevant_items)
    recall_at_k = recommended_relevant / len(relevant_items) if len(relevant_items) > 0 else 0

    # Return the list of similar products and the recall at k.
    return similar_products, recall_at_k

# Retrieves similar clients to a given client based on a pre-calculated similarity matrix.
def get_similar_clients(similarity_matrix, interaction_matrix, client_id, k=5):
    # Find the row index of the given client_id in the interaction_matrix
    client_index = interaction_matrix.index.get_loc(client_id)

    # Ensure the similarity matrix has the correct dimensions (it must be square)
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square")

    # Ensure the client index is within the bounds of the similarity matrix
    if client_index >= similarity_matrix.shape[0]:
        raise IndexError(f"Client index {client_index} is out of bounds for similarity matrix with size {similarity_matrix.shape[0]}")

    # Sort indices of the similarity scores for the client in descending order
    sorted_indices = similarity_matrix[client_index].argsort()[::-1]

    # If there are not enough clients, select all except the client itself (index 0)
    # Otherwise, select the top k similar clients (excluding the client itself)
    if len(sorted_indices) <= k:
        similar_indices = sorted_indices[1:]
    else:
        similar_indices = sorted_indices[1:k+1]

    # Retrieve the client IDs corresponding to the similar indices
    similar_clients = interaction_matrix.index[similar_indices].tolist()

    # Return the list of similar client IDs
    return similar_clients

# Recommends clients to a given client based on similarity scores calculated using a similarity matrix.
def recommend_clients_for_client(similarity_matrix, interaction_matrix, client_id, k=5):
    # Get the list of similar clients using the get_similar_clients function
    similar_clients = get_similar_clients(similarity_matrix, interaction_matrix, client_id, k)

    # Find the items that the given client has interacted with (non-zero interactions)
    relevant_items = interaction_matrix.loc[client_id][interaction_matrix.loc[client_id] > 0].index.tolist()

    # Count how many of the similar clients have interacted with the same items
    recommended_relevant = sum(1 for item in similar_clients if item in relevant_items)

    # The total number of relevant items for the given client
    recall_at_k = len(relevant_items)

    # Return the list of similar clients and the recall value
    return similar_clients, recall_at_k

# Generates client-to-client recommendations based on the interaction matrix.
def client_to_client_recommender(interaction_matrix):
    # Instantiate the similarity matrix based on the interaction matrix
    similarity_matrix = instantiate_similarity_matrix(interaction_matrix)

    # Initialize dictionaries to store recommendations and recall values for each client
    all_recommendations = {}
    all_recall_at_k = {}

    # Iterate through each client in the interaction matrix
    for client_id in interaction_matrix.index:
        try:
            # Generate recommendations for the current client
            recommended_clients, recall_at_k = recommend_clients_for_client(similarity_matrix, interaction_matrix, client_id, k=5)

            # Store the recommendations and recall value for the current client
            all_recommendations[client_id] = recommended_clients
            all_recall_at_k[client_id] = recall_at_k
        except IndexError as e:
            # Print an error message if an IndexError occurs and continue with the next client
            print(f"Error processing client {client_id}: {e}")
            continue

    # Initialize a list to store the final result table
    result_table = []

    # Populate the result table with client IDs, their recommendations, and recall values
    for client_id, recommended_clients in all_recommendations.items():
        recall = all_recall_at_k[client_id]
        result_table.append([client_id, recommended_clients, recall])

    # Return the final result table
    return result_table
