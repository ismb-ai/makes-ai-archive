# product_client_recsys.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This file contains the function that recommends clients to products, i.e.,
# performs the reciprocal task of a standard recommender system. Take notice
# that the core functions of recommender system's engine are contained in the
# file client_product_recsys.py.

import pandas as pd
import numpy as np

# Function to recommend clients to products
def recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5):
    # Get the index of the specified product_id in the interaction_matrix columns
    product_index = interaction_matrix.columns.get_loc(product_id)

    # Extract the embedding vector of the specified product
    product_embedding = embeddings[:, product_index]

    # Transpose the embeddings matrix to iterate over client embeddings
    transposed_embeddings = embeddings.T

    # Initialize an empty list to store recommendation scores for each client
    recommendations = []

    # Iterate over each client embedding
    for client_embedding in transposed_embeddings:
        # Calculate the cosine similarity between the client embedding and the product embedding
        similarity = np.dot(client_embedding, product_embedding)

        # Append the similarity score to the recommendations list
        recommendations.append(similarity)

    # Convert the recommendations list to a numpy array
    recommendations = np.array(recommendations)

    # Find the indices of the top k recommendations
    valid_indices = np.where(recommendations.argsort()[::-1] < len(interaction_matrix.columns))[0]
    client_indices = recommendations.argsort()[::-1][valid_indices][:k]

    # Get the client IDs corresponding to the top k recommendations
    recommended_clients = interaction_matrix.index[client_indices]

    # Return the list of recommended client IDs
    return recommended_clients
