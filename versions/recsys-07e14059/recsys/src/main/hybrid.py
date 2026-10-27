# Hybrid Recommendation System
#
# Igor Mol <igor.mol@makes.ai>
# 
# Hybrid recommendation is a technique that combines the strengths of two distinct 
# recommendation systems, k-NN (k-Nearest Neighbors) and autoencoder. The goal is 
# to enhance the precision and diversity of product recommendations for clients.
# 
# The process begins with loading and preprocessing sales data from a CSV file. 
# This data is then transformed into a user-item interaction matrix, where rows 
# represent clients, columns represent products, and the values denote interaction 
# counts (e.g., purchases, views).
# 
# Subsequently, an autoencoder model is constructed and trained using the interaction 
# matrix. The autoencoder's primary purpose is dimensionality reduction and feature 
# learning, extracting meaningful embeddings that capture underlying patterns in 
# user-product interactions.
# 
# Concurrently, a user-user similarity matrix is computed using k-NN. This matrix 
# leverages cosine similarity to identify clients with similar preferences. The 
# number of neighbors (k) in k-NN influences the level of similarity considered.
# 
# The autoencoder embeddings and the user-user similarity matrix are then combined 
# to generate a hybrid recommendation system. For each client, the system computes 
# a weighted sum of their embeddings and those of their similar users. This hybrid 
# approach aims to balance individual preferences with collaborative filtering.
# 
# The final step involves recommending products to each client based on their hybrid 
# embeddings. These recommendations consider both the reconstructed autoencoder 
# embeddings and the collaborative filtering aspect from the user-user similarity 
# matrix. Predicted scores quantify the likelihood of client preference.
# 
# The recommendations, along with product details and scores, are written to an 
# output file for further analysis or user presentation. The result is a personalized 
# and diverse set of product recommendations, combining the strengths of autoencoder 
# feature learning and collaborative filtering through k-NN.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout

# The load_data function loads data from a CSV file into a pandas DataFrame,
# performs data normalization on the 'Product_ID' column by grouping, and excludes
# lines from the DataFrame based on specified conditions in 'Product_Name' and 
# 'Brand_Name'.
#
# Input:
#   - file_path: A string representing the path to the CSV file containing the 
# data.
#
# Returns:
#   - df: A pandas DataFrame with loaded data, normalized 'Product_ID', and 
#         excluded rows based on specified conditions in 'Product_Name' and 
#         'Brand_Name'.
    
def load_data(file_path):
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Normalize Product_ID by grouping by Product_Name and taking the first value for each group
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')

    # Define a list of substrings to exclude from Product_Name
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    
    # Define a list of brands to exclude from Brand_Name
    excluded_brands = ['DEERE']

    # Exclude rows where Product_Name contains any of the specified substrings (case-insensitive)
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    
    # Exclude rows where Brand_Name contains any of the specified brands (case-insensitive)
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]

    # Return the processed DataFrame
    return df


# The function create_interaction_matrix aims to create a user-item interaction
# matrix from a DataFrame.
# 
# Input:
#   - data_frame: A pandas DataFrame containing user-item interaction data.
# 
# Returns:
#   - interaction_matrix: A matrix where rows represent clients, columns represent
#     products, and the values denote the interaction count between clients and products.

def create_interaction_matrix(data_frame):
    # Group the DataFrame by 'Client_ID' and 'Product_ID', then calculate the size of each group
    # This represents the number of interactions (purchases, views, etc.) between each client and product
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size()
    
    # Unstack the grouped data to create a matrix where rows are clients, columns are products,
    # and the values represent the count of interactions
    # fill_value=0 is used to fill NaN values with 0 in the resulting matrix
    interaction_matrix = interaction_matrix.unstack(fill_value=0)
    
    return interaction_matrix

# The function build_autoencoder_model builds an autoencoder model for dimensio-
# nality reduction and feature learning.
#
# Input:
#   - input_size: An integer representing the number of features in the input data.
#
# Returns:
#   - model: An autoencoder model configured for dimensionality reduction and feature learning.

def build_autoencoder_model(input_size):
    # Create a Sequential model
    model = Sequential()
    
    # Add a dense layer with 128 neurons, ReLU activation function, and input shape specified by input_size
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))
    
    # Add dropout layer with a dropout rate of 0.5 to prevent overfitting
    model.add(Dropout(0.5))
    
    # Add another dense layer with 64 neurons and ReLU activation function
    model.add(Dense(64, activation='relu'))
    
    # Add another dense layer with 128 neurons and ReLU activation function
    model.add(Dense(128, activation='relu'))
    
    # Add dropout layer with a dropout rate of 0.5
    model.add(Dropout(0.5))
    
    # Add the output layer with dense layer and sigmoid activation function for reconstruction
    model.add(Dense(input_size, activation='sigmoid'))
    
    # Compile the model using Adam optimizer and mean squared error as the loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# The following function is responsible for performing the training of an auto-
# encoder model.
#
# Input:
#   - autoencoder_model: An autoencoder model to be trained.
#   - X_train: The training data for the autoencoder.
#   - X_test: The testing/validation data for the autoencoder.
#   - epochs: An integer representing the number of training epochs (default is 10).
#   - batch_size: An integer representing the batch size for training (default is 32).

def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    # Train the autoencoder model using the provided training data
    autoencoder_model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test)
    )

# Function to compute user-user similarity matrix using k-NN.
#
# Input:
#   - interaction_matrix: A user-item interaction matrix where rows represent clients,
#                         columns represent products, and values denote interaction counts.
#   - k: An integer representing the number of neighbors to consider for k-NN (default is 5).
#
# Returns:
#   - similarity_matrix: A matrix representing the cosine similarity between users.

def compute_similarity_matrix(interaction_matrix, k=5):
    # Create a k-NN model with cosine similarity and brute-force algorithm
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    
    # Fit the k-NN model using the user-item interaction matrix
    knn.fit(interaction_matrix.values)

    # Compute the cosine similarity matrix based on the k-NN model
    similarity_matrix = 1 - knn.kneighbors_graph(mode='distance').toarray()
    
    # Set diagonal elements to 0 to avoid self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

# Function to get matrix embeddings using the encoder model.
#
# Input:
#   - encoder_model: The encoder portion of an autoencoder model.
#   - X: The input data for which embeddings are to be generated.
#
# Returns:
#   - embeddings: Matrix embeddings generated by the encoder model.

def get_embeddings(encoder_model, X):
    # Use the encoder model to generate embeddings for the input data
    embeddings = encoder_model.predict(X)
    return embeddings

# Function to recommend products for a given client using hybrid embeddings.
#
# Input:
#   - embeddings: Matrix embeddings for clients and products.
#   - user_similarity_matrix: Matrix representing the similarity between users.
#   - client_id: The ID of the target client for whom recommendations are generated.
#   - interaction_matrix: A user-item interaction matrix with purchase history.
#   - k: An integer representing the number of products to recommend (default is 5).
#
# Returns:
#   - recommended_products: A list of recommended product IDs for the target client.
#   - predicted_scores: Array of similarity scores corresponding to the recommended products.

def recommend_products_hybrid(embeddings, user_similarity_matrix, client_id, interaction_matrix, k=5):
    # Get the index of the target client in the interaction matrix
    client_index = interaction_matrix.index.get_loc(client_id)
    
    # Extract the embedding for the target client
    client_embedding = embeddings[client_index]

    # Compute weighted embeddings based on user similarity matrix
    weighted_embeddings = np.dot(user_similarity_matrix[client_index], embeddings)
    
    # Combine the client's embedding with the weighted embeddings
    combined_embeddings = 0.5 * client_embedding + 0.5 * weighted_embeddings

    # Compute cosine similarities between combined_embeddings and all other product embeddings
    similarities = np.dot(embeddings, combined_embeddings) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(combined_embeddings))

    # Filter out invalid indices and get the top-k product indices
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]

    # Get the client's purchased product IDs
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    
    # Filter recommended products that the client has already purchased
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]

    # Get the predicted similarity scores for the recommended products
    predicted_scores = similarities[product_indices]

    return recommended_products, predicted_scores

# Main routine to execute a hybrid recommendation system for clients based on 
# autoencoder embeddings.
#
# Workflow:
# 1. Load data from a CSV file into a pandas DataFrame using the `load_data` 
#    function.
# 2. Create a user-item interaction matrix from the DataFrame using the 
#    `create_interaction_matrix` function.
# 3. Split the interaction matrix into training and testing data using 
#    `train_test_split`.
# 4. Build and train an autoencoder model using the `build_autoencoder_model` 
#    and `train_autoencoder` functions.
# 5. Extract the encoder portion of the trained autoencoder model and generate 
#    embeddings for the entire interaction matrix.
# 6. Compute a user-user similarity matrix based on the generated embeddings 
#    using the `compute_similarity_matrix` function.
# 7. Prepare an autoencoder input by performing a dot product between the user 
#    similarity matrix and the embeddings.
# 8. Rebuild and retrain the autoencoder model using the new autoencoder input.
# 9. Generate updated embeddings for the entire interaction matrix.
# 10. Iterate through each client and obtain hybrid recommendations using the 
#     `recommend_products_hybrid` function.
# 11. Write the top recommendations and predicted scores to an output text file.
#
# Note: This routine assumes the existence of functions such as `load_data`, 
# `create_interaction_matrix`, `build_autoencoder_model`, `train_autoencoder`, 
# `get_embeddings`, `compute_similarity_matrix`, and `recommend_products_hybrid` 
# defined earlier in the code. It also assumes the use of external libraries 
# such as pandas, numpy, and Keras.

def main():
    # File path for the CSV data
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    
    # Load data and create an interaction matrix
    df = load_data(file_path)
    interaction_matrix = create_interaction_matrix(df)

    # Split interaction matrix into training and testing data
    train_data, test_data = train_test_split(interaction_matrix.values, 
                                             test_size=0.2, random_state=42)

    # Build and train the autoencoder model
    autoencoder_model = build_autoencoder_model(
        input_size=len(interaction_matrix.columns)
    )
    train_autoencoder(autoencoder_model, train_data, test_data, 
                      epochs=10, batch_size=32)

    # Extract encoder model and generate embeddings
    encoder_model = Model(inputs=autoencoder_model.input, 
                          outputs=autoencoder_model.get_layer(index=1).output)
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)

    # Compute user-user similarity matrix
    user_similarity_matrix = compute_similarity_matrix(interaction_matrix, k=5)

    # Prepare autoencoder input by performing a dot product
    autoencoder_input = np.dot(user_similarity_matrix, embeddings)

    # Rebuild and retrain the autoencoder model with the new input
    autoencoder_model = build_autoencoder_model(
        input_size=len(autoencoder_input[0])
    )
    train_autoencoder(autoencoder_model, autoencoder_input, autoencoder_input, 
                      epochs=10, batch_size=32)

    # Generate updated embeddings
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)

    # Generate recommendations for all clients
    top_recommendations_for_all_clients = {}

    for client_id in interaction_matrix.index:
        recommended_products, predicted_scores = recommend_products_hybrid(
            embeddings, user_similarity_matrix, client_id, interaction_matrix, k=15
        )
        top_recommendations_for_all_clients[int(client_id)] = {
            "recommended_products": recommended_products, 
            "predicted_scores": predicted_scores
        }

    # Write recommendations to an output text file
    output_file_path = "/Volumes/Untitled/cpp-recsys-products-hybrid.txt"
    with open(output_file_path, 'w') as output_file:
        for client_id, recommendations_info in top_recommendations_for_all_clients.items():
            recommended_products = recommendations_info["recommended_products"]
            predicted_scores = recommendations_info["predicted_scores"]

            output_file.write(f"Top recommendations for Client {client_id}:\n")
            output_file.write(f"{'Product ID':<15} {'Product Name':<30} "
                              f"{'Brand':<20} {'Predicted Score':<20}\n")
            output_file.write("-" * 95 + "\n")

            for product_id, score in zip(recommended_products, predicted_scores):
                product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                output_file.write(f"{product_id:<15} {product_name:<30} "
                                  f"{brand:<20} {score:<20.4f}\n")

            output_file.write("\n")

if __name__ == "__main__":
    main()
