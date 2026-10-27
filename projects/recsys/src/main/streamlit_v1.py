# Hybrid k-NN / Auto-encoder Web App Recommender System
#
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
import streamlit as st

# Loads and preprocesses data from a CSV file.
#
# Parameters:
# - file_path: The path to the CSV file.
#
# Returns:
# - df: The preprocessed DataFrame.
def load_data(file_path):
    # Read data from the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group by 'Product_Name' and assign the first 'Product_ID' within each group
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')
    
    # Define excluded substrings and brands to filter out from the data
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']

    # Filter out rows containing excluded substrings in 'Product_Name' or excluded brands in 'Brand_Name'
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]

    # Return the preprocessed DataFrame
    return df

# Create an interaction matrix from a DataFrame.
#
# Parameters:
# - data_frame: The input DataFrame containing interaction data.
#
# Returns:
# - interaction_matrix: A matrix representing interactions between clients and 
# products.
def create_interaction_matrix(data_frame):   
    # Group by 'Client_ID' and 'Product_ID' and count interactions
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size()
    
    # Unstack the result, filling missing values with 0
    interaction_matrix = interaction_matrix.unstack(fill_value=0)
    
    # Return the resulting interaction matrix
    return interaction_matrix

# Build an autoencoder model.
#
# Parameters:
# - input_size: The size of the input data.
#
# Returns:
# - model: The compiled autoencoder model.
def build_autoencoder_model(input_size):
    # Create a Sequential model
    model = Sequential()

    # Add a dense layer with 128 neurons, ReLU activation, and specified input shape
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))

    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Add another dense layer with 64 neurons and ReLU activation
    model.add(Dense(64, activation='relu'))

    # Add another dense layer with 128 neurons and ReLU activation
    model.add(Dense(128, activation='relu'))

    # Add another dropout layer
    model.add(Dropout(0.5))

    # Add a dense output layer with input_size neurons and sigmoid activation
    model.add(Dense(input_size, activation='sigmoid'))

    # Compile the model using Adam optimizer and Mean Squared Error (MSE) loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the compiled autoencoder model
    return model

# Train the provided autoencoder model on the input data.
#
# Parameters:
# - autoencoder_model: The autoencoder model to be trained.
# - X_train: The input data for training.
# - X_test: The input data for validation.
# - epochs: Number of training epochs (default is 10).
# - batch_size: Size of batches used for training (default is 32).
def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):

    autoencoder_model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test)
    )

# Computes a similarity matrix using k-Nearest Neighbors (kNN) approach.
#
# Parameters:
# - interaction_matrix: The matrix representing interactions between clients and products.
# - k: The number of neighbors to consider (default is 5).
#
# Returns:
# - similarity_matrix: The computed similarity matrix.
def compute_similarity_matrix(interaction_matrix, k=5):
    # Create a kNN model with cosine similarity and brute-force algorithm
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    
    # Fit the kNN model to the values of the interaction matrix
    knn.fit(interaction_matrix.values)
    
    # Compute the similarity matrix using the kNN distances
    similarity_matrix = 1 - knn.kneighbors_graph(mode='distance').toarray()
    
    # Fill the diagonal with zeros to avoid self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    
    # Return the computed similarity matrix
    return similarity_matrix

# Obtain embeddings using an encoder model:
def get_embeddings(encoder_model, X):
    # Use the encoder model to obtain embeddings for the input data
    embeddings = encoder_model.predict(X)
    
    # Return the obtained embeddings
    return embeddings
    
# Recommends products for a given client using a hybrid recommendation approach.
# 
# Parameters:
# - embeddings: Embeddings representing clients and products.
# - user_similarity_matrix: Similarity matrix based on user interactions.
# - client_id: The ID of the target client for whom recommendations are made.
# - interaction_matrix: Matrix representing interactions between clients and products.
# - k: Number of recommendations to generate (default is 5).
#
# Returns:
# - recommended_products: List of recommended product IDs.
# - predicted_scores: Predicted scores for the recommended products.
def recommend_products_hybrid(embeddings, user_similarity_matrix, client_id, interaction_matrix, k=5):
    # Obtain the index of the target client in the interaction matrix
    client_index = interaction_matrix.index.get_loc(client_id)
    
    # Extract the embedding for the target client
    client_embedding = embeddings[client_index]

    # Calculate weighted embeddings based on user similarity matrix
    weighted_embeddings = np.dot(user_similarity_matrix[client_index], embeddings)
    
    # Combine the client's embedding with the weighted embeddings
    combined_embeddings = 0.5 * client_embedding + 0.5 * weighted_embeddings

    # Calculate cosine similarities between embeddings
    similarities = np.dot(embeddings, combined_embeddings) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(combined_embeddings))
    
    # Filter out invalid indices and get the indices of top-k similar products
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]

    # Get the client's previous purchases from the interaction matrix
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values

    # Filter out already purchased products and select the top-k recommendations
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]

    # Extract predicted scores for the recommended products
    predicted_scores = similarities[product_indices]

    # Return the recommended products and their predicted scores
    return recommended_products, predicted_scores


# Main function for the Hybrid Recommender System
def main():
    # Set the title for the Streamlit web app
    st.title("Hybrid Recommender System")

    # Allow the user to upload a CSV file
    file_path = st.file_uploader("Upload your CSV file", type=["csv"])

    # Proceed if a file is uploaded
    if file_path is not None:
        # Load data from the uploaded CSV file and create the interaction matrix
        df = load_data(file_path)
        interaction_matrix = create_interaction_matrix(df)

        # Split the interaction matrix into training and testing data
        train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)

        # Build and train the autoencoder model on the training data
        autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
        train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)

        # Extract embeddings using the trained encoder model
        encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
        embeddings = get_embeddings(encoder_model, interaction_matrix.values)

        # Compute user similarity matrix based on embeddings
        user_similarity_matrix = compute_similarity_matrix(interaction_matrix, k=5)

        # Perform a second round of autoencoder training using user similarity-weighted embeddings
        autoencoder_input = np.dot(user_similarity_matrix, embeddings)
        autoencoder_model = build_autoencoder_model(input_size=len(autoencoder_input[0]))
        train_autoencoder(autoencoder_model, autoencoder_input, autoencoder_input, epochs=10, batch_size=32)

        # Obtain final embeddings after the second round of autoencoder training
        embeddings = get_embeddings(encoder_model, interaction_matrix.values)

        # Add a slider to input the number of items to recommend
        num_items_to_recommend = st.slider("Select the number of items to recommend", 1, 20, 5)

        # Generate top recommendations for all clients
        top_recommendations_for_all_clients = {}

        for client_id in interaction_matrix.index:
            recommended_products, predicted_scores = recommend_products_hybrid(
                embeddings, user_similarity_matrix, client_id, interaction_matrix, k=num_items_to_recommend
            )
            top_recommendations_for_all_clients[int(client_id)] = {
                "recommended_products": recommended_products,
                "predicted_scores": predicted_scores
            }

        # Display recommendations for each client
        for client_id, recommendations_info in top_recommendations_for_all_clients.items():
            st.subheader(f"Top recommendations for Client {client_id}:")

            product_ids = recommendations_info['recommended_products']
            product_names = [df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0] for product_id in product_ids]
            brands = [df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0] for product_id in product_ids]
            predicted_scores = recommendations_info['predicted_scores']

            # Ensure consistent lengths of arrays
            min_length = min(len(product_ids), len(product_names), len(brands), len(predicted_scores))
            product_ids = product_ids[:min_length]
            product_names = product_names[:min_length]
            brands = brands[:min_length]
            predicted_scores = predicted_scores[:min_length]

            # Create a DataFrame for displaying recommendations
            recommendations_df = pd.DataFrame({
                'Product ID': product_ids,
                'Product Name': product_names,
                'Brand': brands,
                'Predicted Score': predicted_scores
            })

            # Display the recommendations in a table
            st.table(recommendations_df)

if __name__ == "__main__":
    main()
