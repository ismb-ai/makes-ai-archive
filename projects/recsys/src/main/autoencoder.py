# Autoencoder Recommender System

# Igor Mol <igor.mol@makes.ai>

# Autoencoders find utility in recommender systems by addressing challenges
# related to high-dimensional and sparse user-item interaction matrices.
# These matrices represent the interactions between users and items, where most
# entries are missing due to users not interacting with the majority of items.
# By using autoencoders, the system learns a lower-dimensional representation
# (embedding) of users and items. This learned representation captures latent
# features and patterns in the data, enabling the model to reconstruct the
# original input accurately. The encoder, responsible for dimensionality
# reduction, helps in revealing meaningful user and item embeddings. The use of
# autoencoders enhances collaborative filtering by mitigating the sparsity
# issue and capturing intricate relationships between users and items. The model
# learns to recommend items based on similarities in the learned latent space,
# providing more accurate and personalized suggestions to users.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import recall_score
from tensorflow.keras.layers import Dropout

# The function load_data load data from a CSV file into a DataFrame, perform 
# data normalization, and exclude lines based on specified conditions.

# Input:
# - file_path: Path to the CSV file containing the data.

# Output:
# - df: DataFrame with loaded and processed data.

def load_data(file_path):
    # Load the CSV file into a DataFrame.
    df = pd.read_csv(file_path)

    # Normalize the data by ensuring the same Product_Name has the same Product_ID.
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')

    # Exclude lines based on specified conditions.
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV',
                           'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']

    # Exclude lines where "Product_Name" contains specified substrings.
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]

    # Exclude lines where "Brand_Name" contains the substring "DEERE".
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]

    return df


# The function create_interaction_matrix creates an user-item interaction matrix
# from a DataFrame received as the function's argument.

# Input:
# - data_frame: DataFrame containing user-item interaction data.

# Output:
# - User-item interaction matrix where rows represent clients, columns represent
#   products, and values represent the interaction frequency. Missing values are
#   filled with 0.

def create_interaction_matrix(data_frame):
    # Create a user-item interaction matrix using the size of each group.
    # Unstack the matrix and fill missing values with 0.
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)


# The function build_autoencoder_model builds an autoencoder model for dimensio-
# nality reduction and feature learning.

# Input:
# - input_size: Number of features in the input data.

# Output:
# - Autoencoder model with encoder and decoder layers for learning a lower-dimensional
#   representation of the input data.

def build_autoencoder_model(input_size):
    # Create a Sequential model for the autoencoder.
    model = Sequential()
    
    # Encoder layers
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(Dense(64, activation='relu'))
    
    # Decoder layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(Dense(input_size, activation='sigmoid'))
    
    # Compile the model with Adam optimizer and mean squared error loss.
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# The function train_autoencoder trains an autoencoder model using the training
# data received as the function's argument.

# Input:
# - autoencoder_model: Autoencoder model to be trained.
# - X_train: Training data for the autoencoder.
# - X_test: Testing data for validation during training.
# - epochs: Number of training epochs (default: 10).
# - batch_size: Size of batches for training (default: 32).

# Note: The autoencoder is trained to reconstruct its input, so both X_train and X_test
# are used as both input and target during training.

def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    # Fit the autoencoder model to the training data.
    # Shuffle the data and use X_test for validation during training.
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

# Get matrix embeddings using the encoder model.

# Input:
# - encoder_model: Trained encoder model.
# - X: Input data for which embeddings are to be obtained.

# Output:
# - Matrix embeddings obtained by predicting the output using the encoder model.
def get_embeddings(encoder_model, X):
    return encoder_model.predict(X)

# The next function recommends products for a given client using autoencoder
# embeddings.

# Input:
# - embeddings: Embeddings matrix obtained from the autoencoder.
# - client_id: Identifier of the target client for whom recommendations are generated.
# - interaction_matrix: User-item interaction matrix.
# - k: Number of products to recommend (default: 5).

# Output:
# - List of recommended product IDs for the specified client.

def recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    # Get the index of the target client in the interaction matrix.
    client_index = interaction_matrix.index.get_loc(client_id)
    client_embedding = embeddings[client_index]

    # Calculate cosine similarity between the client embedding and all other embeddings.
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))

    # Get indices of k highest similarities (excluding the client itself) within valid range.
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]

    # Get recommended product IDs excluding those already purchased by the client.
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]

    return recommended_products


# Compute Recall at K for a given client.

# Input:
# - actual_products: List of actual products purchased by the client.
# - recommended_products: List of products recommended to the client.
# - k: Number of recommended products to consider.

# Output:
# - Recall at K value for the specified client.

def calculate_recall_at_k(actual_products, recommended_products, k):
    # Calculate Recall at K using the formula:
    # Recall at K = (Number of Recommended Items in Actual Purchases) / (Total Number of Actual Purchases)
    
    # Determine the client's actual purchases from the interaction matrix.
    client_purchases = interaction_matrix.loc[client_id]
    
    # Calculate and return Recall at K.
    return len(set(recommended_products)) / len(client_purchases[client_purchases > 0])

# Main routine:

def main():
    # Load the CSV file into a DataFrame
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)

    # Create a user-item interaction matrix
    interaction_matrix = create_interaction_matrix(df)

    # Split data into train and test sets
    train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)

    # Build and train the autoencoder
    autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
    train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)

    # Extract embeddings from the trained autoencoder
    encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)

    # Recommend top 5 products for all clients using autoencoder
    top_recommendations_for_all_clients = {}
    recall_at_k_values = []

    for client_id in interaction_matrix.index:
        recommended_products = recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=15)
        actual_products = interaction_matrix.loc[client_id][interaction_matrix.loc[client_id] > 0].index.values

        top_recommendations_for_all_clients[int(client_id)] = {"recommended_products": recommended_products}

        recall_at_k = calculate_recall_at_k(actual_products, recommended_products, k=5)
        recall_at_k_values.append(recall_at_k)

    # Save the results to a text file
    output_file_path = "/Volumes/Untitled/cpp-recsys-products-autoencoder.txt"
    with open(output_file_path, 'w') as output_file:
        for client_id, recommendations_info in top_recommendations_for_all_clients.items():
            recommended_products = recommendations_info["recommended_products"]

            if 0 <= int(client_id) < len(recall_at_k_values):
                output_file.write(f"Top recommendations for Client {client_id} (Recall at K = 5: {recall_at_k_values[int(client_id)]:}):\n")
            else:
                output_file.write(f"Top recommendations for Client {client_id} (Recall at K = 5: N/A):\n")

            output_file.write(f"{'Product ID':<15} {'Product Name':<30} {'Brand':<20}\n")
            output_file.write("-" * 80 + "\n")

            for product_id in recommended_products:
                product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                output_file.write(f"{product_id:<15} {product_name:<30} {brand:<20}\n")

            output_file.write("\n")

if __name__ == "__main__":
    main()
