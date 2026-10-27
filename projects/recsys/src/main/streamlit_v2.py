# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Function to load data from a CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to create a product-client interaction matrix from the DataFrame
def create_interaction_matrix(data_frame):
    return data_frame.groupby(['Product_ID', 'Client_ID']).size().unstack(fill_value=0)

# Function to instantiate and fit a NearestNeighbors model using the cosine similarity metric
def instantiate_knn_model(interaction_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(interaction_matrix.values)
    return knn_model

# Function to get k nearest neighbors for a given product in the interaction matrix
def get_nearest_neighbors(interaction_matrix, knn_model, product_id, k=5):
    product_index = interaction_matrix.index.get_loc(product_id)
    distances, indices = knn_model.kneighbors([interaction_matrix.iloc[product_index].values], n_neighbors=k+1)
    neighbor_ids = interaction_matrix.index[indices.flatten()[1:]]
    return neighbor_ids, distances.flatten()[1:]

# Function to recommend clients for a given product based on nearest neighbors
def recommend_clients_for_product(interaction_matrix, knn_model, product_id, k=5):
    neighbor_ids, distances = get_nearest_neighbors(interaction_matrix, knn_model, product_id, k)
    product_clients = interaction_matrix.loc[product_id]
    
    recommended_clients = []
    predicted_scores = []
    for neighbor_id, distance in zip(neighbor_ids, distances):
        neighbor_clients = interaction_matrix.loc[neighbor_id]
        recommended_clients.extend(neighbor_clients[neighbor_clients > 0].index)
        predicted_scores.append(1 - distance)  # Convert cosine distance to similarity
        
    return list(set(recommended_clients) - set(product_clients[product_clients > 0].index))[:k], predicted_scores[:k]

# Function to recommend top clients for all products in the interaction matrix
def recommend_top_clients_for_all_products(interaction_matrix, knn_model, k=5):
    all_products = interaction_matrix.index
    all_recommendations = {}

    for product_id in all_products:
        recommended_clients, predicted_scores = recommend_clients_for_product(interaction_matrix, knn_model, product_id, k)
        
        if recommended_clients:
            all_recommendations[product_id] = {"recommended_clients": recommended_clients, "predicted_scores": predicted_scores}

    return all_recommendations

# Main function to create the Streamlit web app
def main():
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Create a product-client interaction matrix
        interaction_matrix = create_interaction_matrix(df)

        # Instantiate the NearestNeighbors model and fit it to the interaction matrix
        knn_model = instantiate_knn_model(interaction_matrix)

        # User input for the number of clients to recommend
        num_clients_to_recommend = st.slider("Select the number of clients to recommend", 1, 10, 5)

        # Recommend top clients for all products
        top_recommendations_for_all_products = recommend_top_clients_for_all_products(interaction_matrix, knn_model, k=num_clients_to_recommend)

        # Streamlit web app
        st.title("Product Recommender System")
        product_id = st.selectbox("Select a product:", list(top_recommendations_for_all_products.keys()))

        st.header(f"Top {num_clients_to_recommend} recommended clients for Product {product_id}:")

        # Display recommended clients and their predicted scores
        recommended_clients = top_recommendations_for_all_products[product_id]["recommended_clients"]
        predicted_scores = top_recommendations_for_all_products[product_id]["predicted_scores"]

        for client_id, score in zip(recommended_clients, predicted_scores):
            st.write(f"Client ID: {client_id}, Predicted Score: {score:.4f}")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
