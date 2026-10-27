# ALS Recommender System (Products to Clients)

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import implicit
import numpy as np
from scipy.sparse import coo_matrix

# Load data from CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess DataFrame
def preprocess_data(df):
    exclude_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    df['Product_Name'] = df['Product_Name'].astype(str)
    df['Brand_Name'] = df['Brand_Name'].astype(str)
    return df, exclude_substrings

# Create a user-item interaction matrix
def create_interaction_matrix(df):
    return df.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

# Convert interaction matrix to a sparse matrix
def convert_to_sparse_matrix(interaction_matrix):
    return coo_matrix(interaction_matrix)

# Instantiate and fit the ALS model
def fit_als_model(interaction_matrix_sparse):
    als_model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=50)
    als_model.fit(interaction_matrix_sparse.T)
    return als_model

# Get k-nearest neighbors for a given client using ALS
def get_als_nearest_neighbors(client_id, interaction_matrix, als_model, k=5):
    if client_id not in interaction_matrix.index:
        return []  # Skip clients not present in the interaction matrix

    client_index = interaction_matrix.index.get_loc(client_id)
    
    if client_index >= len(als_model.user_factors):
        return []  # Skip clients with an index out of bounds in ALS model
    
    user_factors = als_model.user_factors[client_index]

    # Calculate cosine similarity between the user and all other users
    cosine_similarities = np.dot(als_model.user_factors, user_factors)

    # Get indices of k users with the highest similarity (excluding the user itself)
    indices = np.argsort(cosine_similarities)[-k-1:-1][::-1]

    # Get client IDs of nearest neighbors
    neighbor_ids = interaction_matrix.index[indices]
    return neighbor_ids

# Recommend products for a given client based on neighbors using ALS
def recommend_products_for_client(client_id, interaction_matrix, als_model, exclude_substrings, k=5):
    neighbor_ids = get_als_nearest_neighbors(client_id, interaction_matrix, als_model, k)
    client_purchases = interaction_matrix.loc[client_id]

    recommended_products = set()

    for neighbor_id in neighbor_ids:
        neighbor_purchases = interaction_matrix.loc[neighbor_id]
        
        # Exclude products with names containing the specified substrings
        neighbor_purchases = neighbor_purchases[neighbor_purchases.index.astype(str).str.contains('|'.join(exclude_substrings)) == False]
        
        recommended_products.update(neighbor_purchases[neighbor_purchases > 0].index)

    return list(recommended_products - set(client_purchases[client_purchases > 0].index))

# Compute Recall at K for a given client
def compute_recall_at_k(client_id, recommended_products, k=15):
    client_purchases = interaction_matrix.loc[client_id]
    
    return len(set(recommended_products)) / len(client_purchases[client_purchases > 0])


# Recommend top products for all clients
def recommend_top_products_for_all_clients(interaction_matrix, als_model, exclude_substrings, k=5):
    all_clients = interaction_matrix.index
    all_recommendations = {}

    for client_id in all_clients:
        recommended_products = recommend_products_for_client(client_id, interaction_matrix, als_model, exclude_substrings, k)
        recall_at_k = compute_recall_at_k(client_id, recommended_products, k)

        # Skip clients with no recommendations
        if recommended_products:
            all_recommendations[client_id] = {
                "recommended_products": recommended_products,
                "recall_at_k": recall_at_k,
            }

    return all_recommendations

# Save recommendations to a text file
def save_recommendations_to_file(top_recommendations, output_file_path, df):
    with open(output_file_path, 'w') as output_file:
        for client_id, recommendations_info in top_recommendations.items():
            recommended_products = recommendations_info["recommended_products"]
            recall_at_k = recommendations_info["recall_at_k"]

            output_file.write(f"Top recommendations for Client {client_id}: Recall at K: ")
            
            if recall_at_k is not None:
                output_file.write(f"{recall_at_k:.4f}\n")
            else:
                output_file.write("N/A\n")

            output_file.write(f"{'Product ID':<15} {'Product Name':<30} {'Brand':<20}\n")
            output_file.write("-" * 80 + "\n")

            for product_id in recommended_products:
                product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                output_file.write(f"{product_id:<15} {product_name:<30} {brand:<20}\n")

            output_file.write("\n")


# Main routine
def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)

    df, exclude_substrings = preprocess_data(df)
    interaction_matrix = create_interaction_matrix(df)
    interaction_matrix_sparse = convert_to_sparse_matrix(interaction_matrix)

    als_model = fit_als_model(interaction_matrix_sparse)

    top_recommendations_for_all_clients = recommend_top_products_for_all_clients(interaction_matrix, als_model, exclude_substrings, k=5)

    output_file_path = "/Volumes/Untitled/cpp-recsys-clients-ALS.txt"
    save_recommendations_to_file(top_recommendations_for_all_clients, output_file_path, df)

if __name__ == "__main__":
    main()
