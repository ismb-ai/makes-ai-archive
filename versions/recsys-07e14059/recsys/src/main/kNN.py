# k-NN Recommender System (Producs to Clients)

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_data(file_path):
    # Load the CSV file into a DataFrame.
    return pd.read_csv(file_path)

def create_interaction_matrix(data_frame):
    # Create a user-item interaction matrix.
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def instantiate_knn_model(interaction_matrix):
    # Instantiate the NearestNeighbors model and fit it to the interaction matrix.
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(interaction_matrix.values)
    return knn_model

def get_nearest_neighbors(interaction_matrix, knn_model, client_id, k=5):
    # Get k-nearest neighbors for a given client.
    client_index = interaction_matrix.index.get_loc(client_id)
    distances, indices = knn_model.kneighbors([interaction_matrix.iloc[client_index].values], n_neighbors=k+1)
    neighbor_ids = interaction_matrix.index[indices.flatten()[1:]]  # Exclude the client itself
    return neighbor_ids

def recommend_products_for_client(interaction_matrix, client_id, knn_model, k=5):
    # Recommend products for a given client based on neighbors.
    neighbor_ids = get_nearest_neighbors(interaction_matrix, knn_model, client_id, k)
    client_purchases = interaction_matrix.loc[client_id]
    
    recommended_products = []
    for neighbor_id in neighbor_ids:
        neighbor_purchases = interaction_matrix.loc[neighbor_id]
        recommended_products.extend(neighbor_purchases[neighbor_purchases > 0].index)
    
    return list(set(recommended_products) - set(client_purchases[client_purchases > 0].index))

def compute_recall_at_k(interaction_matrix, client_id, recommended_products, k=15):
    # Compute Recall at K for a given client.
    client_purchases = interaction_matrix.loc[client_id]
    return len(set(recommended_products)) / len(client_purchases[client_purchases > 0])

def recommend_top_products_for_all_clients(interaction_matrix, knn_model, k=5):
    # Recommend top products for all clients.
    all_clients = interaction_matrix.index
    all_recommendations = {}

    for client_id in all_clients:
        recommended_products = recommend_products_for_client(interaction_matrix, client_id, knn_model, k)
        recall_at_k = compute_recall_at_k(interaction_matrix, client_id, recommended_products, k)
        
        # Skip clients with no recommendations
        if recommended_products:
            all_recommendations[client_id] = {"recommended_products": recommended_products, "recall_at_k": recall_at_k}

    return all_recommendations

def main():
    # Load the CSV file into a DataFrame
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)

    # Create a user-item interaction matrix
    interaction_matrix = create_interaction_matrix(df)

    # Instantiate the NearestNeighbors model and fit it to the interaction matrix
    knn_model = instantiate_knn_model(interaction_matrix)

    # Recommend top 5 products for all clients
    top_recommendations_for_all_clients = recommend_top_products_for_all_clients(interaction_matrix, knn_model, k=5)

    # Save the results to a text file
    output_file_path = "/Volumes/Untitled/cpp-recsys-clients-kNN.txt"
    with open(output_file_path, 'w') as output_file:
        for client_id, recommendations_info in top_recommendations_for_all_clients.items():
            recommended_products = recommendations_info["recommended_products"]
            recall_at_k = recommendations_info["recall_at_k"]
            
            output_file.write(f"Top recommendations for Client {client_id}: Recall at K: {recall_at_k:.4f}\n")
            output_file.write(f"{'Product ID':<15} {'Product Name':<30} {'Brand':<20}\n")
            output_file.write("-" * 80 + "\n")
            
            for product_id in recommended_products:
                product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                output_file.write(f"{product_id:<15} {product_name:<30} {brand:<20}\n")
            
            output_file.write("\n")

if __name__ == "__main__":
    main()
