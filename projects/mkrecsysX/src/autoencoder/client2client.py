import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

def load_data(file_path):
    return pd.read_csv(file_path)

def create_interaction_matrix(data_frame):
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def instantiate_similarity_matrix(interaction_matrix):
    similarity_matrix = cosine_similarity(interaction_matrix)
    return similarity_matrix

def get_similar_clients(similarity_matrix, interaction_matrix, client_id, k=5):
    client_index = interaction_matrix.index.get_loc(client_id)
    similar_indices = similarity_matrix[client_index].argsort()[:-k-1:-1]
    similar_clients = interaction_matrix.iloc[similar_indices].index.tolist()[1:]
    return similar_clients

def recommend_clients_for_client(similarity_matrix, interaction_matrix, client_id, k=5):
    similar_clients = get_similar_clients(similarity_matrix, interaction_matrix, client_id, k)
    
    # Calculate recall at k
    relevant_items = interaction_matrix.loc[client_id][interaction_matrix.loc[client_id] > 0].index.tolist()
    recommended_relevant = sum(1 for item in similar_clients if item in relevant_items)
    recall_at_k = len(relevant_items)
    
    return similar_clients, recall_at_k

def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)
    interaction_matrix = create_interaction_matrix(df)
    similarity_matrix = instantiate_similarity_matrix(interaction_matrix)
    
    all_recommendations = {}
    all_recall_at_k = {}
    
    for client_id in interaction_matrix.index:
        recommended_clients, recall_at_k = recommend_clients_for_client(similarity_matrix, interaction_matrix, client_id, k=5)
        all_recommendations[client_id] = recommended_clients
        all_recall_at_k[client_id] = recall_at_k
    
    # Display recommendations for all clients along with recall at k
    result_table = []
    for client_id, recommended_clients in all_recommendations.items():
        recall = all_recall_at_k[client_id]
        result_table.append([client_id, recommended_clients, recall])
    
    headers = ["Client ID", "Recommended Clients", "Recall at K"]
    print(tabulate(result_table, headers=headers, tablefmt='grid'))

if __name__ == "__main__":
    main()
