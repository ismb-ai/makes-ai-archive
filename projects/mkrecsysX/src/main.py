import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

def create_interaction_matrix(data_frame):
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def instantiate_knn_model(interaction_matrix):
    return NearestNeighbors(metric='cosine', algorithm='brute').fit(interaction_matrix.values)

def get_nearest_neighbors(interaction_matrix, knn_model, client_id, k=5):
    client_index = interaction_matrix.index.get_loc(client_id)
    distances, indices = knn_model.kneighbors([interaction_matrix.iloc[client_index].values], n_neighbors=k+1)
    return interaction_matrix.index[indices.flatten()[1:]]

def recommend_products_for_client(interaction_matrix, client_id, knn_model, k=5):
    neighbor_ids = get_nearest_neighbors(interaction_matrix, knn_model, client_id, k)
    client_purchases = interaction_matrix.loc[client_id]
    
    recommended_products = []
    for neighbor_id in neighbor_ids:
        neighbor_purchases = interaction_matrix.loc[neighbor_id]
        recommended_products.extend(neighbor_purchases[neighbor_purchases > 0].index)
    
    return list(set(recommended_products) - set(client_purchases[client_purchases > 0].index))

def compute_recall_at_k(interaction_matrix, client_id, recommended_products, k=15):
    client_purchases = interaction_matrix.loc[client_id]
    return len(set(recommended_products)) / len(client_purchases[client_purchases > 0])

def recommend_top_products_for_all_clients(interaction_matrix, knn_model, k=5):
    all_clients = interaction_matrix.index
    all_recommendations = {}

    for client_id in all_clients:
        recommended_products = recommend_products_for_client(interaction_matrix, client_id, knn_model, k)
        recall_at_k = compute_recall_at_k(interaction_matrix, client_id, recommended_products, k)
        
        if recommended_products:
            all_recommendations[client_id] = {"recommended_products": recommended_products, "recall_at_k": recall_at_k}

    return all_recommendations

def compute_rfm(df):
    current_date = df['Sales_Date'].max()
    rfm_data = df.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Recency
        'Product_ID': 'count',  # Frequency
        'Monetary_Value': 'sum'  # Monetary
    }).reset_index()
    
    rfm_data['Recency'] = pd.qcut(rfm_data['Sales_Date'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Monetary'] = pd.qcut(rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1
    
    rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']
    rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / \
                                       (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
    return rfm_data

def perform_clustering(rfm_data):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_data[['RFM_Score_Normalized']])
    silhouette_avg = silhouette_score(rfm_data[['RFM_Score_Normalized']], rfm_data['Cluster'])
    cluster_silhouette_scores = silhouette_samples(rfm_data[['RFM_Score_Normalized']], rfm_data['Cluster'])
    rfm_data['Silhouette'] = cluster_silhouette_scores
    return rfm_data, silhouette_avg

def plot_clusters(rfm_data):
    plt.figure(figsize=(10, 6))
    
    colors = {0: '#4C72B0',  # Diamond - blue
              1: '#55A868',  # Gold - green
              2: '#C44E52'}  # Silver - red

    for cluster, color in colors.items():
        cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
        plt.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Client_ID'], color=color, label=f'Cluster {cluster}')
        
        # Compute the mean silhouette score for the cluster
        cluster_mean_silhouette = cluster_data['Silhouette'].mean()
        # Plot the silhouette score as an error bar
        plt.errorbar(cluster_mean_silhouette, max(cluster_data['Client_ID']), xerr=cluster_data['Silhouette'].std(), fmt='o', color='black')

    plt.xlabel('RFM Score (Normalized)')
    plt.ylabel('Client ID')
    plt.title('Clustering of Clients')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)
    interaction_matrix = create_interaction_matrix(df)
    knn_model = instantiate_knn_model(interaction_matrix)
    top_recommendations_for_all_clients = recommend_top_products_for_all_clients(interaction_matrix, knn_model, k=5)

    print("Product Recommendations:")
    headers = ["Client ID", "Recall at K", "Product ID", "Product Name", "Brand"]
    table_data = []

    for client_id, recommendations_info in top_recommendations_for_all_clients.items():
        recommended_products = recommendations_info["recommended_products"]
        recall_at_k = recommendations_info["recall_at_k"]
        
        for product_id in recommended_products:
            product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
            brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
            table_data.append([client_id, recall_at_k, product_id, product_name, brand])

    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("\n")

    # RFM Clustering
    rfm_data = compute_rfm(df)
    rfm_data, silhouette_avg = perform_clustering(rfm_data)
    plot_clusters(rfm_data)
    print_results_table(rfm_data, silhouette_avg)

if __name__ == "__main__":
    main()
