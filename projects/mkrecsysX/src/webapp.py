import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples

def load_data(file):
    df = pd.read_csv(file)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {0: '#4C72B0',  # Diamond - blue
              1: '#55A868',  # Gold - green
              2: '#C44E52'}  # Silver - red

    for cluster, color in colors.items():
        cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
        ax.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Client_ID'], color=color, label=f'Cluster {cluster}')
        
        # Compute the mean silhouette score for the cluster
        cluster_mean_silhouette = cluster_data['Silhouette'].mean()
        # Plot the silhouette score as an error bar
        ax.errorbar(cluster_mean_silhouette, max(cluster_data['Client_ID']), xerr=cluster_data['Silhouette'].std(), fmt='o', color='black')

    ax.set_xlabel('RFM Score (Normalized)')
    ax.set_ylabel('Client ID')
    ax.set_title('Clustering of Clients')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def recommend_clients_for_product(interaction_matrix, product_id, knn_model, k=5):
    product_index = interaction_matrix.columns.get_loc(product_id)
    distances, indices = knn_model.kneighbors(interaction_matrix.iloc[:, product_index].values.reshape(1, -1), n_neighbors=k+1)
    client_ids = interaction_matrix.index[indices.flatten()[1:]]
    return client_ids

def recommend_products_for_all_clients(interaction_matrix, knn_model, k=5):
    all_clients = interaction_matrix.index
    all_recommendations = {}

    for client_id in all_clients:
        recommended_products = recommend_products_for_client(interaction_matrix, client_id, knn_model, k)
        recall_at_k = compute_recall_at_k(interaction_matrix, client_id, recommended_products, k)
        
        if recommended_products:
            all_recommendations[client_id] = {"recommended_products": recommended_products, "recall_at_k": recall_at_k}

    return all_recommendations

def main():
    st.title("Makes RecSys X")

    # Generate unique keys for file uploaders
    upload_key = "Upload CSV File"

    # Upload CSV file
    file = st.sidebar.file_uploader(upload_key, type=["csv"])
    if file is not None:
        df = load_data(file)
        
        # Create interaction matrix
        interaction_matrix = create_interaction_matrix(df)
        
        # Instantiate KNN model
        knn_model = instantiate_knn_model(interaction_matrix)
        
        # Recommend top products for all clients
        top_recommendations_for_all_clients = recommend_top_products_for_all_clients(interaction_matrix, knn_model, k=5)
        
        # Recommend top clients for all products
        top_recommendations_for_all_products = recommend_products_for_all_clients(interaction_matrix, knn_model, k=5)  # Corrected function name
        
        # Create tabs for recommendations and RFM analysis
        tabs = st.sidebar.radio("Tabs", ["Product Recommendations", "Client Recommendations", "RFM Analysis"])
        
        if tabs == "Product Recommendations":
            st.subheader("Product Recommendations")
            for client_id, recommendations_info in top_recommendations_for_all_clients.items():
                recommended_products = recommendations_info["recommended_products"]
                recall_at_k = recommendations_info["recall_at_k"]
                with st.expander(f"Client ID: {client_id}"):
                    st.write(f"Recall at K: {recall_at_k}")
                    for product_id in recommended_products:
                        product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                        brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                        st.write(f"- Product ID: {product_id}, Product Name: {product_name}, Brand: {brand}")
        
        elif tabs == "Client Recommendations":
            st.subheader("Client Recommendations")
            for product_id, recommendations_info in top_recommendations_for_all_products.items():
                recommended_clients = recommendations_info["recommended_products"]  # Fix the key here
                recall_at_k = recommendations_info["recall_at_k"]
                with st.expander(f"Product ID: {product_id}"):
                    st.write(f"Recall at K: {recall_at_k}")
                    for client_id in recommended_clients:
                        st.write(f"- Client ID: {client_id}")

        elif tabs == "RFM Analysis":
            st.subheader("RFM Clustering")
            rfm_data = compute_rfm(df)
            rfm_data, silhouette_avg = perform_clustering(rfm_data)
            
            # Plot clustering of clients
            plot_clusters(rfm_data)
            
            # Display RFM analysis table with normalized RFM score and silhouette score
            st.write("Average Silhouette Score:", silhouette_avg)
            st.table(rfm_data[['Client_ID', 'RFM_Score_Normalized', 'Silhouette']])

if __name__ == "__main__":
    main()
