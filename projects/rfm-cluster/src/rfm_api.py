import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import streamlit as st

class RecSysRFM:
    def __init__(self):
        pass

    # Load data from a CSV file into a DataFrame
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Create a product-client interaction matrix from the DataFrame
    def create_interaction_matrix(self, data_frame):
        return data_frame.groupby(['Product_ID', 'Client_ID']).size().unstack(fill_value=0)

    # Instantiate and fit a Nearest Neighbors model using cosine similarity metric
    def instantiate_knn_model(self, interaction_matrix):
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model.fit(interaction_matrix.values)
        return knn_model

    # Get k nearest neighbors for a given product in the interaction matrix
    def get_nearest_neighbors(self, interaction_matrix, knn_model, product_id, k=5):
        product_index = interaction_matrix.index.get_loc(product_id)
        distances, indices = knn_model.kneighbors([interaction_matrix.iloc[product_index].values], n_neighbors=k+1)
        neighbor_ids = interaction_matrix.index[indices.flatten()[1:]]
        return neighbor_ids, distances.flatten()[1:]

    # Recommend clients for a given product based on nearest neighbors
    def recommend_clients_for_product(self, interaction_matrix, knn_model, product_id, k=5):
        neighbor_ids, distances = self.get_nearest_neighbors(interaction_matrix, knn_model, product_id, k)
        product_clients = interaction_matrix.loc[product_id]
        
        recommended_clients = []
        predicted_scores = []
        for neighbor_id, distance in zip(neighbor_ids, distances):
            neighbor_clients = interaction_matrix.loc[neighbor_id]
            recommended_clients.extend(neighbor_clients[neighbor_clients > 0].index)
            predicted_scores.append(1 - distance)  # Convert cosine distance to similarity
            
        return list(set(recommended_clients) - set(product_clients[product_clients > 0].index))[:k], predicted_scores[:k]

    # Recommend top clients for all products in the interaction matrix
    def recommend_top_clients_for_all_products(self, interaction_matrix, knn_model, k=5):
        all_products = interaction_matrix.index
        all_recommendations = {}

        for product_id in all_products:
            recommended_clients, predicted_scores = self.recommend_clients_for_product(interaction_matrix, knn_model, product_id, k)
            
            if recommended_clients:
                all_recommendations[product_id] = {"recommended_clients": recommended_clients, "predicted_scores": predicted_scores}

        return all_recommendations

    # Compute RFM (Recency, Frequency, Monetary) metrics for clients based on sales data
    def compute_rfm(self, df):
        current_date = df['Sale_Date'].max()
        rfm_data = df.groupby('Client_ID').agg({
            'Sale_Date': lambda x: (current_date - x.max()).days,
            'Product_ID': 'count',
            'Sale_Value': 'sum'
        }).reset_index()
        
        rfm_data['Recency'] = pd.qcut(rfm_data['Sale_Date'], q=10, labels=False, duplicates='drop') + 1
        rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
        rfm_data['Monetary'] = pd.qcut(rfm_data['Sale_Value'], q=10, labels=False, duplicates='drop') + 1
        
        rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']
        
        rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / \
                                            (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
        
        return rfm_data

    # Perform clustering on RFM data using KMeans algorithm
    def perform_clustering(self, rfm_data):
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm_data['Cluster'] = kmeans.fit_predict(rfm_data[['RFM_Score_Normalized']])
        return rfm_data

    # Plot clusters of clients based on RFM scores
    def plot_clusters(self, rfm_data):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {0: '#FFFFED', 1: '#FFD700', 2: '#CD7F32'}
        cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}

        for cluster, color in colors.items():
            cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
            ax.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Client_ID'], color=color, label=cluster_labels[cluster])

        ax.set_xlabel('RFM Score (Normalized)')
        ax.set_ylabel('Client ID')
        ax.set_title('Clustering of Clients')
        ax.legend()
        ax.grid(True)
        ax.set_facecolor('#21421E')
        st.pyplot(fig)

    # Print a table of RFM clustering results
    def print_results_table(self, rfm_data):
        cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
        rfm_data['RFM_Label'] = rfm_data['Cluster'].map(cluster_labels)
        centroids = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        rfm_data['Distance_to_Centroid'] = rfm_data.apply(
            lambda row: np.linalg.norm(row[['Recency', 'Frequency', 'Monetary']] - centroids.loc[row['Cluster']]), axis=1)
        
        max_distance = rfm_data['Distance_to_Centroid'].max()
        min_distance = rfm_data['Distance_to_Centroid'].min()
        rfm_data['Distance_to_Centroid_Scaled'] = ((rfm_data['Distance_to_Centroid'] - min_distance) / (max_distance - min_distance)) * 10
        rfm_data['Distance_to_Centroid_Scaled'] = rfm_data['Distance_to_Centroid_Scaled'].round().astype(int)
        
        table_data = rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Distance_to_Centroid_Scaled']]
        table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
        st.write("RFM Clustering Results:")
        st.dataframe(table_data)
        st.write("**Legend:**")
        st.write("- RFM Score: Higher values indicate better recency, frequency, and monetary metrics.")
        st.write("- RFM Label: Represents the cluster label assigned to each client (Diamond, Gold, Silver).")
        st.write("- Distance to Centroid: Indicates how close each client is to the centroid of their cluster. Smaller values imply closer proximity.")

    # Create the recommendation system web app
    def recsys_web_app(self):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            interaction_matrix = self.create_interaction_matrix(df)
            knn_model = self.instantiate_knn_model(interaction_matrix)
            num_clients_to_recommend = st.slider("Select the number of clients to recommend", 1, 10, 5)
            top_recommendations_for_all_products = self.recommend_top_clients_for_all_products(interaction_matrix, knn_model, k=num_clients_to_recommend)
            st.title("Product Recommender System")
            product_id = st.selectbox("Select a product:", list(top_recommendations_for_all_products.keys()))
            st.header(f"Top {num_clients_to_recommend} recommended clients for Product {product_id}:")
            recommended_clients = top_recommendations_for_all_products[product_id]["recommended_clients"]
            predicted_scores = top_recommendations_for_all_products[product_id]["predicted_scores"]
            for client_id, score in zip(recommended_clients, predicted_scores):
                st.write(f"Client ID: {client_id}, Predicted Score: {score:.4f}")

    # Create the RFM clustering analysis web app
    def rfm_clustering_web_app(self):
        st.title("RFM Clustering Analysis")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = self.load_data(uploaded_file)
            rfm_data = self.compute_rfm(df)
            rfm_data = self.perform_clustering(rfm_data)
            self.plot_clusters(rfm_data)
            self.print_results_table(rfm_data)

    # Main function to initialize the Streamlit web app and navigate between recommendation system and RFM clustering analysis
    def main(self):
        st.title("Unified RecSys & RFM Clustering Analysis Library")
        st.sidebar.title("Navigation")
        selected_page = st.sidebar.radio("Go to", ["Recommendation System", "RFM Clustering Analysis"])
        if selected_page == "Recommendation System":
            self.recsys_web_app()
        elif selected_page == "RFM Clustering Analysis":
            self.rfm_clustering_web_app()

if __name__ == "__main__":
    RecSysRFM().main()
