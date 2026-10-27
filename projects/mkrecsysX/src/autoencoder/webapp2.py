import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tabulate import tabulate

# Functions for the autoencoder-based recommendation system
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]
    return df

def create_interaction_matrix(data_frame):
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def build_autoencoder_model(input_size):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_size, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

def get_embeddings(encoder_model, X):
    return encoder_model.predict(X)

def recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    client_index = interaction_matrix.index.get_loc(client_id)
    client_embedding = embeddings[client_index]
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]
    return recommended_products

def recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5):
    product_index = interaction_matrix.columns.get_loc(product_id)
    product_embedding = embeddings[:, product_index]  # Extracting the product embedding
    transposed_embeddings = embeddings.T
    recommendations = []
    for client_embedding in transposed_embeddings:
        similarity = np.dot(client_embedding, product_embedding)
        recommendations.append(similarity)
    recommendations = np.array(recommendations)
    valid_indices = np.where(recommendations.argsort()[::-1] < len(interaction_matrix.columns))[0]
    client_indices = recommendations.argsort()[::-1][valid_indices][:k]
    return interaction_matrix.index[client_indices]

# RFM analysis functions
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])  # Convert Sales_Date to datetime
    return df

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
    rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
    return rfm_data

def perform_clustering(rfm_data):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_data[['RFM_Score_Normalized']])
    silhouette_avg = silhouette_score(rfm_data[['RFM_Score_Normalized']], rfm_data['Cluster'])
    cluster_silhouette_scores = silhouette_samples(rfm_data[['RFM_Score_Normalized']], rfm_data['Cluster'])
    rfm_data['Silhouette'] = cluster_silhouette_scores
    return rfm_data, silhouette_avg

def plot_rfm_clusters(rfm_data):
    plt.figure(figsize=(10, 6))
    colors = {0: '#4C72B0', 1: '#55A868', 2: '#C44E52'}
    for cluster, color in colors.items():
        cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
        plt.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Client_ID'], color=color, label=f'Cluster {cluster}')
        cluster_mean_silhouette = cluster_data['Silhouette'].mean()
        plt.errorbar(cluster_mean_silhouette, max(cluster_data['Client_ID']), xerr=cluster_data['Silhouette'].std(), fmt='o', color='black')
    plt.xlabel('RFM Score (Normalized)')
    plt.ylabel('Client ID')
    plt.title('Clustering of Clients based on RFM Scores')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def print_rfm_results_table(rfm_data, silhouette_avg):
    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
    rfm_data['RFM_Label'] = rfm_data['Cluster'].map(cluster_labels)
    table_data = rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Silhouette']]
    table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
    st.write("RFM Clustering Results:")
    st.dataframe(table_data)
    st.write(f"Average Silhouette Score: {silhouette_avg}")

# Functions for frequently bought items recommender system (new)
def load_data_new(file_path):
    return pd.read_csv(file_path)

def create_interaction_matrix_new(data_frame):
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def instantiate_similarity_matrix(interaction_matrix):
    similarity_matrix = cosine_similarity(interaction_matrix.T)
    return similarity_matrix

def get_similar_products(similarity_matrix, interaction_matrix, product_id, k=5):
    product_index = interaction_matrix.columns.get_loc(product_id)
    similar_indices = similarity_matrix[product_index].argsort()[:-k-1:-1]
    similar_products = interaction_matrix.columns[similar_indices].tolist()[1:]
    return similar_products

def recommend_products_for_product(similarity_matrix, interaction_matrix, product_id, k=5):
    similar_products = get_similar_products(similarity_matrix, interaction_matrix, product_id, k)
    relevant_items = interaction_matrix[interaction_matrix[product_id] > 0].index.tolist()
    recommended_relevant = sum(1 for item in similar_products if item in relevant_items)
    recall_at_k = len(relevant_items)
    return similar_products, recall_at_k

def main():
    st.title("Makes RecSys X Advanced")

    # Home page description
    st.write("""
    Welcome to Makes RecSys X Advanced!
    
    This sophisticated web application offers advanced recommendation systems and RFM analysis tailored for our esteemed clientele. 
    Upload your CSV file to embark on a journey of exploring cutting-edge recommendation techniques and gaining insightful business intelligence.
    """)

    # Generate unique keys for file uploaders
    upload_key = "Upload CSV File"

    # Upload CSV file
    file = st.sidebar.file_uploader(upload_key, key="file_uploader", type=["csv"])
    if file is not None:
        df = load_data(file)
        interaction_matrix = create_interaction_matrix(df)

        # Train autoencoder
        train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)
        autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
        train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)
        encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
        embeddings = get_embeddings(encoder_model, interaction_matrix.values)

        # Create tabs
        tabs = ["Recommendations: Clients to Products", 
                "Recommendations: Products to Clients", 
                "RFM Analysis",
                "Frequently Bought Items"]
        selected_tab = st.sidebar.radio("Navigation", tabs)

        # Recommendations: Clients to Products
        if selected_tab == "Recommendations: Clients to Products":
            st.subheader("Client-Centric Product Recommendations")
            st.write("""
            This section provides discerning clients with personalized recommendations, meticulously crafted based on their historical product interactions. 
            Leveraging sophisticated autoencoder techniques, our system discerns latent features to furnish tailored product suggestions of utmost relevance.
            """)
            for client_id in interaction_matrix.index:
                with st.expander(f"Client {client_id} Recommendations"):
                    recommended_products = recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=15)
                    actual_products = interaction_matrix.loc[client_id]
                    recall_at_k = len(set(recommended_products).intersection(actual_products[actual_products > 0].index)) / len(actual_products[actual_products > 0])
                    table_data = []
                    for product_id in recommended_products:
                        product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                        brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                        table_data.append([product_id, product_name, brand])
                    st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                    st.dataframe(pd.DataFrame(table_data, columns=["Product ID", "Product Name", "Brand"]))

        # Recommendations: Products to Clients
        elif selected_tab == "Recommendations: Products to Clients":
            st.subheader("Product-Focused Client Recommendations")
            st.write("""
            This segment offers refined recommendations tailored for each product, pinpointing prospective clients likely to exhibit interest. 
            By identifying target clientele, it empowers businesses with precise targeting strategies, optimizing marketing endeavors.
            """)
            for product_id in interaction_matrix.columns:
                with st.expander(f"Product {product_id} Recommendations"):
                    recommended_clients = recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)
                    if len(recommended_clients) > 0:
                        recall_at_k = len(set(recommended_clients).intersection(interaction_matrix.index)) / len(interaction_matrix.index)
                        table_data = []
                        for client_id in recommended_clients:
                            table_data.append([client_id])
                        st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                        st.dataframe(pd.DataFrame(table_data, columns=["Client ID"]))
                    else:
                        st.write("No recommendations for this product.")

        # RFM Analysis
        elif selected_tab == "RFM Analysis":
            st.subheader("RFM Analysis: Unveiling Client Segmentation")
            st.write("""
            This section conducts meticulous RFM (Recency, Frequency, Monetary) analysis to unveil nuanced client segments. 
            By clustering clients based on their behavior, it provides actionable insights, enabling tailored marketing strategies for diverse clientele.
            """)
            rfm_data = compute_rfm(df)
            rfm_data, silhouette_avg = perform_clustering(rfm_data)
            plot_rfm_clusters(rfm_data)
            print_rfm_results_table(rfm_data, silhouette_avg)

        # Frequently Bought Items
        elif selected_tab == "Frequently Bought Items":
            st.subheader("Frequently Bought Items Recommender System")
            st.write("""
            This cutting-edge recommender system suggests frequently bought items analogous to each product, amplifying cross-selling and upselling initiatives. 
            By identifying correlated product pairs, it augments overall sales performance, fostering sustainable business growth.
            """)
            similarity_matrix = instantiate_similarity_matrix(interaction_matrix)
                
            # Display recommendations for each product
            for product_id in interaction_matrix.columns:
                similar_products, recall_at_k = recommend_products_for_product(similarity_matrix, interaction_matrix, product_id, k=5)
                st.subheader(f"Recommendations for Product ID: {product_id} (Recall at K: {recall_at_k:.2f})")
                recommended_product_info = df[df['Product_ID'].isin(similar_products)][['Product_ID', 'Product_Name', 'Brand_Name']].drop_duplicates()
                st.dataframe(recommended_product_info)

if __name__ == "__main__":
    main()
