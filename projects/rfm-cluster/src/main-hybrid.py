import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
import streamlit as st

# Hide the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define functions for RFM analysis
def load_data_rfm(file_path):
    # Read the CSV file into a DataFrame using pandas
    df = pd.read_csv(file_path)
    # Convert the 'Sale_Date' column to datetime format for better handling
    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
    return df

def compute_rfm(df):
    current_date = df['Sale_Date'].max()
    rfm_data = df.groupby('Client_ID').agg({
        'Sale_Date': lambda x: (current_date - x.max()).days,  # Calculate recency
        'Product_ID': 'count',  # Count frequency of purchases
        'Sale_Value': 'sum'  # Calculate total monetary value of purchases
    }).reset_index()
    rfm_data['Recency'] = pd.qcut(rfm_data['Sale_Date'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Monetary'] = pd.qcut(rfm_data['Sale_Value'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']
    rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / \
                                       (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
    return rfm_data

def perform_clustering(rfm_data):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_data[['RFM_Score_Normalized']])
    return rfm_data

def plot_clusters(rfm_data):
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

def print_results_table(rfm_data):
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

# Define functions for the recommender system
def load_data_recommender(file_object):
    try:
        # Attempt to read the CSV file into a DataFrame using pandas
        df = pd.read_csv(file_object)
        excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
        excluded_brands = ['DEERE']
        df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
        df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]
        return df
    except pd.errors.EmptyDataError:
        # Handle the case where the file is empty or contains no data
        st.error("The uploaded file is empty or contains no data.")
        return None
    except Exception as e:
        # Handle any other exceptions that might occur during file reading
        st.error(f"An error occurred while reading the file: {e}")
        return None

def create_interaction_matrix(data_frame):   
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size()
    interaction_matrix = interaction_matrix.unstack(fill_value=0)
    return interaction_matrix

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
    autoencoder_model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test)
    )

def compute_similarity_matrix(interaction_matrix, k=5):
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    knn.fit(interaction_matrix.values)
    similarity_matrix = 1 - knn.kneighbors_graph(mode='distance').toarray()
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix

def get_embeddings(encoder_model, X):
    embeddings = encoder_model.predict(X)
    return embeddings

def recommend_products_hybrid(embeddings, user_similarity_matrix, client_id, interaction_matrix, k=5):
    client_index = interaction_matrix.index.get_loc(client_id)
    client_embedding = embeddings[client_index]
    weighted_embeddings = np.dot(user_similarity_matrix[client_index], embeddings)
    combined_embeddings = 0.5 * client_embedding + 0.5 * weighted_embeddings
    similarities = np.dot(embeddings, combined_embeddings) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(combined_embeddings))
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]
    predicted_scores = similarities[product_indices]
    return recommended_products, predicted_scores

def main():
    st.title("Unified Web App")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        df_rfm = load_data_rfm(uploaded_file)
        df_recommender = load_data_recommender(uploaded_file)

        if df_recommender is None:
            return  # Stop further processing if the recommender DataFrame is empty

        # Compute RFM
        rfm_data = compute_rfm(df_rfm)

        # Perform clustering
        rfm_data = perform_clustering(rfm_data)

        # Plot clusters
        plot_clusters(rfm_data)

        # Print results table
        print_results_table(rfm_data)

        # Create interaction matrix for recommender system
        interaction_matrix = create_interaction_matrix(df_recommender)

        # Split interaction matrix
        train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)

        # Build and train autoencoder model
        autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
        train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)

        # Extract embeddings using encoder model
        encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
        embeddings = get_embeddings(encoder_model, interaction_matrix.values)

        # Compute user similarity matrix
        user_similarity_matrix = compute_similarity_matrix(interaction_matrix, k=5)

        # Obtain recommendations for all clients
        top_recommendations_for_all_clients = {}
        for client_id in interaction_matrix.index:
            recommended_products, predicted_scores = recommend_products_hybrid(
                embeddings, user_similarity_matrix, client_id, interaction_matrix, k=5
            )
            top_recommendations_for_all_clients[int(client_id)] = {
                "recommended_products": recommended_products,
                "predicted_scores": predicted_scores
            }

        # Display recommendations for each client
        for client_id, recommendations_info in top_recommendations_for_all_clients.items():
            st.subheader(f"Top recommendations for Client {client_id}:")

            product_ids = recommendations_info['recommended_products']
            product_names = [df_recommender.loc[df_recommender['Product_ID'] == product_id, 'Product_Name'].iloc[0] for product_id in product_ids]
            brands = [df_recommender.loc[df_recommender['Product_ID'] == product_id, 'Brand_Name'].iloc[0] for product_id in product_ids]
            predicted_scores = recommendations_info['predicted_scores']

            min_length = min(len(product_ids), len(product_names), len(brands), len(predicted_scores))
            product_ids = product_ids[:min_length]
            product_names = product_names[:min_length]
            brands = brands[:min_length]
            predicted_scores = predicted_scores[:min_length]

            recommendations_df = pd.DataFrame({
                'Product ID': product_ids,
                'Product Name': product_names,
                'Brand': brands,
                'Predicted Score': predicted_scores
            })

            st.table(recommendations_df)

if __name__ == "__main__":
    main()
