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
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Functions for autoencoder-based recommendation system
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])  # Convert Sales_Date to datetime
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
    
    # Transpose the embeddings matrix so that each row represents a client embedding
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
def compute_rfm(df):
    current_date = pd.to_datetime(df['Sales_Date']).max()  # Convert current_date to Timestamp
    rfm_data = df.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - pd.to_datetime(x).max()).days,  # Recency
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

def plot_rfm_clusters(rfm_data):
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
    st.dataframe(table_data)  # Display RFM clustering results as a table
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
    
    # Calculate recall at k
    relevant_items = interaction_matrix[interaction_matrix[product_id] > 0].index.tolist()
    recommended_relevant = sum(1 for item in similar_products if item in relevant_items)
    recall_at_k = len(relevant_items)
    
    return similar_products, recall_at_k
    
# Functions for client-to-client recommender system
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

def client_to_client_recommender(interaction_matrix):
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
    
    return result_table
    
# RFM analysis
def preprocess_rfm_data(df):
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    current_date = df['Sales_Date'].max()
    rfm = df.groupby(['Client_ID', pd.Grouper(key='Sales_Date', freq='M')]).agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,
        'Sales_Date': 'count',
        'Monetary_Value': 'sum'
    }).rename(columns={'Sales_Date': 'Recency', 'Monetary_Value': 'Monetary'}).reset_index()
    return rfm

def cluster_clients(rfm):
    angular_coefficients = []
    for client_id, group in rfm.groupby('Client_ID'):
        if len(group) < 2:  # Skip clients with insufficient data
            continue
        
        X = group.index.values.reshape(-1, 1)
        y = group['Recency'].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        angular_coefficient = lr.coef_[0]
        angular_coefficients.append([client_id, angular_coefficient])
    
    angular_coefficients_df = pd.DataFrame(angular_coefficients, columns=['Client_ID', 'Angular_Coefficient'])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(angular_coefficients_df[['Angular_Coefficient']])
    angular_coefficients_df['Cluster'] = kmeans.labels_
    
    cluster_labels = {
        0: "Loyal customer",
        1: "Stable customer",
        2: "Customer at risk"
    }
    angular_coefficients_df['Cluster_Label'] = angular_coefficients_df['Cluster'].map(cluster_labels)
    
    return angular_coefficients_df

def plot_cluster_results(angular_coefficients_df):
    plt.figure(figsize=(10, 6))
    for label, color in zip(angular_coefficients_df['Cluster_Label'].unique(), ['blue', 'green', 'red']):
        subset = angular_coefficients_df[angular_coefficients_df['Cluster_Label'] == label]
        plt.scatter(subset['Client_ID'], subset['Angular_Coefficient'], label=label, color=color)

    plt.title('Clustering of Clients based on Angular Coefficients')
    plt.xlabel('Client ID')
    plt.ylabel('Angular Coefficient')
    plt.legend()
    st.pyplot(plt)

def forecast_rfm_trends(rfm, angular_coefficients_df):
    for client_id, group in rfm.groupby('Client_ID'):
        if len(group) < 2:  # Skip clients with insufficient data
            continue
        
        X = group.index.values.reshape(-1, 1)
        y = group['Recency'].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        y_pred = lr.predict(X)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(group.index, group['Recency'], label='Historical Data')
        ax.plot(group.index, y_pred, label='Forecast', linestyle='--')
        
        ax.set_title(f'RFM Score Forecast for Client ID: {client_id}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Recency')
        ax.legend()
        ax.annotate(f'Angular Coefficient: {angular_coefficients_df[angular_coefficients_df["Client_ID"] == client_id]["Angular_Coefficient"].values[0]}', 
                    xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        st.pyplot(fig)
        
def print_clustering_results_table(angular_coefficients_df):
    st.subheader("Clustering Results")
    st.markdown("Table of clustering results:")
    st.table(angular_coefficients_df[['Client_ID', 'Cluster', 'Cluster_Label']])
    
# Function to load product data from CSV
def load_product_data(file_path):
    df = pd.read_csv(file_path)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

# Function to compute RFM metrics for products
def compute_product_rfm(df):
    product_rfm_data = df.groupby('Product_ID').agg({
        'Sales_Date': lambda x: len(x.unique()),  # Recency (count of unique sales dates)
        'Client_ID': 'count',  # Frequency (count of clients)
        'Monetary_Value': 'sum'  # Monetary (sum of monetary value)
    }).reset_index()
    
    product_rfm_data.rename(columns={'Sales_Date': 'Recency', 'Client_ID': 'Frequency'}, inplace=True)
    
    product_rfm_data['Recency'] = pd.qcut(product_rfm_data['Recency'], q=10, labels=False, duplicates='drop') + 1
    product_rfm_data['Frequency'] = pd.qcut(product_rfm_data['Frequency'], q=10, labels=False, duplicates='drop') + 1
    product_rfm_data['Monetary'] = pd.qcut(product_rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1
    
    product_rfm_data['RFM_Score'] = product_rfm_data['Recency'] + product_rfm_data['Frequency'] + product_rfm_data['Monetary']
    product_rfm_data['RFM_Score_Normalized'] = (product_rfm_data['RFM_Score'] - product_rfm_data['RFM_Score'].min()) / \
                                               (product_rfm_data['RFM_Score'].max() - product_rfm_data['RFM_Score'].min()) * 100
    return product_rfm_data

# Function to perform KMeans clustering on product RFM data
def perform_product_clustering(product_rfm_data):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    product_rfm_data['Cluster'] = kmeans.fit_predict(product_rfm_data[['RFM_Score_Normalized']])
    silhouette_avg = silhouette_score(product_rfm_data[['RFM_Score_Normalized']], product_rfm_data['Cluster'])
    cluster_silhouette_scores = silhouette_samples(product_rfm_data[['RFM_Score_Normalized']], product_rfm_data['Cluster'])
    product_rfm_data['Silhouette'] = cluster_silhouette_scores
    return product_rfm_data, silhouette_avg

# Function to plot product clusters
def plot_product_clusters(product_rfm_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {0: '#4C72B0',  # Diamond - blue
              1: '#55A868',  # Gold - green
              2: '#C44E52'}  # Silver - red

    for cluster, color in colors.items():
        cluster_data = product_rfm_data[product_rfm_data['Cluster'] == cluster]
        ax.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Product_ID'], color=color, label=f'Cluster {cluster}')
        
        # Compute the mean silhouette score for the cluster
        cluster_mean_silhouette = cluster_data['Silhouette'].mean()
        # Plot the silhouette score as an error bar
        ax.errorbar(cluster_mean_silhouette, max(cluster_data['Product_ID']), xerr=cluster_data['Silhouette'].std(), fmt='o', color='black')

    ax.set_xlabel('RFM Score (Normalized)')
    ax.set_ylabel('Product ID')
    ax.set_title('Clustering of Products')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Function to print product clustering results table
def print_product_results_table(product_rfm_data, silhouette_avg):
    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
    product_rfm_data['RFM_Label'] = product_rfm_data['Cluster'].map(cluster_labels)
    
    table_data = product_rfm_data[['Product_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Silhouette']]
    table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
    st.subheader("Product RFM Clustering Results")
    st.markdown("Table of clustering results:")
    st.table(table_data)
    st.write(f"Average Silhouette Score: {silhouette_avg}")
    
# Function to load data from CSV
def load_data(file_path):
    """Load data from CSV."""
    return pd.read_csv(file_path)

# Function to calculate product sales and classify products into ABC classes
def calculate_product_sales(data):
    """Calculate product sales."""
    product_sales = data.groupby('Product_ID')['Monetary_Value'].sum().reset_index()
    product_sales = product_sales.sort_values(by='Monetary_Value', ascending=False)
    product_sales['Cumulative_Percentage'] = product_sales['Monetary_Value'].cumsum() / product_sales['Monetary_Value'].sum() * 100
    return product_sales

def classify_product(row):
    """Classify products into A, B, and C categories based on cumulative percentage."""
    if row['Cumulative_Percentage'] <= 70:
        return 'A'
    elif row['Cumulative_Percentage'] <= 90:
        return 'B'
    else:
        return 'C'

# Function to plot the clustering of products according to their ABC class
def plot_product_clustering(product_sales):
    """Plot the clustering of products according to their ABC class."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Product_ID', y='Cumulative_Percentage', hue='ABC_Class', data=product_sales)
    plt.title('Clustering of Products by ABC Class')
    plt.xlabel('Product ID')
    plt.ylabel('Cumulative Percentage of Sales')
    plt.legend(title='ABC Class')
    st.pyplot(plt)

# Function to add reorder point and delivery frequency to the product sales dataframe
def add_reorder_and_frequency(product_sales):
    """Add re-order point and delivery frequency to the dataframe."""
    reorder_points = {'A': 5000, 'B': 2000, 'C': 1000}
    delivery_frequencies = {'A': 'Weekly', 'B': 'Bi-Weekly', 'C': 'Monthly'}
    product_sales['Reorder_Point'] = product_sales['ABC_Class'].map(reorder_points)
    product_sales['Delivery_Frequency'] = product_sales['ABC_Class'].map(delivery_frequencies)
    return product_sales

# Function to print the ABC classification results in a formatted table
def print_results(product_sales):
    """Print the results in a formatted table."""
    st.write("ABC Classification of Products with Suggested Re-order Point and Delivery Frequency:")
    st.dataframe(product_sales)
    
# Function to read CSV file into a DataFrame
def sales_read_data(file_path):
    return pd.read_csv(file_path)

# Function to convert 'Sales_Date' to timestamps
def sales_convert_to_timestamps(df):
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

# Function to calculate RFM scores
def sales_calculate_rfm_scores(df):
    today = pd.to_datetime('today')
    rfm = df.groupby('Product_ID').agg({
        'Sales_Date': lambda x: (today - x.max()).days,  # Recency
        'Client_ID': 'nunique',                         # Frequency
        'Monetary_Value': 'sum'                         # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

# Function to compute the geometric mean of RFM scores
def sales_calculate_geometric_mean(rfm_scores):
    rfm_scores['Geometric_Mean'] = np.power(rfm_scores.prod(axis=1), 1/3)
    return rfm_scores['Geometric_Mean']

# Function to perform clustering
def sales_perform_clustering(product_features):
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(product_features[['Angular_Coefficient', 'Mean_RFM_Score']])
    product_features['Cluster'] = clusters
    return product_features

# Function to plot clustering results
def sales_plot_clusters(product_features):
    cluster_labels = {
        0: "High performance",
        1: "Stable performance",
        2: "Low performance",
        3: "Loss"
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id, label in cluster_labels.items():
        cluster_data = product_features[product_features['Cluster'] == cluster_id]
        ax.scatter(cluster_data.index, cluster_data['Mean_RFM_Score'], label=label)
    ax.set_xlabel('Product ID')
    ax.set_ylabel('Mean RFM Score')
    ax.set_title('K-means Clustering of Product Sales')
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    st.pyplot(fig)


# Function to print the table of each product with its features
def sales_print_product_features(product_features):
    table = product_features.reset_index().rename(columns={'index': 'Product ID'})
    st.write(tabulate(table, headers='keys', tablefmt='pipe'))

# Function to remove outliers using Bollinger bands
def sales_remove_outliers(group):
    std_dev = group['Monetary_Value'].std()
    mean = group['Monetary_Value'].mean()
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    return group[(group['Monetary_Value'] >= lower_bound) & (group['Monetary_Value'] <= upper_bound)]
    
# Plot sales forecast
def sales_plot_forecast(grouped_data, rfm_scores, num_products=10):
    product_ids = list(grouped_data.groups.keys())
    sampled_ids = np.random.choice(product_ids, size=min(num_products, len(product_ids)), replace=False)
    for product_id in sampled_ids:
        fig, ax = plt.subplots(figsize=(8, 5))
        group = grouped_data.get_group(product_id)
        
        # Remove outliers using Bollinger bands
        group = sales_remove_outliers(group)
        
        product_data = group[group['Product_ID'] == product_id]
        if len(product_data) == 0:
            st.write(f"No data available for Product ID: {product_id}")
            continue
        
        X = product_data['Monetary_Value'].values.reshape(-1, 1)
        y = pd.to_datetime(product_data['Sales_Date']).astype(int)  # Convert dates to timestamps
        
        model = LinearRegression()
        
        if len(X) == 0 or len(y) == 0:
            st.write(f"Not enough data for Product ID: {product_id}")
            continue
        
        model.fit(X, y)
        forecasted_sales = model.predict(X)
        ax.scatter(X, y, color='blue', alpha=0.5, label='Actual')
        ax.plot(X, forecasted_sales, color='red', alpha=0.5, label='Forecast')
        ax.set_xlabel('Monetary Value')
        ax.set_ylabel('Sales Date')
        ax.set_title(f'Linear Regression Forecast for Product ID: {product_id}')
        ax.legend()
        st.pyplot(fig)

def main():
    st.title("Makes Business Analytics X")
    
    # Introductory text box
    intro_text = """
    Welcome to Makes Business Analytics X!

    Makes Business Analytics X is an advanced recommendation system web application designed to optimize customer engagement and enhance business profitability. By leveraging cutting-edge machine learning algorithms, Makes RecSys Pro offers a comprehensive suite of analysis tools, including:

    1. Personalized Product Recommendations: Tailors product recommendations to individual customers based on their historical purchasing behavior, fostering customer satisfaction and loyalty.
    2. Dynamic RFM Analysis: Utilizes RFM (Recency, Frequency, Monetary) analysis to segment customers and identify high-value customer segments, enabling targeted marketing strategies for maximum revenue generation.
    3. Client Similarity Analysis: Identifies groups of clients with similar purchasing behavior, allowing businesses to create tailored marketing campaigns and personalized recommendations to improve customer satisfaction.
    4. Product Performance Analysis: Evaluates the performance of products based on their recency, frequency, and monetary value, empowering businesses to optimize inventory management and drive sales growth.

    Experience the power of data-driven decision-making with Makes RecSys Pro! Simply upload your CSV file to get started.
    """
    st.markdown(intro_text)
    
    # Generate unique keys for file uploaders
    upload_key = "Upload CSV File"

    # Upload CSV file
    file = st.sidebar.file_uploader(upload_key, key="file_uploader", type=["csv"])
    if file is not None:
        df = sales_read_data(file)
        interaction_matrix = None  # Define interaction_matrix
        df = sales_convert_to_timestamps(df)
        rfm_scores = sales_calculate_rfm_scores(df)
        rfm_scores['Geometric_Mean'] = sales_calculate_geometric_mean(rfm_scores[['Recency', 'Frequency', 'Monetary']])
        grouped_data = df.groupby('Product_ID')
        product_features = pd.DataFrame(columns=['Angular_Coefficient', 'Mean_RFM_Score', 'Cluster'])
        for product_id, group in grouped_data:
            X = group['Monetary_Value'].values.reshape(-1, 1)
            y = pd.to_datetime(group['Sales_Date']).astype(int)
            model = LinearRegression()
            model.fit(X, y)
            angular_coefficient = model.coef_[0]
            mean_rfm_score = rfm_scores.loc[product_id, 'Geometric_Mean']
            product_features.loc[product_id] = [angular_coefficient, mean_rfm_score, np.nan]
        product_features = sales_perform_clustering(product_features)

        # Create tabs for different analyses
        tabs = ["Recommendations: Clients to Products", "Recommendations: Products to Clients", "RFM Analysis",
                "Client Similarity Analysis", "Product Analysis", "RFM Trends Analysis", "ABC Analysis", "Sales Performance"]
        selected_tab = st.sidebar.radio("Select Analysis", tabs)
        
        if selected_tab == "Recommendations: Clients to Products":
            # Recommendations: Clients to Products
            st.subheader("Recommendations for Clients to Products")
            st.markdown("""
            This analysis provides personalized product recommendations for clients based on their historical purchasing behavior. By leveraging machine learning algorithms, it identifies products that are likely to be of interest to individual clients, enabling businesses to offer tailored recommendations and enhance customer satisfaction.
            """)
            if interaction_matrix is None:
                interaction_matrix = create_interaction_matrix(df)  # Define interaction_matrix if not defined
            train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)
            autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
            train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)
            encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
            embeddings = get_embeddings(encoder_model, interaction_matrix.values)
            
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
                    st.table(pd.DataFrame(table_data, columns=["Product ID", "Product Name", "Brand"]))
        
        elif selected_tab == "Recommendations: Products to Clients":
            # Recommendations: Products to Clients
            st.subheader("Recommendations for Products to Clients")
            st.markdown("""
            This analysis provides recommendations for products that are likely to be purchased by clients based on their preferences and historical buying patterns. By understanding the interests of individual clients, businesses can optimize their product offerings and enhance customer engagement.
            """)
            if interaction_matrix is None:
                interaction_matrix = create_interaction_matrix(df)  # Define interaction_matrix if not defined
            train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)
            autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
            train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)
            encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
            embeddings = get_embeddings(encoder_model, interaction_matrix.values)
            
            for product_id in interaction_matrix.columns:
                with st.expander(f"Product {product_id} Recommendations"):
                    recommended_clients = recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)
                    if len(recommended_clients) > 0:
                        recall_at_k = len(set(recommended_clients).intersection(interaction_matrix.index)) / len(interaction_matrix.index)
                        table_data = []
                        for client_id in recommended_clients:
                            table_data.append([client_id])
                        st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                        st.table(pd.DataFrame(table_data, columns=["Client ID"]))
                    else:
                        st.write("No recommendations for this product.")
        
        elif selected_tab == "RFM Analysis":
            # RFM Analysis
            st.subheader("RFM Analysis")
            st.markdown("""
            RFM (Recency, Frequency, Monetary) analysis segments customers based on their purchasing behavior. It helps businesses identify high-value customers, understand their preferences, and tailor marketing strategies to enhance customer loyalty and maximize revenue.
            """)
            rfm_data = compute_rfm(df)
            rfm_data, silhouette_avg = perform_clustering(rfm_data)
            plot_rfm_clusters(rfm_data)
            print_rfm_results_table(rfm_data, silhouette_avg)
        
        elif selected_tab == "Client Similarity Analysis":
            # Client Similarity Analysis
            st.subheader("Client Similarity Analysis")
            st.markdown("""
            Client similarity analysis identifies groups of clients with similar purchasing behavior. By segmenting clients based on their preferences and habits, businesses can create targeted marketing campaigns, personalized recommendations, and improve customer satisfaction.
            """)
            if interaction_matrix is None:
                interaction_matrix = create_interaction_matrix(df)  # Define interaction_matrix if not defined
            result_table = client_to_client_recommender(interaction_matrix)
            organized_table = []
            for client_id, recommended_clients, recall_at_k in result_table:
                organized_table.append([client_id, ", ".join(map(str, recommended_clients)), recall_at_k])
            headers = ["Client ID", "Similar Clients", "Similarity Score"]
            st.table(pd.DataFrame(organized_table, columns=headers))
            
        elif selected_tab == "RFM Trends Analysis":
            # RFM Trends Analysis
            st.subheader("RFM Trends Analysis")
            st.markdown("""
            This analysis examines the RFM trends of clients over time and clusters them based on their RFM trend patterns. It provides insights into customer behavior and identifies segments of clients with similar RFM trend trajectories.
            """)

            # Preprocess RFM data
            rfm = preprocess_rfm_data(df)

            # Cluster clients
            angular_coefficients_df = cluster_clients(rfm)

            # Plot clustering results
            plot_cluster_results(angular_coefficients_df)
            
            # Print clustering results table
            print_clustering_results_table(angular_coefficients_df)

            # Forecast RFM trends
            forecast_rfm_trends(rfm, angular_coefficients_df)

        # Run the selected analysis
        elif selected_tab == "Product Analysis":
            st.subheader("Product Analysis")

            # Compute RFM metrics for products
            product_rfm_data = compute_product_rfm(df)

            # Perform KMeans clustering on product RFM data
            product_rfm_data, silhouette_avg = perform_product_clustering(product_rfm_data)

            # Plot product clusters
            plot_product_clusters(product_rfm_data)

            # Print product clustering results table
            print_product_results_table(product_rfm_data, silhouette_avg)
            
        elif selected_tab == "ABC Analysis":
            st.subheader("ABC Analysis")
            st.markdown("""
            ABC analysis categorizes products or clients into three groups (A, B, and C) based on their importance or contribution to the business. It helps in prioritizing resources and optimizing inventory management.
            """)
            product_sales = calculate_product_sales(df)
            product_sales['ABC_Class'] = product_sales.apply(classify_product, axis=1)
            plot_product_clustering(product_sales)
            product_sales = add_reorder_and_frequency(product_sales)
            print_results(product_sales)
            
        elif selected_tab == "Sales Performance":
            # Sales Performance Analysis
            st.subheader("Sales Performance Analysis")
            st.markdown("""
            This analysis focuses on evaluating the performance of products based on their sales data. It includes RFM analysis, clustering of products, and forecast of sales trends using linear regression.
            """)
            
            # Plot clustering results
            sales_plot_clusters(product_features)
            
            # Print the table of each product with its features
            sales_print_product_features(product_features)
            
            # Plot real data against forecasted values for each product ID
            sales_plot_forecast(grouped_data, rfm_scores, num_products=10)
        

if __name__ == "__main__":
    main()

