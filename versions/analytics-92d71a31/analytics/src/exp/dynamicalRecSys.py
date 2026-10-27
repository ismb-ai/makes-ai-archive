import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

def read_data(file_path):
    return pd.read_csv(file_path)

def convert_to_timestamps(df):
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

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
        if len(group) < 2:
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
    cluster_labels = {0: "Loyal customer", 1: "Stable customer", 2: "Customer at risk"}
    angular_coefficients_df['Cluster_Label'] = angular_coefficients_df['Cluster'].map(cluster_labels)
    return angular_coefficients_df

def plot_cluster_results(cluster_df, id_column, title):
    plt.figure(figsize=(10, 6))
    for label, color in zip(cluster_df['Cluster_Label'].unique(), ['blue', 'green', 'red']):
        subset = cluster_df[cluster_df['Cluster_Label'] == label]
        plt.scatter(subset[id_column], subset['Angular_Coefficient'], label=label, c=color)
    plt.xlabel(id_column)
    plt.ylabel('Angular Coefficient')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=90)
    st.pyplot()

def print_clustering_results_table(cluster_df):
    st.write(tabulate(cluster_df, headers='keys', tablefmt='pipe'))

def calculate_product_rfm_scores(df):
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    current_date = df['Sales_Date'].max()
    rfm = df.groupby(['Product_ID', pd.Grouper(key='Sales_Date', freq='M')]).agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,
        'Client_ID': 'count',
        'Monetary_Value': 'sum'
    }).rename(columns={'Sales_Date': 'Recency', 'Client_ID': 'Frequency', 'Monetary_Value': 'Monetary'}).reset_index()
    return rfm

def cluster_products(rfm):
    angular_coefficients = []
    for product_id, group in rfm.groupby('Product_ID'):
        if len(group) < 2:
            continue
        X = group.index.values.reshape(-1, 1)
        y = group['Recency'].values
        lr = LinearRegression()
        lr.fit(X, y)
        angular_coefficient = lr.coef_[0]
        angular_coefficients.append([product_id, angular_coefficient])
    angular_coefficients_df = pd.DataFrame(angular_coefficients, columns=['Product_ID', 'Angular_Coefficient'])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(angular_coefficients_df[['Angular_Coefficient']])
    angular_coefficients_df['Cluster'] = kmeans.labels_
    cluster_labels = {0: "High performance", 1: "Stable performance", 2: "Low performance"}
    angular_coefficients_df['Cluster_Label'] = angular_coefficients_df['Cluster'].map(cluster_labels)
    return angular_coefficients_df

def recommend_products_to_clients(df, client_clusters_df, product_clusters_df, client_cluster_choice, product_cluster_choice):
    selected_clients = client_clusters_df[client_clusters_df['Cluster_Label'] == client_cluster_choice]['Client_ID']
    selected_products = product_clusters_df[product_clusters_df['Cluster_Label'] == product_cluster_choice]['Product_ID']

    recommendations = []
    for client_id in selected_clients:
        client_purchases = df[df['Client_ID'] == client_id]
        recommended_products = selected_products[~selected_products.isin(client_purchases['Product_ID'])]
        recommendations.append((client_id, recommended_products.values))

    recommendation_df = pd.DataFrame(recommendations, columns=['Client_ID', 'Recommended_Products'])
    st.write("Recommendations:")
    st.write(tabulate(recommendation_df, headers='keys', tablefmt='pipe'))


# Main function to run the app
def main():
    # Step 1: Read and preprocess the sales data
    file_path = '/Users/igormol/Desktop/analytics/sample/CPP.csv'  # Replace with your CSV file path
    df = read_data(file_path)
    df = convert_to_timestamps(df)

    st.title("Sales Data Analysis and Recommendations")

    # Step 2: Calculate and cluster the RFM trends for clients
    st.subheader("Client RFM Trends Analysis")
    rfm_client = preprocess_rfm_data(df)
    client_clusters_df = cluster_clients(rfm_client)
    plot_cluster_results(client_clusters_df, "Client_ID", "Client RFM Trends Clustering")
    print_clustering_results_table(client_clusters_df)

    # Step 3: Calculate and cluster the RFM trends for products
    st.subheader("Product RFM Trends Analysis")
    rfm_product = calculate_product_rfm_scores(df)
    product_clusters_df = cluster_products(rfm_product)
    plot_cluster_results(product_clusters_df, "Product_ID", "Product RFM Trends Clustering")
    print_clustering_results_table(product_clusters_df)

    # Step 4: Allow user selection and recommend products to clients
    st.subheader("Recommendations")
    client_cluster_choice = st.selectbox("Choose a client cluster", client_clusters_df['Cluster_Label'].unique())
    product_cluster_choice = st.selectbox("Choose a product cluster", product_clusters_df['Cluster_Label'].unique())

    recommend_products_to_clients(df, client_clusters_df, product_clusters_df, client_cluster_choice, product_cluster_choice)

if __name__ == "__main__":
    main()
