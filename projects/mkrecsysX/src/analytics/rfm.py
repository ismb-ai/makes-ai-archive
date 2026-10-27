# rfm.def

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

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
