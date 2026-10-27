# rfm_analysis.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This program contains functions for loading, analyzing, and visualizing RFM
# (Recency, Frequency, Monetary) scores for clients. Our core routine employs
# k-means clustering algorithm to classify clients according to their relative
# "unified" RFM score, latter being defined as the RFM's arithmetic means.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.cluster import KMeans
import numpy as np
from tabulate import tabulate

# Function to load and pre-process dataframe
def load_data(file_path):
    # Read data from a CSV file located at the specified file_path
    df = pd.read_csv(file_path)

    # Convert the 'Sales_Date' column to datetime format
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])

    # Return the DataFrame containing the loaded data
    return df

# Compute the RFM scores of each client
def compute_rfm(df):
    # Find the maximum date in the 'Sales_Date' column
    current_date = df['Sales_Date'].max()

    # Group the DataFrame by 'Client_ID' and aggregate the data
    rfm_data = df.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Calculate Recency
        'Product_ID': 'count',  # Count the frequency of purchases (Frequency)
        'Monetary_Value': 'sum'  # Calculate Monetary value
    }).reset_index()

    # Calculate Recency, Frequency, and Monetary based on quantiles
    rfm_data['Recency'] = pd.qcut(rfm_data['Sales_Date'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Monetary'] = pd.qcut(rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1

    # Calculate RFM Score by summing Recency, Frequency, and Monetary scores
    rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']

    # Normalize RFM Score to a scale of 0 to 100
    rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100

    # Return the DataFrame containing RFM data
    return rfm_data

# Apply k-means clustering algorithm to perform client clustering based on RFM
def perform_clustering(rfm_data):
    # Initialize KMeans clustering algorithm with specified parameters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    # Perform clustering on the normalized RFM scores and assign cluster labels
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_data[['RFM_Score_Normalized']])

    # Calculate the average silhouette score for the clustering
    silhouette_avg = silhouette_score(rfm_data[['RFM_Score_Normalized']], rfm_data['Cluster'])

    # Calculate silhouette scores for each sample (client) in the dataset
    cluster_silhouette_scores = silhouette_samples(rfm_data[['RFM_Score_Normalized']], rfm_data['Cluster'])

    # Assign silhouette scores to the DataFrame
    rfm_data['Silhouette'] = cluster_silhouette_scores

    # Return the DataFrame with cluster labels and silhouette scores, along with the average silhouette score
    return rfm_data, silhouette_avg

# Plot RFM clusters such that the user can easliy visualize the results
def plot_rfm_clusters(rfm_data):
    # Create a new figure for the plot with specified size
    plt.figure(figsize=(10, 6))

    # Define colors for each cluster
    colors = {0: '#4C72B0', 1: '#55A868', 2: '#C44E52'}

    # Iterate over each cluster and its corresponding color
    for cluster, color in colors.items():
        # Filter data for the current cluster
        cluster_data = rfm_data[rfm_data['Cluster'] == cluster]

        # Scatter plot of RFM scores for clients in the current cluster
        plt.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Client_ID'], color=color, label=f'Cluster {cluster}')

        # Calculate the mean silhouette score for the current cluster
        cluster_mean_silhouette = cluster_data['Silhouette'].mean()

        # Add error bars to the plot to represent the variability of silhouette scores within the cluster
        plt.errorbar(cluster_mean_silhouette, max(cluster_data['Client_ID']), xerr=cluster_data['Silhouette'].std(), fmt='o', color='black')

    # Set labels for the x and y axes
    plt.xlabel('RFM Score (Normalized)')
    plt.ylabel('Client ID')

    # Set title for the plot
    plt.title('Clustering of Clients based on RFM Scores')

    # Add legend to the plot
    plt.legend()

    # Enable grid on the plot
    plt.grid(True)

    # Display the plot using Streamlit's st.pyplot() function
    st.pyplot(plt)

# Print the results of the k-means algorithm in table with clients vs. labels
def print_rfm_results_table(rfm_data, silhouette_avg):
    # Define labels for each cluster
    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}

    # Map cluster labels to the 'Cluster' column in the DataFrame
    rfm_data['RFM_Label'] = rfm_data['Cluster'].map(cluster_labels)

    # Select columns for the table and sort the DataFrame by 'RFM_Score_Normalized'
    table_data = rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Silhouette']]
    table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)

    # Display the RFM clustering results in a table format
    st.write("RFM Clustering Results:")
    st.dataframe(table_data)

    # Display the average silhouette score
    st.write(f"Average Silhouette Score: {silhouette_avg}")
