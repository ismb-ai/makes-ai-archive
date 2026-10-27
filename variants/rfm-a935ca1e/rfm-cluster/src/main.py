# RFM Clustering

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tabulate import tabulate
import streamlit as st

# Hide the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define a function to load data from a CSV file
def load_data(file_path):
    # Read the CSV file into a DataFrame using pandas
    df = pd.read_csv(file_path)
    
    # Convert the 'Sale_Date' column to datetime format for better handling
    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
    
    # Return the DataFrame with the loaded data
    return df

# Define a function to compute RFM (Recency, Frequency, Monetary) metrics for 
# clients based on sales data
def compute_rfm(df):
    # Determine the current date from the maximum sale date in the DataFrame
    current_date = df['Sale_Date'].max()
    
    # Group the DataFrame by client ID and aggregate metrics
    rfm_data = df.groupby('Client_ID').agg({
        'Sale_Date': lambda x: (current_date - x.max()).days,  # Calculate recency
        'Product_ID': 'count',  # Count frequency of purchases
        'Sale_Value': 'sum'  # Calculate total monetary value of purchases
    }).reset_index()
    
    # Divide the recency, frequency, and monetary values into deciles and assign labels
    rfm_data['Recency'] = pd.qcut(rfm_data['Sale_Date'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Monetary'] = pd.qcut(rfm_data['Sale_Value'], q=10, labels=False, duplicates='drop') + 1
    
    # Calculate RFM score by summing up recency, frequency, and monetary scores
    rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']
    
    # Normalize RFM scores to a 0-100 scale
    rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / \
                                       (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
    
    # Return the computed RFM metrics DataFrame
    return rfm_data


# Define a function to perform clustering on RFM data using KMeans algorithm
def perform_clustering(rfm_data):
    # Initialize a KMeans clustering object with 3 clusters, a fixed random state for reproducibility, and 10 initializations
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    # Fit the KMeans model to the normalized RFM scores and assign cluster labels
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_data[['RFM_Score_Normalized']])
    
    # Return the DataFrame with cluster labels assigned
    return rfm_data

def plot_clusters(rfm_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {0: '#FFFFED',  # Diamond
              1: '#FFD700',  # Gold
              2: '#CD7F32'}  # Silver

    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}

    for cluster, color in colors.items():
        cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
        ax.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Client_ID'], color=color, label=cluster_labels[cluster])

    ax.set_xlabel('RFM Score (Normalized)')
    ax.set_ylabel('Client ID')
    ax.set_title('Clustering of Clients')
    ax.legend()
    ax.grid(True)
    
    # Set background color to blackboard green
    ax.set_facecolor('#21421E')

    # Show plot
    st.pyplot(fig)

def print_results_table(rfm_data):
    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
    rfm_data['RFM_Label'] = rfm_data['Cluster'].map(cluster_labels)
    
    table_data = rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label']]
    table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
    st.write("RFM Clustering Results:")
    st.dataframe(table_data)

    # Print WCSS
    st.write("Within-cluster Sum of Squares (WCSS):")
    st.write(rfm_data.groupby('Cluster')['RFM_Score_Normalized'].sum())

def main():
    st.title("RFM Clustering Analysis")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        # Compute RFM
        rfm_data = compute_rfm(df)

        # Perform clustering
        rfm_data = perform_clustering(rfm_data)

        # Plot clusters
        plot_clusters(rfm_data)

        # Print results table
        print_results_table(rfm_data)

if __name__ == "__main__":
    main()
