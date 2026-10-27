# RFM Clustering Analysis Web App
#
# Igor Mol <igor.mol@makes.ai>
#
# In this program, given a data frame consisting of sales inventory information,
# uploaded by the user of our Streamlit web app as a CSV file, we compute the 
# RFM scores for each client ID. Then, we apply a k-means clustering algorithm
# to cluster the clients according to their RFM scores into the (obvious) labels
# "Diamond", "Gold" and "Silver". We then compute the distance to the centroid
# metric for each client, convert it to a scale ranging from 0 to 10, and round
# it up to an integer. Finally, we display the results into our web app using a
# cluster plot and a table of client IDs, RFM scores, labels and distance to the
# centroid.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
    
    # Divide the recency, frequency, and monetary values into deciles
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
    
    # Compute centroids
    centroids = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    # Compute distance to centroid for each client
    rfm_data['Distance_to_Centroid'] = rfm_data.apply(
        lambda row: np.linalg.norm(row[['Recency', 'Frequency', 'Monetary']] - centroids.loc[row['Cluster']]), axis=1)
    
    # Scale distance to centroid to range from 0 to 10
    max_distance = rfm_data['Distance_to_Centroid'].max()
    min_distance = rfm_data['Distance_to_Centroid'].min()
    rfm_data['Distance_to_Centroid_Scaled'] = ((rfm_data['Distance_to_Centroid'] - min_distance) / (max_distance - min_distance)) * 10
    
    # Round scaled distance to centroid to integer
    rfm_data['Distance_to_Centroid_Scaled'] = rfm_data['Distance_to_Centroid_Scaled'].round().astype(int)
    
    table_data = rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Distance_to_Centroid_Scaled']]
    table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
    st.write("RFM Clustering Results:")
    st.dataframe(table_data)

    # Print legend
    st.write("**Legend:**")
    st.write("- RFM Score: Higher values indicate better recency, frequency, and monetary metrics.")
    st.write("- RFM Label: Represents the cluster label assigned to each client (Diamond, Gold, Silver).")
    st.write("- Distance to Centroid: Indicates how close each client is to the centroid of their cluster. Smaller values imply closer proximity.")

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
