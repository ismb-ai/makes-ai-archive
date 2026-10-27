# k-means Clustering algorithm to classify clients according to RFM score

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

def load_main_dataframe(file_path):
    # Load the main DataFrame.
    return pd.read_csv(file_path)

def extract_features_for_clustering(data_frame):
    # Extract the relevant columns for clustering.
    return data_frame[['RFM_Score', 'Client_ID']]

def standardize_data(data):
    # Standardize the data using StandardScaler.
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def apply_kmeans_clustering(data, n_clusters=3, random_state=42, n_init=10):
    # Apply k-means clustering with explicit n_init value.
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    return kmeans.fit_predict(data)

def map_cluster_labels(data_frame, cluster_labels_mapping):
    # Map cluster labels to new meaningful names.
    data_frame['Cluster_Label'] = data_frame['Cluster'].map(cluster_labels_mapping)

def load_additional_dataframe(file_path, columns_with_mixed_types):
    # Load the additional DataFrame with client names.
    return pd.read_csv(file_path, dtype={col: str for col in columns_with_mixed_types}, low_memory=False)

def merge_dataframes(main_dataframe, additional_dataframe):
    # Merge the main DataFrame with the additional DataFrame based on 'Client_ID' and 'id'.
    return pd.merge(main_dataframe, additional_dataframe, left_on='Client_ID', right_on='id', how='left')

def create_table(data_frame):
    # Create a table with Client_ID, RFM_Score, Cluster_Label, and client names.
    return data_frame[['Client_ID', 'RFM_Score', 'Cluster_Label', 'nome']]

def save_table_as_text(data_table, file_path):
    # Save the table as a text file using tabulate.
    table_str = tabulate(data_table, headers='keys', tablefmt='plain', showindex=False)
    with open(file_path, 'w') as txt_file:
        txt_file.write(table_str)

def print_confirmation_message(file_path):
    # Print a confirmation message.
    print(f"Table saved to {file_path}")

def create_scatter_plot(data_frame):
    # Create a scatter plot to visualize the clusters.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Client_ID', y='RFM_Score', hue='Cluster', data=data_frame, palette='viridis', legend='full')
    plt.title('RFM Clustering')
    plt.xlabel('Client ID')
    plt.ylabel('RFM Score')
    plt.show()

def main():
    # Load the main DataFrame.
    main_df = load_main_dataframe("/Volumes/Untitled/cpp-complete.csv")

    # Extract the relevant columns for clustering.
    features_for_clustering = extract_features_for_clustering(main_df)

    # Standardize the data using StandardScaler.
    scaled_data = standardize_data(features_for_clustering)

    # Apply k-means clustering with explicit n_init value.
    clusters = apply_kmeans_clustering(scaled_data)

    # Map cluster labels to new meaningful names.
    cluster_labels_mapping = {0: "SILVER", 1: "GOLD", 2: "DIAMOND"}
    main_df['Cluster'] = clusters
    map_cluster_labels(main_df, cluster_labels_mapping)

    # Load the additional DataFrame with client names.
    columns_with_mixed_types = ['column_name1', 'column_name2', ...]  # Replace with actual column names
    clients_df = load_additional_dataframe("/Volumes/Untitled/account_202401081403.csv", columns_with_mixed_types)

    # Merge the main DataFrame with the additional DataFrame based on 'Client_ID' and 'id'.
    merged_df = merge_dataframes(main_df, clients_df)

    # Create a table with Client_ID, RFM_Score, Cluster_Label, and client names.
    table_data = create_table(merged_df)

    # Save the table as a text file using tabulate.
    txt_file_path = "/Volumes/Untitled/client-classification.txt"
    save_table_as_text(table_data, txt_file_path)

    # Print a confirmation message.
    print_confirmation_message(txt_file_path)

    # Create a scatter plot to visualize the clusters.
    create_scatter_plot(main_df)

if __name__ == "__main__":
    main()
