# k-Means Clustering to Classify Products According to Profitability

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

def load_data(file_path):
    # Load the CSV file into a pandas DataFrame.
    return pd.read_csv(file_path)

def compute_profitability(data_frame):
    # Compute profitability and add it as a new column.
    data_frame['Profitability'] = data_frame['Monetary_Value'] - data_frame['Product_Costs']

def create_cluster_data(data_frame):
    # Create a DataFrame for clustering with relevant columns.
    return data_frame[['Product_ID', 'Profitability', 'Product_Name', 'Brand_Name']].groupby('Product_ID').agg({
        'Profitability': 'sum',
        'Product_Name': 'first',
        'Brand_Name': 'first'
    }).reset_index()

def standardize_data(data_frame):
    # Standardize the data (important for k-means).
    scaler = StandardScaler()
    data_frame['Profitability'] = scaler.fit_transform(data_frame['Profitability'].values.reshape(-1, 1))

def apply_kmeans_clustering(data_frame, n_clusters=3, random_state=42, n_init=10):
    # Apply k-means clustering with explicit n_init parameter.
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    data_frame['Cluster_Label'] = kmeans.fit_predict(data_frame[['Profitability']])

def map_cluster_labels(data_frame, label_mapping):
    # Map numerical cluster labels to categorical labels.
    data_frame['Cluster_Label'] = data_frame['Cluster_Label'].map(label_mapping)

def create_result_dataframe(cluster_data):
    # Create a new DataFrame with relevant columns.
    return cluster_data[['Product_ID', 'Product_Name', 'Brand_Name', 'Cluster_Label']]

def save_result_dataframe_as_text(result_data_frame, file_path):
    # Save the resulting DataFrame as a text file using tabulate.
    with open(file_path, 'w') as file:
        file.write(tabulate(result_data_frame, headers='keys', tablefmt='plain', showindex=False))

def print_confirmation_message(file_path):
    # Print a confirmation message.
    print(f"Resulting DataFrame saved to: {file_path}")

def main():
    # Specify the file path.
    file_path = "/Volumes/Untitled/cpp-complete.csv"

    # Load the CSV file into a pandas DataFrame.
    df = load_data(file_path)

    # Compute profitability and add it as a new column.
    compute_profitability(df)

    # Create a DataFrame for clustering with relevant columns.
    cluster_data = create_cluster_data(df)

    # Standardize the data (important for k-means).
    standardize_data(cluster_data)

    # Apply k-means clustering with explicit n_init parameter.
    apply_kmeans_clustering(cluster_data, n_clusters=3, random_state=42, n_init=10)

    # Map numerical cluster labels to categorical labels.
    label_mapping = {0: 'High Profitability', 1: 'Average Profitability', 2: 'Low Profitability'}
    map_cluster_labels(cluster_data, label_mapping)

    # Create a new DataFrame with relevant columns.
    result_df = create_result_dataframe(cluster_data)

    # Save the resulting DataFrame as a text file using tabulate.
    output_file_path = "/Volumes/Untitled/cpp-products-profitability.txt"
    save_result_dataframe_as_text(result_df, output_file_path)

    # Print a confirmation message.
    print_confirmation_message(output_file_path)

if __name__ == "__main__":
    main()
