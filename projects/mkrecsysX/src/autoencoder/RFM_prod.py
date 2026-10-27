import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from tabulate import tabulate

def load_product_data(file_path):
    df = pd.read_csv(file_path)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

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

def perform_product_clustering(product_rfm_data):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    product_rfm_data['Cluster'] = kmeans.fit_predict(product_rfm_data[['RFM_Score_Normalized']])
    silhouette_avg = silhouette_score(product_rfm_data[['RFM_Score_Normalized']], product_rfm_data['Cluster'])
    cluster_silhouette_scores = silhouette_samples(product_rfm_data[['RFM_Score_Normalized']], product_rfm_data['Cluster'])
    product_rfm_data['Silhouette'] = cluster_silhouette_scores
    return product_rfm_data, silhouette_avg

def plot_product_clusters(product_rfm_data):
    plt.figure(figsize=(10, 6))
    
    colors = {0: '#4C72B0',  # Diamond - blue
              1: '#55A868',  # Gold - green
              2: '#C44E52'}  # Silver - red

    for cluster, color in colors.items():
        cluster_data = product_rfm_data[product_rfm_data['Cluster'] == cluster]
        plt.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Product_ID'], color=color, label=f'Cluster {cluster}')
        
        # Compute the mean silhouette score for the cluster
        cluster_mean_silhouette = cluster_data['Silhouette'].mean()
        # Plot the silhouette score as an error bar
        plt.errorbar(cluster_mean_silhouette, max(cluster_data['Product_ID']), xerr=cluster_data['Silhouette'].std(), fmt='o', color='black')

    plt.xlabel('RFM Score (Normalized)')
    plt.ylabel('Product ID')
    plt.title('Clustering of Products')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_product_results_table(product_rfm_data, silhouette_avg):
    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
    product_rfm_data['RFM_Label'] = product_rfm_data['Cluster'].map(cluster_labels)
    
    table_data = product_rfm_data[['Product_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Silhouette']]
    table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
    print("Product RFM Clustering Results:")
    print(tabulate(table_data, headers='keys', tablefmt='grid'))
    print(f"Average Silhouette Score: {silhouette_avg}")

def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_product_data(file_path)
    product_rfm_data = compute_product_rfm(df)
    product_rfm_data, silhouette_avg = perform_product_clustering(product_rfm_data)
    plot_product_clusters(product_rfm_data)
    print_product_results_table(product_rfm_data, silhouette_avg)

if __name__ == "__main__":
    main()

