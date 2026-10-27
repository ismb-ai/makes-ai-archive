import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from tabulate import tabulate

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

def compute_rfm(df):
    current_date = df['Sales_Date'].max()
    rfm_data = df.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Recency
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

def plot_clusters(rfm_data):
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
    plt.title('Clustering of Clients')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_results_table(rfm_data, silhouette_avg):
    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
    rfm_data['RFM_Label'] = rfm_data['Cluster'].map(cluster_labels)
    
    table_data = rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Silhouette']]
    table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
    print("RFM Clustering Results:")
    print(tabulate(table_data, headers='keys', tablefmt='grid'))
    print(f"Average Silhouette Score: {silhouette_avg}")

def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)
    rfm_data = compute_rfm(df)
    rfm_data, silhouette_avg = perform_clustering(rfm_data)
    plot_clusters(rfm_data)
    print_results_table(rfm_data, silhouette_avg)

if __name__ == "__main__":
    main()
