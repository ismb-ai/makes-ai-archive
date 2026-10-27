import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from tabulate import tabulate

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
    plt.figure(figsize=(10, 6))
    for cluster_id, label in cluster_labels.items():
        cluster_data = product_features[product_features['Cluster'] == cluster_id]
        plt.scatter(cluster_data.index, cluster_data['Mean_RFM_Score'], label=label)
    plt.xlabel('Product ID')
    plt.ylabel('Mean RFM Score')
    plt.title('K-means Clustering of Product Sales')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

# Function to print the table of each product with its features
def sales_print_product_features(product_features):
    table = product_features.reset_index().rename(columns={'index': 'Product ID'})
    print(tabulate(table, headers='keys', tablefmt='pretty'))

# Function to remove outliers using Bollinger bands
def sales_remove_outliers(group):
    std_dev = group['Monetary_Value'].std()
    mean = group['Monetary_Value'].mean()
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    return group[(group['Monetary_Value'] >= lower_bound) & (group['Monetary_Value'] <= upper_bound)]

# Function to plot real data against forecasted values for each product ID
def sales_plot_forecast(grouped_data, rfm_scores, num_products=10):
    product_ids = list(grouped_data.groups.keys())
    sampled_ids = np.random.choice(product_ids, size=min(num_products, len(product_ids)), replace=False)
    for product_id in sampled_ids:
        plt.figure(figsize=(8, 5))
        group = grouped_data.get_group(product_id)
        
        # Remove outliers using Bollinger bands
        group = sales_remove_outliers(group)
        
        product_data = group[group['Product_ID'] == product_id]
        if len(product_data) == 0:
            print(f"No data available for Product ID: {product_id}")
            continue
        
        X = product_data['Monetary_Value'].values.reshape(-1, 1)
        y = pd.to_datetime(product_data['Sales_Date']).astype(int)  # Convert dates to timestamps
        
        model = LinearRegression()
        
        if len(X) == 0 or len(y) == 0:
            print(f"Not enough data for Product ID: {product_id}")
            continue
        
        model.fit(X, y)
        forecasted_sales = model.predict(X)
        plt.scatter(X, y, color='blue', alpha=0.5, label='Actual')
        plt.plot(X, forecasted_sales, color='red', alpha=0.5, label='Forecast')
        plt.xlabel('Monetary Value')
        plt.ylabel('Sales Date')
        plt.title(f'Linear Regression Forecast for Product ID: {product_id}')
        plt.legend()
        plt.show()


# Main function to perform analysis
def sales_main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    
    # Read data
    data = sales_read_data(file_path)
    
    # Preprocess data
    data = sales_convert_to_timestamps(data)
    
    # Calculate RFM scores
    rfm_scores = sales_calculate_rfm_scores(data)
    
    # Calculate geometric mean of RFM scores
    rfm_scores['Geometric_Mean'] = sales_calculate_geometric_mean(rfm_scores[['Recency', 'Frequency', 'Monetary']])
    
    # Group the data by 'Product_ID'
    grouped_data = data.groupby('Product_ID')
    
    # Create a DataFrame to store features
    product_features = pd.DataFrame(columns=['Angular_Coefficient', 'Mean_RFM_Score', 'Cluster'])
    
    # Populate the DataFrame with features for each product
    for product_id, group in grouped_data:
        X = group['Monetary_Value'].values.reshape(-1, 1)
        y = pd.to_datetime(group['Sales_Date']).astype(int)  # Convert dates to timestamps
        model = LinearRegression()
        model.fit(X, y)
        angular_coefficient = model.coef_[0]
        mean_rfm_score = rfm_scores.loc[product_id, 'Geometric_Mean']
        product_features.loc[product_id] = [angular_coefficient, mean_rfm_score, np.nan]
    
    # Perform clustering
    product_features = sales_perform_clustering(product_features)
    
    # Plot clustering results
    sales_plot_clusters(product_features)
    
    # Print the table of each product with its features
    sales_print_product_features(product_features)
    
    # Plot real data against forecasted values for each product ID
    sales_plot_forecast(grouped_data, rfm_scores, num_products=10)

if __name__ == "__main__":
    sales_main()
