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
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id, label in cluster_labels.items():
        cluster_data = product_features[product_features['Cluster'] == cluster_id]
        ax.scatter(cluster_data.index, cluster_data['Mean_RFM_Score'], label=label)
    ax.set_xlabel('Product ID')
    ax.set_ylabel('Mean RFM Score')
    ax.set_title('K-means Clustering of Product Sales')
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    st.pyplot(fig)


# Function to print the table of each product with its features
def sales_print_product_features(product_features):
    table = product_features.reset_index().rename(columns={'index': 'Product ID'})
    st.write(tabulate(table, headers='keys', tablefmt='pipe'))

# Function to remove outliers using Bollinger bands
def sales_remove_outliers(group):
    std_dev = group['Monetary_Value'].std()
    mean = group['Monetary_Value'].mean()
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    return group[(group['Monetary_Value'] >= lower_bound) & (group['Monetary_Value'] <= upper_bound)]

# Plot sales forecast
def sales_plot_forecast(grouped_data, rfm_scores, num_products=10):
    product_ids = list(grouped_data.groups.keys())
    sampled_ids = np.random.choice(product_ids, size=min(num_products, len(product_ids)), replace=False)
    for product_id in sampled_ids:
        fig, ax = plt.subplots(figsize=(8, 5))
        group = grouped_data.get_group(product_id)

        # Remove outliers using Bollinger bands
        group = sales_remove_outliers(group)

        product_data = group[group['Product_ID'] == product_id]
        if len(product_data) == 0:
            st.write(f"No data available for Product ID: {product_id}")
            continue

        X = product_data['Monetary_Value'].values.reshape(-1, 1)
        y = pd.to_datetime(product_data['Sales_Date']).astype(int)  # Convert dates to timestamps

        model = LinearRegression()

        if len(X) == 0 or len(y) == 0:
            st.write(f"Not enough data for Product ID: {product_id}")
            continue

        model.fit(X, y)
        forecasted_sales = model.predict(X)
        ax.scatter(X, y, color='blue', alpha=0.5, label='Actual')
        ax.plot(X, forecasted_sales, color='red', alpha=0.5, label='Forecast')
        ax.set_xlabel('Monetary Value')
        ax.set_ylabel('Sales Date')
        ax.set_title(f'Linear Regression Forecast for Product ID: {product_id}')
        ax.legend()
        st.pyplot(fig)
