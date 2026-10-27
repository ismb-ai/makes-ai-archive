# trends.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document defines the functions that forecasts the trends of
# RFM scores for clients and products (using a linear regression model). We
# utilize a k-means clustering algorithm to classify clients and products
# with respect to the time-derivatives of the regression model.

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

# Read CSV file into a DataFrame
def sales_read_data(file_path):
    return pd.read_csv(file_path)

# Convert 'Sales_Date' to timestamps
def sales_convert_to_timestamps(df):
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

# Computes RFM scores for each product based on sales data.
def sales_calculate_rfm_scores(df):
    # Get the current date.
    today = pd.to_datetime('today')

    # Group the DataFrame by 'Product_ID' and calculate the recency, frequency, and monetary value for each product.
    rfm = df.groupby('Product_ID').agg({
        'Sales_Date': lambda x: (today - x.max()).days,  # Calculate the number of days between the latest sale date and today (Recency)
        'Client_ID': 'nunique',                         # Count the number of unique clients (Frequency)
        'Monetary_Value': 'sum'                         # Sum the monetary value of sales (Monetary)
    })

    # Rename the columns to better represent the RFM scores.
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Return the DataFrame containing RFM scores for each product.
    return rfm

# Compute the geometric mean of RFM scores
def sales_calculate_geometric_mean(rfm_scores):
    rfm_scores['Geometric_Mean'] = np.power(rfm_scores.prod(axis=1), 1/3)
    return rfm_scores['Geometric_Mean']

# Performs clustering on product features using KMeans algorithm.
def sales_perform_clustering(product_features):
    # Initialize KMeans with 4 clusters, seed 42, and 10 re-initializations.
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

    # Perform clustering on the features 'Angular_Coefficient' and 'Mean_RFM_Score'.
    clusters = kmeans.fit_predict(product_features[['Angular_Coefficient', 'Mean_RFM_Score']])

    # Assign cluster labels to the product features.
    product_features['Cluster'] = clusters

    # Return the product features with cluster assignments.
    return product_features

# Plots clustering results
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

# Prints the table of each product with its features
def sales_print_product_features(product_features):
    table = product_features.reset_index().rename(columns={'index': 'Product ID'})
    st.write(tabulate(table, headers='keys', tablefmt='pipe'))

# Removes outliers from a group of sales data using Bollinger bands.
def sales_remove_outliers(group):
    # Calculate the standard deviation and mean of the 'Monetary_Value' column in the group.
    std_dev = group['Monetary_Value'].std()
    mean = group['Monetary_Value'].mean()

    # Calculate the lower and upper bounds of the Bollinger bands.
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    # Filter the group to include only the data points within the Bollinger bands.
    return group[(group['Monetary_Value'] >= lower_bound) & (group['Monetary_Value'] <= upper_bound)]

# Plots sales forecast
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

# This function preprocesses sales data for RFM analysis.
def preprocess_rfm_data(df):
    # Convert the 'Sales_Date' column to datetime format.
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])

    # Find the current date, which is the maximum date in the 'Sales_Date' column.
    current_date = df['Sales_Date'].max()

    # Group the DataFrame by 'Client_ID' and monthly sales date ('Sales_Date') to calculate RFM metrics.
    # Calculate Recency as the number of days between the last purchase date and the current date.
    # Calculate Frequency as the count of sales within each month.
    # Calculate Monetary as the sum of monetary values of sales within each month.
    rfm = df.groupby(['Client_ID', pd.Grouper(key='Sales_Date', freq='M')]).agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Recency
        'Sales_Date': 'count',                                  # Frequency
        'Monetary_Value': 'sum'                                 # Monetary
    }).rename(columns={'Sales_Date': 'Recency', 'Monetary_Value': 'Monetary'}).reset_index()

    # Return the preprocessed RFM DataFrame.
    return rfm

# This function clusters clients based on the angular coefficient of their RFM scores.
def cluster_clients(rfm):
    angular_coefficients = []  # Initialize a list to store client IDs and their corresponding angular coefficients.

    # Iterate over each client in the RFM DataFrame.
    for client_id, group in rfm.groupby('Client_ID'):
        # Skip clients with insufficient data (less than 2 data points).
        if len(group) < 2:
            continue

        # Prepare the data for linear regression: reshape index values and extract 'Recency' values.
        X = group.index.values.reshape(-1, 1)
        y = group['Recency'].values

        # Fit linear regression to the data.
        lr = LinearRegression()
        lr.fit(X, y)

        # Extract the angular coefficient (slope) of the linear regression line.
        angular_coefficient = lr.coef_[0]

        # Append the client ID and angular coefficient to the list.
        angular_coefficients.append([client_id, angular_coefficient])

    # Create a DataFrame from the list of client IDs and angular coefficients.
    angular_coefficients_df = pd.DataFrame(angular_coefficients, columns=['Client_ID', 'Angular_Coefficient'])

    # Initialize KMeans with 3 clusters and fit it to the angular coefficients.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(angular_coefficients_df[['Angular_Coefficient']])

    # Assign cluster labels to clients based on the KMeans clustering.
    angular_coefficients_df['Cluster'] = kmeans.labels_

    # Map cluster labels to meaningful cluster names.
    cluster_labels = {
        0: "Loyal customer",
        1: "Stable customer",
        2: "Customer at risk"
    }
    angular_coefficients_df['Cluster_Label'] = angular_coefficients_df['Cluster'].map(cluster_labels)

    # Return the DataFrame containing client IDs, angular coefficients, cluster labels, and cluster names.
    return angular_coefficients_df

# This function forecasts RFM trends for each client using linear regression.
def forecast_rfm_trends(rfm, angular_coefficients_df):
    # Iterate over each client in the RFM DataFrame.
    for client_id, group in rfm.groupby('Client_ID'):
        # Skip clients with insufficient data (less than 2 data points).
        if len(group) < 2:
            continue

        # Prepare the data for linear regression: reshape index values and extract 'Recency' values.
        X = group.index.values.reshape(-1, 1)
        y = group['Recency'].values

        # Fit linear regression to the data.
        lr = LinearRegression()
        lr.fit(X, y)

        # Make predictions using the linear regression model.
        y_pred = lr.predict(X)

        # Plot the historical data and the forecasted trend.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(group.index, group['Recency'], label='Historical Data')
        ax.plot(group.index, y_pred, label='Forecast', linestyle='--')

        # Set plot title, labels, and legend.
        ax.set_title(f'RFM Score Forecast for Client ID: {client_id}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Recency')
        ax.legend()

        # Annotate the plot with the client's angular coefficient.
        ax.annotate(f'Angular Coefficient: {angular_coefficients_df[angular_coefficients_df["Client_ID"] == client_id]["Angular_Coefficient"].values[0]}',
                    xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        # Display the plot.
        st.pyplot(fig)  # (Assuming 'st' is a Streamlit function for displaying plots)

# Plots the results of clustering clients based on their angular coefficients.
def plot_cluster_results(angular_coefficients_df):
    # Create a figure and set its size.
    plt.figure(figsize=(10, 6))

    # Iterate over unique cluster labels and corresponding colors.
    for label, color in zip(angular_coefficients_df['Cluster_Label'].unique(), ['blue', 'green', 'red']):
        # Filter the DataFrame to get clients belonging to the current cluster label.
        subset = angular_coefficients_df[angular_coefficients_df['Cluster_Label'] == label]

        # Scatter plot of client IDs against their angular coefficients, using the specified color.
        plt.scatter(subset['Client_ID'], subset['Angular_Coefficient'], label=label, color=color)

    # Set plot title, x-label, and y-label.
    plt.title('Clustering of Clients based on Angular Coefficients')
    plt.xlabel('Client ID')
    plt.ylabel('Angular Coefficient')

    # Add legend to the plot.
    plt.legend()

    # Display the plot.
    st.pyplot(plt)  # (Assuming 'st' is a Streamlit function for displaying plots)

# Prints the clustering results in a table format.
def print_clustering_results_table(angular_coefficients_df):
    # Display a subheader indicating the section title.
    st.subheader("Clustering Results")

    # Display a markdown message describing the table.
    st.markdown("Table of clustering results:")

    # Display a table containing columns 'Client_ID', 'Cluster', and 'Cluster_Label'.
    st.table(angular_coefficients_df[['Client_ID', 'Cluster', 'Cluster_Label']])
