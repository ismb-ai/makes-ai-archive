# roi.def

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the routines responsible for analyzing the
# Return on Investiment (ROI) and Return on Equity (ROE) metrics. The core
# functions are declared in Section 1, whilst Section 2 defines some Simple
# routines to compute RFM-related metrics.

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

# Section 1: ROI/ROE Analysis

# Calculates the monthly ROI (Return on Investment) and ROE (Return on Equity) metrics for each product.
def roi_calculate_monthly_metrics(df):
    """Calculate ROI and ROE per month for each product ID."""
    # Convert 'Sales_Date' column to datetime format.
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])

    # Calculate total cost after discount for each sale.
    df['Total_Cost'] = df['Monetary_Value'] * (1 - df['Discount'] / 100)

    # Group sales data by month and product ID, then sum the total cost for each group.
    monthly_data = df.groupby([df['Sales_Date'].dt.to_period('M'), 'Product_ID']).agg({'Total_Cost': 'sum'}).reset_index()

    # Rename the 'Sales_Date' column to 'Month'.
    monthly_data = monthly_data.rename(columns={'Sales_Date': 'Month'})

    # Calculate monthly ROI for each product ID (percentage change in total cost).
    monthly_data['ROI'] = monthly_data.groupby('Product_ID')['Total_Cost'].pct_change()

    # Calculate monthly ROE for each product ID (difference in total cost from the previous month).
    monthly_data['ROE'] = monthly_data.groupby('Product_ID')['Total_Cost'].diff()

    # Replace NaN values with 0.
    monthly_data.fillna(0, inplace=True)

    # Return the DataFrame containing monthly metrics.
    return monthly_data

# This function prints a table with ROI and ROE for each product ID.
def roi_print_table(monthly_data):
    """Print table with ROI and ROE for each product ID."""
    # Iterate over each product ID and its corresponding data group.
    for product_id, group in monthly_data.groupby('Product_ID'):
        # Write the product ID as a header.
        st.write(f"Product ID: {int(product_id)}")

        # Display a table containing 'Month', 'ROI', and 'ROE' columns for the current product ID.
        st.table(group[['Month', 'ROI', 'ROE']])

# Plots time series of ROI (Return on Investment) and ROE (Return on Equity) for each product ID.
def roi_plot_metrics(monthly_data):
    """Plot ROI and ROE time series for each product ID."""
    # Iterate over each product ID and its corresponding data group.
    for product_id, group in monthly_data.groupby('Product_ID'):
        # Create a figure with two subplots for ROI and ROE.
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        # Iterate over ROI and ROE metrics.
        for i, metric in enumerate(['ROI', 'ROE']):
            # Prepare data for linear regression.
            X = np.arange(len(group)).reshape(-1, 1)
            y = group[metric].values.reshape(-1, 1)

            # Fit linear regression model.
            model = LinearRegression()
            model.fit(X, y)

            # Make predictions using the linear regression model.
            y_pred = model.predict(X)

            # Plot original time series and regression model.
            axs[i].plot(group['Month'].astype(str), y, label=f'{metric} - Original')
            axs[i].plot(group['Month'].astype(str), y_pred, linestyle='--', label=f'{metric} - Predicted')
            axs[i].set_xlabel('Month')
            axs[i].set_ylabel(metric)
            axs[i].legend()

        # Set title for the entire figure.
        fig.suptitle(f'Product ID {int(product_id)}')

        # Display the plot.
        st.pyplot(fig)  # Assuming 'st' is a Streamlit function for displaying plots

# Section 2: Some functions to deal with the necessary RFM computations

# This function calculates RFM scores for each product based on sales data.
def sales_calculate_rfm_scores(df):
    # Get the current date.
    today = pd.to_datetime('today')

    # Group the DataFrame by 'Product_ID' and aggregate sales metrics:
    # - Recency: Calculate the number of days between the latest sale date of each product and today.
    # - Frequency: Count the number of unique clients (Client_IDs) who purchased each product.
    # - Monetary: Sum the monetary value of sales for each product.
    rfm = df.groupby('Product_ID').agg({
        'Sales_Date': lambda x: (today - x.max()).days,  # Recency
        'Client_ID': 'nunique',                         # Frequency
        'Monetary_Value': 'sum'                         # Monetary
    })

    # Rename the columns to represent the RFM scores.
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Return the DataFrame containing RFM scores for each product.
    return rfm

# Returns the geometric mean of RFM scores for each product.
def sales_calculate_geometric_mean(rfm_scores):
    # Compute the geometric mean of RFM scores for each product.
    # It calculates the product of the three scores and then takes the cube root to find the geometric mean.
    rfm_scores['Geometric_Mean'] = np.power(rfm_scores.prod(axis=1), 1/3)

    # Return the series containing geometric mean scores for each product.
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

# Plots the clustering results of product features.
def sales_plot_clusters(product_features):
    # Define labels for each cluster.
    cluster_labels = {
        0: "High performance",
        1: "Stable performance",
        2: "Low performance",
        3: "Loss"
    }

    # Create a new figure and axis for the plot.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each cluster label and corresponding cluster data.
    for cluster_id, label in cluster_labels.items():
        # Filter the product features DataFrame to get data for the current cluster.
        cluster_data = product_features[product_features['Cluster'] == cluster_id]

        # Scatter plot of product IDs against their mean RFM score for the current cluster.
        ax.scatter(cluster_data.index, cluster_data['Mean_RFM_Score'], label=label)

    # Set labels and title for the plot.
    ax.set_xlabel('Product ID')
    ax.set_ylabel('Mean RFM Score')
    ax.set_title('K-means Clustering of Product Sales')

    # Rotate x-axis labels for better visibility.
    ax.tick_params(axis='x', rotation=90)

    # Add legend to the plot.
    ax.legend()

    # Display the plot.
    st.pyplot(fig)  # Assuming 'st' is a Streamlit function for displaying plots

# Prints a table containing the features of each product.
def sales_print_product_features(product_features):
    # Reset the index of the DataFrame and rename the index column to 'Product ID'.
    table = product_features.reset_index().rename(columns={'index': 'Product ID'})

    # Write the table to the Streamlit app using the 'tabulate' function with Markdown format.
    st.write(tabulate(table, headers='keys', tablefmt='pipe'))

# Remove outliers from a group of sales data using Bollinger bands.
def sales_remove_outliers(group):
    # Calculate the standard deviation and mean of the 'Monetary_Value' column in the group.
    std_dev = group['Monetary_Value'].std()
    mean = group['Monetary_Value'].mean()

    # Calculate the lower and upper bounds of the Bollinger bands.
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    # Filter the group to include only the data points within the Bollinger bands.
    return group[(group['Monetary_Value'] >= lower_bound) & (group['Monetary_Value'] <= upper_bound)]

# This function plots sales forecasts using linear regression for a sample of products.
def sales_plot_forecast(grouped_data, rfm_scores, num_products=10):
    # Get the list of product IDs from the grouped data.
    product_ids = list(grouped_data.groups.keys())

    # Randomly sample a subset of product IDs.
    sampled_ids = np.random.choice(product_ids, size=min(num_products, len(product_ids)), replace=False)

    # Iterate over each sampled product ID.
    for product_id in sampled_ids:
        # Create a new figure and axis for the plot.
        fig, ax = plt.subplots(figsize=(8, 5))

        # Get the group of sales data for the current product ID.
        group = grouped_data.get_group(product_id)

        # Remove outliers using Bollinger bands.
        group = sales_remove_outliers(group)

        # Filter the sales data for the current product ID.
        product_data = group[group['Product_ID'] == product_id]

        # Check if there is any data available for the current product ID.
        if len(product_data) == 0:
            st.write(f"No data available for Product ID: {product_id}")
            continue

        # Prepare the features (X) and target (y) for linear regression.
        X = product_data['Monetary_Value'].values.reshape(-1, 1)
        y = pd.to_datetime(product_data['Sales_Date']).astype(int)  # Convert dates to timestamps

        # Initialize and fit a linear regression model.
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions using the linear regression model.
        forecasted_sales = model.predict(X)

        # Plot actual sales data points and the forecasted trend.
        ax.scatter(X, y, color='blue', alpha=0.5, label='Actual')
        ax.plot(X, forecasted_sales, color='red', alpha=0.5, label='Forecast')

        # Set labels and title for the plot.
        ax.set_xlabel('Monetary Value')
        ax.set_ylabel('Sales Date')
        ax.set_title(f'Linear Regression Forecast for Product ID: {product_id}')
        ax.legend()

        # Display the plot.
        st.pyplot(fig)  # (Assuming 'st' is a Streamlit function for displaying plots)

# Appendix: Simple functions for homekepping

# Reads a CSV file located at the given file path and returns a DataFrame.
def roi_read_data(file_path):
    return pd.read_csv(file_path)

# Reads CSV file into a DataFrame
def sales_read_data(file_path):
    return pd.read_csv(file_path)

# Converts 'Sales_Date' to timestamps
def sales_convert_to_timestamps(df):
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df
