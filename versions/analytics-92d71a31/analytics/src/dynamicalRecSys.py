# dynamicalRecSys.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains functions for our recommender system wherein the user
# is allowed to choose the set of products that will be recommended from a pool
# X, and the set of clients for which recommendations will be shown from a pool
# Y, where: X is a cluster of products according to the tendency, or time-evolu-
# tion, of its sales performance; and Y is a cluster of clients based on the
# time variation of its RFM score. So, here we build a time-dependent, of dyna-
# mical recommender system.

# In sec. 1, we define the functions pertaining to our dynamical RecSys engine.
# In sec. 2, we declare routines to compute RFM scores, adapted to the RecSys
# context, and to print the output using stream-lit web interface.

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from trends import *

# Section 1: Core functions

def dynamicalRecSys_cluster_products(rfm):
    # Initialize an empty list to store the angular coefficients for each product
    angular_coefficients = []

    # Group the RFM data by 'Product_ID'
    for product_id, group in rfm.groupby('Product_ID'):
        # Skip the product if it has fewer than 2 records
        if len(group) < 2:
            continue

        # Prepare the input features (X) and target variable (y) for linear regression
        X = group.index.values.reshape(-1, 1)
        y = group['Recency'].values

        # Create and fit a linear regression model
        lr = LinearRegression()
        lr.fit(X, y)

        # Get the angular coefficient (slope) from the linear regression model
        angular_coefficient = lr.coef_[0]

        # Append the product ID and its angular coefficient to the list
        angular_coefficients.append([product_id, angular_coefficient])

    # Convert the list of angular coefficients to a DataFrame
    angular_coefficients_df = pd.DataFrame(angular_coefficients, columns=['Product_ID', 'Angular_Coefficient'])

    # Create and fit a KMeans clustering model to the angular coefficients
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(angular_coefficients_df[['Angular_Coefficient']])

    # Add the cluster labels to the DataFrame
    angular_coefficients_df['Cluster'] = kmeans.labels_

    # Map the cluster labels to meaningful names
    cluster_labels = {0: "High performance", 1: "Stable performance", 2: "Low performance"}
    angular_coefficients_df['Cluster_Label'] = angular_coefficients_df['Cluster'].map(cluster_labels)

    # Return the DataFrame with product IDs, angular coefficients, and cluster labels
    return angular_coefficients_df

def dynamicalRecSys_recommend_products_to_clients(df, client_clusters_df, product_clusters_df, client_cluster_choice, product_cluster_choice):
    # Select the clients that belong to the chosen client cluster
    selected_clients = client_clusters_df[client_clusters_df['Cluster_Label'] == client_cluster_choice]['Client_ID']

    # Select the products that belong to the chosen product cluster
    selected_products = product_clusters_df[product_clusters_df['Cluster_Label'] == product_cluster_choice]['Product_ID']

    # Initialize an empty list to store the recommendations
    recommendations = []

    # Iterate over each selected client
    for client_id in selected_clients:
        # Get the purchases made by the current client
        client_purchases = df[df['Client_ID'] == client_id]

        # Identify products from the selected products that the client has not purchased
        recommended_products = selected_products[~selected_products.isin(client_purchases['Product_ID'])]

        # Append the client ID and the recommended products to the recommendations list
        recommendations.append((client_id, recommended_products.values))

    # Convert the recommendations list to a DataFrame
    recommendation_df = pd.DataFrame(recommendations, columns=['Client_ID', 'Recommended_Products'])

    # Display the recommendations
    st.write("Recommendations:")
    st.write(tabulate(recommendation_df, headers='keys', tablefmt='pipe'))

# Section 2: Auxiliary functions to prepare the input (i.e., RFM scores) and
# to print the output into the web interface.

def dynamicalRecSys_calculate_product_rfm_scores(df):
    # Convert the 'Sales_Date' column to datetime format
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])

    # Determine the most recent sales date in the dataset
    current_date = df['Sales_Date'].max()

    # Group the data by 'Product_ID' and by month ('Sales_Date' with frequency 'M')
    rfm = df.groupby(['Product_ID', pd.Grouper(key='Sales_Date', freq='M')]).agg({
        # Calculate 'Recency' as the number of days since the last sales date for each product in each month
        'Sales_Date': lambda x: (current_date - x.max()).days,

        # Calculate 'Frequency' as the count of 'Client_ID' (number of sales transactions) for each product in each month
        'Client_ID': 'count',

        # Calculate 'Monetary' as the sum of 'Monetary_Value' for each product in each month
        'Monetary_Value': 'sum'
    }).rename(columns={'Sales_Date': 'Recency', 'Client_ID': 'Frequency', 'Monetary_Value': 'Monetary'}).reset_index()

    # Return the RFM scores DataFrame
    return rfm

def dynamicalRecSys_plot_cluster_results(cluster_df, id_column, title):
    # Create a new figure with specified size
    plt.figure(figsize=(10, 6))

    # Iterate over unique cluster labels and corresponding colors
    for label, color in zip(cluster_df['Cluster_Label'].unique(), ['blue', 'green', 'red']):
        # Select subset of data for the current cluster label
        subset = cluster_df[cluster_df['Cluster_Label'] == label]

        # Scatter plot of 'id_column' vs 'Angular_Coefficient' for the current cluster subset
        plt.scatter(subset[id_column], subset['Angular_Coefficient'], label=label, c=color)

    # Set x-axis label
    plt.xlabel(id_column)

    # Set y-axis label
    plt.ylabel('Angular Coefficient')

    # Set plot title
    plt.title(title)

    # Display legend
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot using Streamlit's pyplot function
    st.pyplot()

def dynamicalRecSys_print_clustering_results_table(cluster_df):
    # Write the clustering results DataFrame as a table using tabulate and Streamlit's write function
    st.write(tabulate(cluster_df, headers='keys', tablefmt='pipe'))
