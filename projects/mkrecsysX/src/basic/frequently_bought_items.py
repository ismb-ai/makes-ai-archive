# frequently_bought_items.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This file contains the functions to determine the similarity beteen products.

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

# Function to load dataframe.
def load_data_new(file_path):
    # Read the data from the CSV file located at the specified file_path
    # and return the DataFrame containing the loaded data
    return pd.read_csv(file_path)

# Define the interaction matrix.
def create_interaction_matrix_new(data_frame):
    # Group the DataFrame by 'Client_ID' and 'Product_ID' and count the occurrences of each combination
    # Then unstack the resulting Series into a DataFrame, filling missing values with 0
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

# Create the similarity matrix.
def instantiate_similarity_matrix(interaction_matrix):
    # Calculate the cosine similarity between the transpose of the interaction matrix
    # This results in a similarity matrix where each element represents the cosine similarity between two products
    return cosine_similarity(interaction_matrix.T)

# Determine product's similarity.
def get_similar_products(similarity_matrix, interaction_matrix, product_id, k=5):
    # Get the index of the specified product_id in the interaction_matrix columns
    product_index = interaction_matrix.columns.get_loc(product_id)

    # Sort the indices of products based on their similarity to the specified product_id
    # Select the top k+1 indices since the most similar product will be itself (hence k+1)
    similar_indices = similarity_matrix[product_index].argsort()[:-k-1:-1]

    # Extract the product IDs corresponding to the sorted indices, excluding the specified product_id itself
    similar_products = interaction_matrix.columns[similar_indices].tolist()[1:]

    # Return the list of similar product IDs
    return similar_products

# Function to perform product similary Analysis
def recommend_products_for_product(similarity_matrix, interaction_matrix, product_id, k=5):
    # Get the list of similar products to the specified product_id based on the similarity_matrix
    similar_products = get_similar_products(similarity_matrix, interaction_matrix, product_id, k)

    # Get the list of relevant items (products purchased) for the specified product_id
    relevant_items = interaction_matrix[interaction_matrix[product_id] > 0].index.tolist()

    # Count the number of recommended similar products that are also relevant (purchased)
    recommended_relevant = sum(1 for item in similar_products if item in relevant_items)

    # Calculate the recall@k metric, which is the total number of relevant items
    recall_at_k = len(relevant_items)

    # Return the list of similar products and the recall@k value
    return similar_products, recall_at_k
