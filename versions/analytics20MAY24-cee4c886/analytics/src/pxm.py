# pxm.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the functions pertaining to our system that
# detect products with anomalous RFM scores using Isolation Forests, and then
# generates clients recommendations for these products using an Autoenconder.

# In sec. 1, we define functions that helps to prepare user's data, builds the
# Isolation Forests model and constructs the Autoenconder achitecture. In sec. 2
# declares the core functions to train the models and extract its predictions.
# Finally, in sec. 3, we define the routines that prints the results into the
# web interface.

# Libraries for Section 1: Functions to handle data and build the architecture
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations
from sklearn.ensemble import IsolationForest  # Anomaly detection
from sklearn.model_selection import train_test_split  # Data splitting

# Libraries for Section 2: Core functions
from tensorflow.keras.models import Sequential  # Building neural network models
from tensorflow.keras.layers import Dense, Input  # Defining layers for neural network
from tensorflow.keras import Model  # Model class for building models

# Libraries for Section 3: User interface
import streamlit as st

# Internal libraries
from recsys import *
from rfm import *

# Section 1: Auxiliary functions to handle data and to build the architecture

def pXm_detect_anomalies(product_rfm):
    # Detect anomalies in the RFM data using Isolation Forest algorithm
    rfm_features = product_rfm[['Recency', 'Frequency', 'Monetary']]
    iso_forest = IsolationForest(contamination=0.1)  # Adjust the contamination parameter as needed
    iso_forest.fit(rfm_features)

    # Compute anomaly score and predict anomalies
    product_rfm['Anomaly_Score'] = iso_forest.decision_function(rfm_features)
    product_rfm['Anomaly'] = iso_forest.predict(rfm_features)

    # Normalize anomaly scores between 0 and 1
    min_anomaly_score = product_rfm['Anomaly_Score'].min()
    max_anomaly_score = product_rfm['Anomaly_Score'].max()
    product_rfm['Anomaly_Score_Normalized'] = (product_rfm['Anomaly_Score'] - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # Filter and return anomalous products
    anomalous_products = product_rfm[product_rfm['Anomaly'] == -1]
    return anomalous_products

# Section 2: Core functions

def pXm_train_and_get_embeddings(interaction_matrix):
    # Extract input data (interaction matrix values) from the interaction matrix.
    X = interaction_matrix.values

    # Split the input data into training and testing sets using train_test_split function.
    # test_size specifies the proportion of the dataset to include in the test split.
    # random_state sets the random seed for reproducibility.
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Build an autoencoder model based on the number of features in the input data using pXm_build_autoencoder_model function.
    autoencoder_model = build_autoencoder_model(X_train.shape[1])

    # Train the autoencoder model using pXm_train_autoencoder function with the training data.
    # Use a predefined number of epochs and batch size for training.
    train_autoencoder(autoencoder_model, X_train, X_test, epochs=50, batch_size=32)

    # Extract the encoder part of the trained autoencoder model.
    encoder_model = Sequential(autoencoder_model.layers[:3])

    # Get embeddings (latent representations) using the encoder part of the autoencoder model
    # with the entire input data (both training and testing sets).
    embeddings = get_embeddings(encoder_model, X)

    # Return the embeddings obtained from the encoder model.
    return embeddings

def pXm_generate_recommendations(anomalous_products, embeddings, interaction_matrix):
    # Initialize an empty dictionary to store recommendation results for each anomalous product.
    recommendation_results = {}

    # Iterate over each anomalous product in the list of anomalous products.
    for product_id in anomalous_products['Product_ID']:
        # Get recommendations for the current anomalous product using pXm_recommend_clients_to_products function.
        # Use the embeddings, interaction matrix, and specify the number of recommendations as 'k=5'.
        recommended_clients = recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)

        # Store the recommended clients for the current product in the recommendation_results dictionary.
        recommendation_results[product_id] = recommended_clients

    # Return the dictionary containing recommendation results for all anomalous products.
    return recommendation_results

def pXm_detect_anomalous_products(df):
    # Compute RFM scores for products using pXm_compute_product_rfm function.
    product_rfm = compute_product_rfm(df)

    # Detect anomalous products based on computed RFM scores using pXm_detect_anomalies function.
    anomalous_products = pXm_detect_anomalies(product_rfm)

    # Return the list of anomalous products.
    return anomalous_products

def pXm_recommend_clients_for_anomalous_products(df, anomalous_products, interaction_matrix):
    # Display a subheader indicating the purpose of the analysis.
    st.subheader("Product Experience Analysis")

    # Provide a brief description of the analysis using Markdown.
    st.markdown("This analysis utilizes Isolation Forests to detect anomalous products based on their RFM scores. It then generates client recommendations for these products using an Autoencoder neural network. The Product Experience Analysis feature enables users to examine the selling behavior of products with anomalous RFM scores.")

    # Check if interaction_matrix is provided, if not, create it from the DataFrame.
    if interaction_matrix is None:
        interaction_matrix = create_interaction_matrix(df)

    # Train an autoencoder model and obtain embeddings from interaction_matrix.
    embeddings = pXm_train_and_get_embeddings(interaction_matrix)

    # Generate recommendations for anomalous products based on embeddings and interaction_matrix.
    recommendation_results = pXm_generate_recommendations(anomalous_products, embeddings, interaction_matrix)

    # Print details about anomalous products.
    pXm_print_anomalous_products(anomalous_products)

    # Print recommendations for anomalous products.
    pXm_print_recommendations(recommendation_results)

# Section 3: User interface

def pXm_print_anomalous_products(anomalous_products):
    # Print anomalous products information using Streamlit subheader and table components.
    # Display columns 'Product_ID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', and 'Anomaly_Score_Normalized'.
    st.subheader("Anomalous Products")
    st.table(anomalous_products[['Product_ID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Anomaly_Score_Normalized']])

def pXm_print_recommendations(recommendation_results):
    # Print recommendations for each product using Streamlit subheader and table components.
    # Iterate over recommendation results dictionary and display recommended clients for each product.
    for product_id, clients in recommendation_results.items():
        st.subheader(f"Recommended Clients for Product {product_id}")
        st.table(clients)
