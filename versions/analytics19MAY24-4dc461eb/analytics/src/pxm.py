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

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import streamlit as st

# Section 1: Auxiliary functions to handle data and to build the architecture

def pXm_compute_product_rfm(df):
    # Compute RFM (Recency, Frequency, Monetary) scores for products based on the given DataFrame
    product_rfm_data = df.groupby('Product_ID').agg({
        'Sales_Date': lambda x: len(x.unique()),  # Recency (count of unique sales dates)
        'Client_ID': 'count',  # Frequency (count of clients)
        'Monetary_Value': 'sum'  # Monetary (sum of monetary value)
    }).reset_index()

    # Rename columns for clarity
    product_rfm_data.rename(columns={'Sales_Date': 'Recency', 'Client_ID': 'Frequency'}, inplace=True)

    # Convert numerical features into quantiles (10 bins) and assign labels
    product_rfm_data['Recency'] = pd.qcut(product_rfm_data['Recency'], q=10, labels=False, duplicates='drop') + 1
    product_rfm_data['Frequency'] = pd.qcut(product_rfm_data['Frequency'], q=10, labels=False, duplicates='drop') + 1
    product_rfm_data['Monetary'] = pd.qcut(product_rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1

    # Calculate RFM score by summing up Recency, Frequency, and Monetary scores
    product_rfm_data['RFM_Score'] = product_rfm_data['Recency'] + product_rfm_data['Frequency'] + product_rfm_data['Monetary']

    # Normalize RFM scores between 0 and 100
    product_rfm_data['RFM_Score_Normalized'] = (product_rfm_data['RFM_Score'] - product_rfm_data['RFM_Score'].min()) / \
                                               (product_rfm_data['RFM_Score'].max() - product_rfm_data['RFM_Score'].min()) * 100
    return product_rfm_data

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

def pXm_create_interaction_matrix(data_frame):
    # Group the DataFrame by 'Client_ID' and 'Product_ID' and calculate the size of each group,
    # representing the frequency of interaction between clients and products.
    # Unstack the grouped data to reshape it into a matrix where rows represent clients,
    # columns represent products, and values represent the frequency of interactions.
    # Fill missing values with 0, indicating no interaction between certain clients and products.
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)
    return interaction_matrix

def pXm_build_autoencoder_model(input_size):
    # Build an autoencoder model using Keras Sequential API.
    # The model consists of several layers:
    # 1. Input layer with 'input_size' neurons, using ReLU activation function.
    # 2. Dropout layer with dropout rate of 0.5, which randomly drops 50% of input units during training to prevent overfitting.
    # 3. Hidden layer with 64 neurons and ReLU activation function.
    # 4. Output layer with 'input_size' neurons and sigmoid activation function, reconstructing the input data.
    # The model is compiled using the Adam optimizer and Mean Squared Error loss function.
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_size,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(input_size, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def pXm_train_autoencoder(autoencoder_model, X_train, X_test, epochs=50, batch_size=32):
    # Train the autoencoder model using the given training and test data.
    # X_train and X_test are used as both input and target since autoencoder is trained to reconstruct the input.
    # Epochs define the number of training iterations, and batch_size specifies the number of samples per gradient update.
    # Shuffle parameter shuffles the training data before each epoch.
    # Validation data (X_test) is used to monitor the model performance during training.
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

def pXm_load_and_process_data(file_path):
    # Load data from the given file path using pXm_load_data function.
    df = pXm_load_data(file_path)

    # Compute RFM (Recency, Frequency, Monetary) scores for products using pXm_compute_product_rfm function.
    product_rfm = pXm_compute_product_rfm(df)

    # Detect anomalous products using pXm_detect_anomalies function based on the computed RFM scores.
    anomalous_products = pXm_detect_anomalies(product_rfm)

    # Create an interaction matrix based on the loaded data using pXm_create_interaction_matrix function.
    interaction_matrix = pXm_create_interaction_matrix(df)

    # Return the loaded DataFrame, product RFM scores, anomalous products, and interaction matrix.
    return df, product_rfm, anomalous_products, interaction_matrix

# Section 2: Core functions

def pXm_train_and_get_embeddings(interaction_matrix):
    # Extract input data (interaction matrix values) from the interaction matrix.
    X = interaction_matrix.values

    # Split the input data into training and testing sets using train_test_split function.
    # test_size specifies the proportion of the dataset to include in the test split.
    # random_state sets the random seed for reproducibility.
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Build an autoencoder model based on the number of features in the input data using pXm_build_autoencoder_model function.
    autoencoder_model = pXm_build_autoencoder_model(X_train.shape[1])

    # Train the autoencoder model using pXm_train_autoencoder function with the training data.
    # Use a predefined number of epochs and batch size for training.
    pXm_train_autoencoder(autoencoder_model, X_train, X_test, epochs=50, batch_size=32)

    # Extract the encoder part of the trained autoencoder model.
    encoder_model = Sequential(autoencoder_model.layers[:3])

    # Get embeddings (latent representations) using the encoder part of the autoencoder model
    # with the entire input data (both training and testing sets).
    embeddings = pXm_get_embeddings(encoder_model, X)

    # Return the embeddings obtained from the encoder model.
    return embeddings

def pXm_recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5):
    # Get the index of the specified product in the interaction matrix.
    product_index = interaction_matrix.columns.get_loc(product_id)

    # Extract the embedding vector for the specified product from the embeddings matrix.
    product_embedding = embeddings[:, product_index]

    # Transpose the embeddings matrix to iterate over clients' embeddings.
    transposed_embeddings = embeddings.T

    # Initialize an empty list to store similarity scores between clients and the specified product.
    recommendations = []

    # Iterate over the transposed embeddings (clients' embeddings).
    for client_embedding in transposed_embeddings:
        # Compute the dot product between the client's embedding and the product's embedding to measure similarity.
        similarity = np.dot(client_embedding, product_embedding)

        # Append the similarity score to the recommendations list.
        recommendations.append(similarity)

    # Convert the recommendations list to a numpy array.
    recommendations = np.array(recommendations)

    # Filter out invalid indices where recommendations exceed the number of products.
    valid_indices = np.where(recommendations.argsort()[::-1] < len(interaction_matrix.columns))[0]

    # Sort the indices based on recommendation scores in descending order, select the top 'k' indices.
    client_indices = recommendations.argsort()[::-1][valid_indices][:k]

    # Get the corresponding client IDs based on the selected indices.
    recommended_clients = interaction_matrix.index[client_indices]

    # Return the recommended client IDs for the specified product.
    return recommended_clients

def pXm_generate_recommendations(anomalous_products, embeddings, interaction_matrix):
    # Initialize an empty dictionary to store recommendation results for each anomalous product.
    recommendation_results = {}

    # Iterate over each anomalous product in the list of anomalous products.
    for product_id in anomalous_products['Product_ID']:
        # Get recommendations for the current anomalous product using pXm_recommend_clients_to_products function.
        # Use the embeddings, interaction matrix, and specify the number of recommendations as 'k=5'.
        recommended_clients = pXm_recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)

        # Store the recommended clients for the current product in the recommendation_results dictionary.
        recommendation_results[product_id] = recommended_clients

    # Return the dictionary containing recommendation results for all anomalous products.
    return recommendation_results

def pXm_detect_anomalous_products(df):
    # Compute RFM scores for products using pXm_compute_product_rfm function.
    product_rfm = pXm_compute_product_rfm(df)

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
        interaction_matrix = pXm_create_interaction_matrix(df)

    # Train an autoencoder model and obtain embeddings from interaction_matrix.
    embeddings = pXm_train_and_get_embeddings(interaction_matrix)

    # Generate recommendations for anomalous products based on embeddings and interaction_matrix.
    recommendation_results = pXm_generate_recommendations(anomalous_products, embeddings, interaction_matrix)

    # Print details about anomalous products.
    pXm_print_anomalous_products(anomalous_products)

    # Print recommendations for anomalous products.
    pXm_print_recommendations(recommendation_results)

def pXm_get_embeddings(encoder_model, X):
    embeddings = encoder_model.predict(X)
    return embeddings

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
