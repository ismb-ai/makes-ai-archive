# bayes.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the routines for the system that employs a
# Bayesian Decision Tree, based on the RFM scores and "Churn Scores" of clients,
# in order to determine which customers requires attention. Then, we use an
# Autoencoder to recommend products to these customers.

# In sec. 1, we write the routines that handles user's data and the Bayesian's
# tree architecture. In sec. 2, we declare the core functions of our decision
# algorithm. Finally, in sec. 3, we define the functions that prints the output.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K

# Section 1: Functions to handle user's data and to build the architecture

def bayes_load_data(filepath):
    # Load data from CSV file into a DataFrame
    df = pd.read_csv(filepath)
    # Convert 'Sales_Date' column to datetime format
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    # Return the DataFrame
    return df

def bayes_compute_rfm(df):
    # Find the most recent sales date
    current_date = pd.to_datetime(df['Sales_Date']).max()
    # Group by 'Client_ID' and aggregate RFM metrics
    rfm_data = df.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - pd.to_datetime(x).max()).days,  # Recency
        'Product_ID': 'count',  # Frequency
        'Monetary_Value': 'sum'  # Monetary
    }).reset_index()

    # Calculate RFM scores and normalize them
    rfm_data['Recency'] = pd.qcut(rfm_data['Sales_Date'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Monetary'] = pd.qcut(rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']
    rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / \
                                       (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
    # Return the RFM data
    return rfm_data

def bayes_preprocess_churn_data(df):
    # Remove rows with missing values
    df = df.dropna()
    # Encode categorical features
    le = LabelEncoder()
    df['Product_Name'] = le.fit_transform(df['Product_Name'])
    df['Brand_Name'] = le.fit_transform(df['Brand_Name'])
    # Calculate total monetary value and total discount per client
    df['Total_Monetary_Value'] = df.groupby('Client_ID')['Monetary_Value'].transform('sum')
    df['Total_Discount'] = df.groupby('Client_ID')['Discount'].transform('sum')
    # Count unique products per client
    df['Unique_Products'] = df.groupby('Client_ID')['Product_ID'].transform('nunique')
    # Calculate days since last purchase per client
    df['Days_Since_Last_Purchase'] = (df['Sales_Date'].max() - df.groupby('Client_ID')['Sales_Date'].transform('max')).dt.days
    # Label churn based on days since last purchase
    df['Churn'] = (df['Days_Since_Last_Purchase'] > 30).astype(int)
    # Drop duplicate rows based on 'Client_ID'
    df = df.drop_duplicates(subset=['Client_ID'])
    # Return the preprocessed DataFrame
    return df

# Section 2: Core functions of the Bayesian Decision Tree

def bayes_check_class_imbalance(y):
    # Calculate class counts
    class_counts = y.value_counts()
    # Check if the ratio of minority to majority class is greater than 2 or vice versa
    return class_counts[0] / class_counts[1] > 2 or class_counts[1] / class_counts[0] > 2

def bayes_handle_class_imbalance(df, target):
    # Calculate class counts
    class_counts = df[target].value_counts()
    # Separate majority and minority class samples
    df_majority = df[df[target] == class_counts.idxmax()]
    df_minority = df[df[target] == class_counts.idxmin()]
    # Upsample the minority class to match the majority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)
    # Concatenate the upsampled minority class with the majority class
    return pd.concat([df_majority, df_minority_upsampled])

def bayes_train_rf_model(df, features, target):
    # Extract features and target variable from the DataFrame
    X, y = df[features], df[target]
    # Check for class imbalance and handle if necessary
    if bayes_check_class_imbalance(y):
        df = bayes_handle_class_imbalance(df, target)
        X, y = df[features], df[target]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train a RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    # Evaluate the model
    bayes_evaluate_model(rf, X_test, y_test)
    # Return the trained RandomForestClassifier
    return rf

def bayes_evaluate_model(model, X_test, y_test):
    # Predict classes and probabilities using the trained model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # Display model evaluation metrics
    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
    st.write("\nClassification Report:\n", classification_report(y_test, y_pred))
    # Plot confusion matrix
    bayes_plot_confusion_matrix(y_test, y_pred)
    # Plot ROC curve
    bayes_plot_roc_curve(y_test, y_prob)

def bayes_train_bayesian_decision_tree(df, features, target):
    # Extract features and target variable from the DataFrame
    X, y = df[features], df[target]
    # Create a pipeline with StandardScaler and Gaussian Naive Bayes
    model = make_pipeline(StandardScaler(), GaussianNB())
    # Fit the model to the data
    model.fit(X, y)
    # Return the trained model
    return model

def bayes_create_interaction_matrix(data_frame):
    # Create interaction matrix from the data frame
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)
    return interaction_matrix

def bayes_recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    # Get the index of the client in the interaction matrix
    client_index = interaction_matrix.index.get_loc(client_id)
    # Extract the embedding of the client
    client_embedding = embeddings[client_index]
    # Compute cosine similarities between client's embedding and all other embeddings
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))
    # Filter out invalid indices (e.g., indices beyond the number of products)
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    # Sort the indices based on similarities and select the top k
    product_indices = similarities.argsort()[::-1][valid_indices][:k]
    # Retrieve the client's purchases
    client_purchases = interaction_matrix.loc[client_id]
    # Extract IDs of products already purchased by the client
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    # Filter recommended products to exclude those already purchased by the client
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]
    return recommended_products

def bayes_build_autoencoder_model(input_size):
    # Define a Sequential model
    model = Sequential([
        # Input layer with ReLU activation
        Dense(128, activation='relu', input_shape=(input_size,)),
        # Dropout layer for regularization
        Dropout(0.5),
        # Hidden layer with ReLU activation
        Dense(64, activation='relu'),
        # Hidden layer with ReLU activation
        Dense(128, activation='relu'),
        # Dropout layer for regularization
        Dropout(0.5),
        # Output layer with sigmoid activation
        Dense(input_size, activation='sigmoid')
    ])
    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def bayes_train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    # Train the autoencoder model
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1)

def bayes_get_embeddings(autoencoder_model, interaction_matrix):
    # Define a Keras function to get embeddings from the autoencoder model
    get_embedding_layer = K.function([autoencoder_model.layers[0].input], [autoencoder_model.layers[2].output])
    # Get embeddings by passing interaction matrix through the defined function
    embeddings = get_embedding_layer([interaction_matrix.to_numpy()])[0]
    return embeddings

# Section 3: Routines to print the results into the web application

def bayes_plot_confusion_matrix(y_test, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix using seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Display the plot in Streamlit
    st.pyplot(plt)

def bayes_plot_roc_curve(y_test, y_prob):
    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # Display the plot in Streamlit
    st.pyplot(plt)

def bayes_display_results_table(combined_df):
    # Display results table in Streamlit using Markdown format
    st.write(combined_df[['Client_ID', 'RFM_Score_Normalized', 'Churn_Risk', 'Attention_Needed']].to_markdown())
