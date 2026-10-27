# recsys.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains the code for the context-aware collaborative filtering
# recommender system using a deep learning approach with Gradient Boosting to
# learn patterns from contextual variables (product's and user's features),
# besides the standard user/item interaction matrix.

# In sec. 1, we define functions responsible for manipulating data. In sec. 2,
# we declare the core routines of the recommender system's neural netwrk. In
# sec. 3, the functions that generates recommendations from the trained neural
# network and computes the perforance metric (recall at k) are defined.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import streamlit as st

# Section 1: Data manipulation

def load_data(file):
    # Loads data from a CSV file into a DataFrame.
    # It takes the following parameter:
    # - 'file': The path to the CSV file.
    # It returns the loaded data as a DataFrame.

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file)

    # Return the loaded data
    return data

def construct_interaction_matrix(data):
    # Constructs an interaction matrix from the data.
    # It groups the data by 'user_id' and 'product_id' and calculates the size of each group.
    # It takes the following parameter:
    # - 'data': The DataFrame containing user-product interactions.
    # If the DataFrame does not contain 'user_id' or 'product_id' columns, it stops the execution with an error message.
    # It returns the constructed interaction matrix as a DataFrame.

    # Check if 'user_id' and 'product_id' columns are present in the data
    if 'user_id' not in data.columns or 'product_id' not in data.columns:
        # If any of the columns is missing, display an error message and stop execution
        st.error("The uploaded file must contain 'user_id' and 'product_id' columns.")
        st.stop()

    # Group the data by 'user_id' and 'product_id' and calculate the size of each group
    # Unstack the grouped data to create an interaction matrix, filling missing values with 0
    interaction_matrix = data.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)

    # Return the constructed interaction matrix
    return interaction_matrix

def drop_timestamp_column(data):
    # Drops the 'timestamp' column from the DataFrame.
    # It takes the following parameter:
    # - 'data': The DataFrame containing the dataset.
    # It returns the DataFrame with the 'timestamp' column dropped.

    # Drop the 'timestamp' column from the DataFrame
    return data.drop(columns=['timestamp'])

def encode_categorical_columns(contextual_features, columns):
    # Encodes categorical columns in a DataFrame using LabelEncoder.
    # It takes the following parameters:
    # - 'contextual_features': The DataFrame containing the contextual features.
    # - 'columns': A list of column names to encode.
    # It returns the DataFrame with encoded categorical columns and a dictionary of LabelEncoders used.

    # Initialize an empty dictionary to store LabelEncoders for each column
    label_encoders = {}

    # Iterate over the columns to be encoded
    for column in columns:
        # Convert the column to categorical type
        contextual_features[column] = contextual_features[column].astype('category')

        # Initialize a new LabelEncoder
        le = LabelEncoder()

        # Encode the values of the column and replace them with encoded values
        contextual_features[column] = le.fit_transform(contextual_features[column])

        # Store the LabelEncoder for the column in the dictionary
        label_encoders[column] = le

    # Return the DataFrame with encoded categorical columns and the dictionary of LabelEncoders
    return contextual_features, label_encoders

def scale_numerical_columns(contextual_features, numerical_columns):
    # Scales numerical columns in a DataFrame using StandardScaler.
    # It takes the following parameters:
    # - 'contextual_features': The DataFrame containing the contextual features.
    # - 'numerical_columns': A list of column names representing numerical columns to scale.
    # It returns the DataFrame with scaled numerical columns and the fitted StandardScaler.

    # Initialize a StandardScaler object
    scaler = StandardScaler()

    # Scale the numerical columns using the fitted scaler and replace them with scaled values
    contextual_features[numerical_columns] = scaler.fit_transform(contextual_features[numerical_columns])

    # Return the DataFrame with scaled numerical columns and the fitted StandardScaler
    return contextual_features, scaler

def prepare_contextual_features(data):
    # Prepares contextual features for modeling.
    # It performs the following steps:
    # 1. Drops the 'timestamp' column from the data.
    # 2. Encodes categorical columns using LabelEncoder.
    # 3. Scales numerical columns using StandardScaler.
    # It takes the following parameter:
    # - 'data': The DataFrame containing the dataset.
    # It returns the preprocessed contextual features, label encoders, and scaler.

    # Step 1: Drop the 'timestamp' column
    contextual_features = drop_timestamp_column(data)

    # Define categorical and numerical columns
    categorical_columns = ['weather_condition', 'operating_system', 'device_type', 'age_group', 'gender', 'income_level', 'mood', 'activity', 'time_of_day', 'day_of_week', 'season']
    numerical_columns = ['temperature']

    # Step 2: Encode categorical columns
    contextual_features, label_encoders = encode_categorical_columns(contextual_features, categorical_columns)

    # Step 3: Scale numerical columns
    contextual_features, scaler = scale_numerical_columns(contextual_features, numerical_columns)

    # Return preprocessed contextual features, label encoders, and scaler
    return contextual_features, label_encoders, scaler

def merge_features(interaction_flattened, contextual_features):
    # Merges interaction features with contextual features.
    # It takes the following parameters:
    # - 'interaction_flattened': DataFrame containing flattened interaction features.
    # - 'contextual_features': DataFrame containing contextual features.
    # It merges the two DataFrames on 'user_id' and 'product_id' and returns the merged DataFrame
    # along with the target variable y.

    # Merge interaction features with contextual features
    X = pd.merge(interaction_flattened, contextual_features, on=['user_id', 'product_id'], how='inner')

    # Extract target variable y
    y = X['interaction_count']

    # Drop the target variable from the features
    X = X.drop(columns=['interaction_count'])

    # Return merged features X and target variable y
    return X, y

def filter_data(data, filters):
    # Filters data based on specified filter conditions.
    # It takes the following parameters:
    # - 'data': The DataFrame containing the dataset.
    # - 'filters': A dictionary where keys are column names and values are filter values.
    # If a filter value is 'any', no filtering is applied for that column.
    # It returns the filtered DataFrame.

    # Iterate over the filter conditions
    for key, value in filters.items():
        # Check if the filter value is not 'any'
        if value != 'any':
            # Apply the filter condition to the DataFrame
            data = data[data[key] == value]

    # Return the filtered DataFrame
    return data

def preprocess_user_data(data, contextual_features, scaler):
    # Preprocesses user data for model prediction.
    # It performs the following steps:
    # 1. Drops unnecessary columns ('user_id', 'product_id', 'timestamp').
    # 2. Encodes categorical variables using one-hot encoding.
    # 3. Adds missing columns from contextual features and fills them with zeros.
    # 4. Scales numerical variables using a pre-fitted scaler.
    # It takes the following parameters:
    # - 'data': The DataFrame containing user data.
    # - 'contextual_features': The DataFrame containing contextual features used for training.
    # - 'scaler': The scaler object used for scaling numerical variables during training.
    # It returns the preprocessed user data as a DataFrame.

    # Step 1: Drop unnecessary columns
    user_context = data.drop(columns=['user_id', 'product_id', 'timestamp'])

    # Step 2: Encode categorical variables using one-hot encoding
    user_context = pd.get_dummies(user_context, drop_first=True)

    # Step 3: Add missing columns from contextual features and fill them with zeros
    missing_cols = set(contextual_features.columns) - set(user_context.columns)
    for col in missing_cols:
        user_context[col] = 0
    user_context = user_context[contextual_features.columns]

    # Step 4: Scale numerical variables using a pre-fitted scaler
    if 'temperature' in user_context.columns:
        user_context[['temperature']] = scaler.transform(user_context[['temperature']])

    # Return the preprocessed user data
    return user_context

# Section 2: Recomender system's engine

def train_model(X_train, y_train, weights):
    # Trains a classification model.
    # It takes the following parameters:
    # - 'X_train': The features for training.
    # - 'y_train': The target variable for training.
    # - 'weights': A dictionary containing feature weights.
    # It applies the specified weights to the corresponding features,
    # trains an XGBoost classifier, and returns the trained model.

    # Apply feature weights to the training features
    weighted_X_train = X_train.copy()
    for column, weight in weights.items():
        if column in weighted_X_train.columns:
            weighted_X_train[column] *= weight

    # Initialize and train an XGBoost classifier
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(weighted_X_train, y_train)

    # Return the trained model
    return model

def get_unique_values(data, columns):
    # Retrieves unique values for specified columns in a DataFrame.
    # It takes the following parameters:
    # - 'data': The DataFrame containing the dataset.
    # - 'columns': A list of column names for which unique values are to be retrieved.
    # It returns a dictionary where keys are column names and values are arrays of unique values.

    # Initialize an empty dictionary to store unique values
    unique_values = {}

    # Iterate over the specified columns
    for column in columns:
        # Retrieve unique values for each column and store them in the dictionary
        unique_values[column] = data[column].unique()

    # Return the dictionary containing unique values for specified columns
    return unique_values

def generate_user_products(user_id, product_ids):
    # Generates user-product pairs for prediction.
    # It creates a DataFrame with 'user_id' and 'product_id' columns.
    # It takes the following parameters:
    # - 'user_id': The ID of the user.
    # - 'product_ids': A list of product IDs to generate pairs for the user.
    # It returns a DataFrame containing user-product pairs.

    # Create a DataFrame with 'user_id' and 'product_id' columns
    user_products = pd.DataFrame({'user_id': [user_id]*len(product_ids), 'product_id': product_ids})

    # Return the DataFrame containing user-product pairs
    return user_products

def predict_top_products(model, user_contextual_features, product_ids):
    # Predicts the top products for a user based on contextual features.
    # It takes the following parameters:
    # - 'model': The trained prediction model.
    # - 'user_contextual_features': The contextual features of the user.
    # - 'product_ids': An array of product IDs for which predictions are to be made.
    # It returns an array of top product IDs predicted for the user.

    # Predict product probabilities for the user
    predictions = model.predict(user_contextual_features)

    # Sort the product indices based on predicted probabilities in descending order
    sorted_indices = np.argsort(predictions)[::-1]

    # Select the top 10 product indices if available
    sorted_indices = sorted_indices[sorted_indices < len(product_ids)]

    # Return the top 10 product IDs predicted for the user
    return product_ids[sorted_indices[:10]]

# Section 3: Generate recommendations and compute performance metric

def generate_recommendations(data, model, contextual_features, scaler, X_train_columns):
    # Generates recommendations for users based on a trained model and contextual features.
    # It takes the following parameters:
    # - 'data': The DataFrame containing user-product interaction data.
    # - 'model': The trained prediction model.
    # - 'contextual_features': The DataFrame containing contextual features used for training.
    # - 'scaler': The scaler object used for scaling numerical variables during training.
    # - 'X_train_columns': The list of column names used in the training data.
    # It returns a dictionary where keys are user IDs and values are arrays of recommended product IDs.

    # Extract unique user IDs and product IDs from the data
    user_ids = data['user_id'].unique()
    product_ids = data['product_id'].unique()

    # Initialize a dictionary to store recommendations for each user
    recommendations = {}

    # Iterate over each user ID
    for user_id in user_ids:
        # Filter data for the current user
        user_data = data[data['user_id'] == user_id]

        # Preprocess user data to get contextual features
        user_context = preprocess_user_data(user_data, contextual_features, scaler)

        # Generate user-product pairs for prediction
        user_products = generate_user_products(user_id, product_ids)

        # Drop unnecessary columns from user contextual features
        user_context = user_context.drop(columns=['user_id', 'product_id'], errors='ignore')

        # Concatenate user-product pairs with user contextual features
        user_contextual_features = pd.concat([user_products.reset_index(drop=True), user_context.reset_index(drop=True)], axis=1)

        # Select only the columns used in the training data
        user_contextual_features = user_contextual_features[X_train_columns]

        # Predict top products for the user
        top_products = predict_top_products(model, user_contextual_features, product_ids)

        # Store recommended products for the user
        recommendations[user_id] = top_products

    # Return the dictionary of recommendations
    return recommendations

def recall_at_k(recommendations, true_interactions, k=10):
    # Calculates the recall@k metric for a set of recommendations.
    # It takes the following parameters:
    # - 'recommendations': A dictionary where keys are user IDs and values are arrays of recommended product IDs.
    # - 'true_interactions': The DataFrame containing true user-product interactions.
    # - 'k': The number of top recommendations to consider for recall calculation (default is 10).
    # It returns the average recall@k across all users.

    # Initialize a variable to store total recall
    total_recall = 0

    # Iterate over each user and their recommended products
    for user_id, recommended_products in recommendations.items():
        # Get the set of relevant items for the current user from true interactions
        relevant_items = set(true_interactions[true_interactions['user_id'] == user_id]['product_id'])

        # Get the top k recommended products for the current user
        recommended_top_k = set(recommended_products[:k])

        # Calculate the number of true positives (intersection between relevant items and top k recommendations)
        true_positives = len(relevant_items & recommended_top_k)

        # Calculate recall@k for the current user
        recall = true_positives / min(len(relevant_items), k)

        # Add the recall@k for the current user to the total recall
        total_recall += recall

    # Calculate the average recall@k across all users
    avg_recall = total_recall / len(recommendations) if len(recommendations) > 0 else 0

    # Return the average recall@k
    return avg_recall
