import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import streamlit as st

# Section 1: Data manipulation

def load_data(file):
    data = pd.read_csv(file)
    return data

def construct_interaction_matrix(data):
    if 'Client_ID' not in data.columns or 'Product_ID' not in data.columns:
        st.error("The uploaded file must contain 'Client_ID' and 'Product_ID' columns.")
        st.stop()

    interaction_matrix = data.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)
    return interaction_matrix

def encode_categorical_columns(contextual_features, columns):
    label_encoders = {}
    for column in columns:
        contextual_features[column] = contextual_features[column].astype('category')
        le = LabelEncoder()
        contextual_features[column] = le.fit_transform(contextual_features[column])
        label_encoders[column] = le
    return contextual_features, label_encoders

def scale_numerical_columns(contextual_features, numerical_columns):
    scaler = StandardScaler()
    contextual_features[numerical_columns] = scaler.fit_transform(contextual_features[numerical_columns])
    return contextual_features, scaler

def prepare_contextual_features(data):
    categorical_columns = ['Sales_Date', 'Product_Name', 'Brand_Name']
    numerical_columns = ['Monetary_Value', 'Discount']

    contextual_features, label_encoders = encode_categorical_columns(data.copy(), categorical_columns)
    contextual_features, scaler = scale_numerical_columns(contextual_features, numerical_columns)

    return contextual_features, label_encoders, scaler

def merge_features(interaction_flattened, contextual_features):
    X = pd.merge(interaction_flattened, contextual_features, left_index=True, right_on=['Client_ID', 'Product_ID'], how='inner')
    y = X['interaction_count']
    X = X.drop(columns=['interaction_count'])
    return X, y

def preprocess_user_data(data, contextual_features, scaler):
    user_context = data.drop(columns=['Client_ID', 'Product_ID'], errors='ignore')
    user_context = pd.get_dummies(user_context, drop_first=True)
    missing_cols = set(contextual_features.columns) - set(user_context.columns)
    for col in missing_cols:
        user_context[col] = 0
    user_context = user_context[contextual_features.columns]
    user_context[['Monetary_Value', 'Discount']] = scaler.transform(user_context[['Monetary_Value', 'Discount']])
    return user_context

# Section 2: Recommender system's engine

def train_model(X_train, y_train, weights):
    weighted_X_train = X_train.copy()
    for column, weight in weights.items():
        if column in weighted_X_train.columns:
            weighted_X_train[column] *= weight
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(weighted_X_train, y_train)
    return model

def get_unique_values(data, columns):
    unique_values = {}
    for column in columns:
        unique_values[column] = data[column].unique()
    return unique_values

def generate_user_products(user_id, product_ids):
    user_products = pd.DataFrame({'Client_ID': [user_id]*len(product_ids), 'Product_ID': product_ids})
    return user_products

def predict_top_products(model, user_contextual_features, product_ids):
    predictions = model.predict_proba(user_contextual_features)[:, 1]
    sorted_indices = np.argsort(predictions)[::-1]
    top_product_ids = product_ids[sorted_indices[:10]]
    return top_product_ids

# Section 3: Generate recommendations and compute performance metric

def generate_recommendations(data, model, contextual_features, scaler, X_train_columns):
    user_ids = data['Client_ID'].unique()
    product_ids = data['Product_ID'].unique()
    recommendations = {}

    for user_id in user_ids:
        user_data = data[data['Client_ID'] == user_id]
        user_context = preprocess_user_data(user_data, contextual_features, scaler)
        user_products = generate_user_products(user_id, product_ids)
        user_contextual_features = pd.concat([user_products.reset_index(drop=True), user_context.reset_index(drop=True)], axis=1)
        user_contextual_features = user_contextual_features[X_train_columns]
        top_products = predict_top_products(model, user_contextual_features, product_ids)
        recommendations[user_id] = top_products

    return recommendations

def recall_at_k(recommendations, true_interactions, k=10):
    total_recall = 0
    for user_id, recommended_products in recommendations.items():
        relevant_items = set(true_interactions[true_interactions['Client_ID'] == user_id]['Product_ID'])
        recommended_top_k = set(recommended_products[:k])
        true_positives = len(relevant_items & recommended_top_k)
        recall = true_positives / min(len(relevant_items), k)
        total_recall += recall
    avg_recall = total_recall / len(recommendations) if len(recommendations) > 0 else 0
    return avg_recall
