import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

# Load and preprocess data
def recsys_products_load_and_preprocess_data(file_path):
    file_path.seek(0)
    data = pd.read_csv(file_path)
    client_name_map = data[['Client_ID']].drop_duplicates().reset_index(drop=True)
    client_name_map['Client_Name'] = client_name_map['Client_ID']

    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])

    return data, client_name_map, client_encoder, product_encoder

# Split data
def recsys_products_split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

# Convert date to numerical format
def recsys_products_convert_date_to_numerical(data):
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date']).astype(int) / 10**9
    return data

# Train model
def recsys_products_train_model(train, features, target):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(train[features], train[target])
    return model

# Predict and evaluate
def recsys_products_predict_and_evaluate(model, test, features, target):
    test['Predicted_Value'] = model.predict(test[features])
    mse = mean_squared_error(test[target], test['Predicted_Value'])
    return test, mse

# Normalize scores
def recsys_products_normalize_scores(df):
    min_score = df['Predicted_Value'].min()
    max_score = df['Predicted_Value'].max()
    df['Normalized_Score'] = 100 * (df['Predicted_Value'] - min_score) / (max_score - min_score)
    return df

# Apply k-means clustering and label clusters
def recsys_products_apply_kmeans(df, max_clusters=3):
    n_clusters = min(max_clusters, len(df))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df = df.sort_values(by='Normalized_Score', ascending=False).reset_index(drop=True)
        df['Cluster'] = kmeans.fit_predict(df[['Normalized_Score']])
        labels = {cluster: chr(65 + i) for i, cluster in enumerate(np.argsort(kmeans.cluster_centers_.flatten())[::-1])}
        df['Cluster_Label'] = df['Cluster'].map(labels)
    else:
        df['Cluster_Label'] = 'A'  # Only one cluster if we have one sample
    return df

# Generate recommendations
def recsys_products_generate_recommendations(test, top_k):
    recommendations = defaultdict(list)
    clients = test['Client_ID'].unique()
    for client in clients:
        client_data = test[test['Client_ID'] == client]
        client_data = recsys_products_apply_kmeans(client_data)
        client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False)
        unique_recommendations = client_recommendations.drop_duplicates(subset='Product_ID').head(top_k)
        recommendations[client] = unique_recommendations
    return recommendations

# Calculate recall scores
def recsys_products_calculate_recall_scores(test, recommendations, top_k):
    recall_scores = []
    clients = test['Client_ID'].unique()
    for client in clients:
        actual_products = test[test['Client_ID'] == client]['Product_ID']
        recall = recsys_products_recall_at_k(actual_products, recommendations[client]['Product_ID'], top_k)
        recall_scores.append(recall)
    return np.mean(recall_scores)

# Compute recall at k
def recsys_products_recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(len(actual_set))

# Map product IDs to names
def recsys_products_map_product_ids_to_names(data):
    return dict(zip(data['Product_ID'], data['Product_Name']))

# Decode client IDs to names
def recsys_products_decode_client_ids_to_names(client_name_map, client_encoder):
    client_name_map['Client_ID'] = client_encoder.transform(client_name_map['Client_Name'])
    return client_name_map

# Main functionality
def recsys_products_main_functionality(data, client_name_map, client_encoder, product_encoder):
    train, test = recsys_products_split_data(data)

    train = recsys_products_convert_date_to_numerical(train)
    test = recsys_products_convert_date_to_numerical(test)

    features = ['Sales_Date', 'Client_ID', 'Product_ID']
    target = 'Monetary_Value'

    model = recsys_products_train_model(train, features, target)
    test, mse = recsys_products_predict_and_evaluate(model, test, features, target)
    st.write(f'Mean Squared Error: {mse}')

    test = recsys_products_normalize_scores(test)

    top_k = 10
    recommendations = recsys_products_generate_recommendations(test, top_k)

    average_recall = recsys_products_calculate_recall_scores(test, recommendations, top_k)
    st.write(f'Average Recall at {top_k}: {average_recall}')

    product_map = recsys_products_map_product_ids_to_names(data)
    client_name_map = recsys_products_decode_client_ids_to_names(client_name_map, client_encoder)

    clients = test['Client_ID'].unique()
    client_names = client_name_map['Client_Name'].unique()

    client_name = st.selectbox('Select a Client', ['All Clients'] + list(client_names))

    if client_name == 'All Clients':
        for client in clients:
            client_name = client_name_map[client_name_map['Client_ID'] == client]['Client_Name'].values[0]
            st.subheader(f'Recommendations for {client_name}')
            client_data = recommendations[client]
            recommendations_list = [{"Product Name": product_map[row['Product_ID']], "Cluster Label": row['Cluster_Label']} for _, row in client_data.iterrows()]
            st.table(pd.DataFrame(recommendations_list))
    else:
        client_id = client_name_map[client_name_map['Client_Name'] == client_name]['Client_ID'].values[0]
        st.subheader(f'Recommendations for {client_name}')
        client_data = recommendations[client_id]
        recommendations_list = [{"Product Name": product_map[row['Product_ID']], "Cluster Label": row['Cluster_Label']} for _, row in client_data.iterrows()]
        st.table(pd.DataFrame(recommendations_list))

# Streamlit app
def main():
    st.title('Product Recommender System')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data, client_name_map, client_encoder, product_encoder = recsys_products_load_and_preprocess_data(uploaded_file)
        recsys_products_main_functionality(data, client_name_map, client_encoder, product_encoder)

if __name__ == "__main__":
    main()
