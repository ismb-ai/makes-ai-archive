import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def pXm_load_data(file_path):
    return pd.read_csv(file_path)

def pXm_compute_product_rfm(df):
    product_rfm_data = df.groupby('Product_ID').agg({
        'Sales_Date': lambda x: len(x.unique()),  # Recency (count of unique sales dates)
        'Client_ID': 'count',  # Frequency (count of clients)
        'Monetary_Value': 'sum'  # Monetary (sum of monetary value)
    }).reset_index()

    product_rfm_data.rename(columns={'Sales_Date': 'Recency', 'Client_ID': 'Frequency'}, inplace=True)

    product_rfm_data['Recency'] = pd.qcut(product_rfm_data['Recency'], q=10, labels=False, duplicates='drop') + 1
    product_rfm_data['Frequency'] = pd.qcut(product_rfm_data['Frequency'], q=10, labels=False, duplicates='drop') + 1
    product_rfm_data['Monetary'] = pd.qcut(product_rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1

    product_rfm_data['RFM_Score'] = product_rfm_data['Recency'] + product_rfm_data['Frequency'] + product_rfm_data['Monetary']
    product_rfm_data['RFM_Score_Normalized'] = (product_rfm_data['RFM_Score'] - product_rfm_data['RFM_Score'].min()) / \
                                               (product_rfm_data['RFM_Score'].max() - product_rfm_data['RFM_Score'].min()) * 100
    return product_rfm_data

def pXm_detect_anomalies(product_rfm):
    rfm_features = product_rfm[['Recency', 'Frequency', 'Monetary']]
    iso_forest = IsolationForest(contamination=0.1)  # Adjust the contamination parameter as needed
    iso_forest.fit(rfm_features)

    product_rfm['Anomaly_Score'] = iso_forest.decision_function(rfm_features)
    product_rfm['Anomaly'] = iso_forest.predict(rfm_features)

    # Normalize anomaly scores between 0 and 1
    min_anomaly_score = product_rfm['Anomaly_Score'].min()
    max_anomaly_score = product_rfm['Anomaly_Score'].max()
    product_rfm['Anomaly_Score_Normalized'] = (product_rfm['Anomaly_Score'] - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    anomalous_products = product_rfm[product_rfm['Anomaly'] == -1]
    return anomalous_products

def pXm_create_interaction_matrix(data_frame):
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)
    return interaction_matrix

def pXm_build_autoencoder_model(input_size):
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
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

def pXm_get_embeddings(encoder_model, X):
    embeddings = encoder_model.predict(X)
    return embeddings

def pXm_recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5):
    product_index = interaction_matrix.columns.get_loc(product_id)
    product_embedding = embeddings[:, product_index]
    transposed_embeddings = embeddings.T
    recommendations = []
    for client_embedding in transposed_embeddings:
        similarity = np.dot(client_embedding, product_embedding)
        recommendations.append(similarity)
    recommendations = np.array(recommendations)
    valid_indices = np.where(recommendations.argsort()[::-1] < len(interaction_matrix.columns))[0]
    client_indices = recommendations.argsort()[::-1][valid_indices][:k]
    recommended_clients = interaction_matrix.index[client_indices]
    return recommended_clients

def pXm_load_and_process_data(file_path):
    df = pXm_load_data(file_path)
    product_rfm = pXm_compute_product_rfm(df)
    anomalous_products = pXm_detect_anomalies(product_rfm)
    interaction_matrix = pXm_create_interaction_matrix(df)
    return df, product_rfm, anomalous_products, interaction_matrix

def pXm_train_and_get_embeddings(interaction_matrix):
    X = interaction_matrix.values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    autoencoder_model = pXm_build_autoencoder_model(X_train.shape[1])
    pXm_train_autoencoder(autoencoder_model, X_train, X_test, epochs=50, batch_size=32)
    encoder_model = Sequential(autoencoder_model.layers[:3])  # Use the encoder part of the autoencoder
    embeddings = pXm_get_embeddings(encoder_model, X)
    return embeddings

def pXm_generate_recommendations(anomalous_products, embeddings, interaction_matrix):
    recommendation_results = {}
    for product_id in anomalous_products['Product_ID']:
        recommended_clients = pXm_recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)
        recommendation_results[product_id] = recommended_clients
    return recommendation_results

def pXm_print_anomalous_products(anomalous_products):
    print("Anomalous Products:")
    print(anomalous_products[['Product_ID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Anomaly_Score_Normalized']])

def pXm_print_recommendations(recommendation_results):
    for product_id, clients in recommendation_results.items():
        print(f"\nRecommended Clients for Product {product_id}:")
        print(clients)

def main():
    file_path = '/Users/igormol/Desktop/analytics/sample/CPP.csv'

    # Load and process data
    df, product_rfm, anomalous_products, interaction_matrix = pXm_load_and_process_data(file_path)

    # Print table of anomalous products
    pXm_print_anomalous_products(anomalous_products)

    # Train autoencoder and get embeddings
    embeddings = pXm_train_and_get_embeddings(interaction_matrix)

    # Generate recommendations
    recommendation_results = pXm_generate_recommendations(anomalous_products, embeddings, interaction_matrix)

    # Print recommendations
    pXm_print_recommendations(recommendation_results)

if __name__ == "__main__":
    main()
