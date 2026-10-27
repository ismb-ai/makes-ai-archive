import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from decimal import Decimal, getcontext

# Load data from a CSV file and process dates
def recsys_clients_to_products_load_data(path):
    data = pd.read_csv(path)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    data.dropna(subset=['Sales_Date'], inplace=True)
    data['Recency'] = (datetime.now() - data['Sales_Date']).dt.days
    return data

# Encode Client and Product IDs
def recsys_products_to_clients_encode_ids(data):
    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])
    return data, client_encoder, product_encoder

# Aggregate duplicate entries by summing monetary values
def recsys_products_to_clients_aggregate_duplicates(data, client_col, product_col, rating_col):
    aggregated_data = data.groupby([client_col, product_col], as_index=False)[rating_col].sum()
    return aggregated_data

# Create an interaction matrix (client-product rating matrix)
def recsys_products_to_clients_create_interaction_matrix(data, client_col, product_col, rating_col):
    interaction_matrix = data.pivot(index=client_col, columns=product_col, values=rating_col).fillna(0)
    return interaction_matrix

# Build a matrix factorization model using TensorFlow and Keras
def recsys_products_to_clients_build_matrix_factorization_model(num_clients, num_products, embedding_dim=25, l2_reg=1e-6, dropout_rate=0.5):
    client_input = keras.layers.Input(shape=(1,), name='client_input')
    client_embedding = keras.layers.Embedding(input_dim=num_clients, output_dim=embedding_dim,
                                              embeddings_regularizer=keras.regularizers.l2(l2_reg),
                                              name='client_embedding')(client_input)
    client_vec = keras.layers.Flatten()(client_embedding)

    product_input = keras.layers.Input(shape=(1,), name='product_input')
    product_embedding = keras.layers.Embedding(input_dim=num_products, output_dim=embedding_dim,
                                               embeddings_regularizer=keras.regularizers.l2(l2_reg),
                                               name='product_embedding')(product_input)
    product_vec = keras.layers.Flatten()(product_embedding)

    concatenated = keras.layers.Concatenate()([client_vec, product_vec])
    dropout = keras.layers.Dropout(dropout_rate)(concatenated)
    output = keras.layers.Dense(1)(dropout)

    model = keras.models.Model(inputs=[client_input, product_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the matrix factorization model
def recsys_products_to_clients_train_matrix_factorization_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

# Generate recommendations for a specific client
def recsys_products_to_clients_generate_recommendations(client_encoder, client_id, model, num_products):
    client_encoded_id = client_id
    client_product_combinations = np.array([(client_encoded_id, i) for i in range(num_products)])

    client_ids_encoded = client_product_combinations[:, 0]
    product_ids = client_product_combinations[:, 1]
    predictions = model.predict([client_ids_encoded, product_ids]).flatten()

    scaler = MinMaxScaler(feature_range=(0, 100))
    predictions_normalized = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

    return pd.DataFrame({
        'Client_ID': [client_encoded_id] * len(predictions_normalized),
        'Product_ID': product_ids,
        'Predicted_Rating': predictions_normalized
    })

# Calculate precision at k using decimal for high precision
def precision_at_k(actual, predicted, k):
    actual_set = set(actual[:k])
    predicted_set = set(predicted[:k])
    intersection_count = Decimal(len(actual_set & predicted_set))
    return intersection_count / Decimal(k)

# Calculate recall at k using decimal for high precision
def recall_at_k(actual, predicted, k):
    actual_set = set(actual[:k])
    predicted_set = set(predicted[:k])
    intersection_count = Decimal(len(actual_set & predicted_set))
    return intersection_count / Decimal(len(actual_set))

# Main function to run the recommender system and generate recommendations for all clients
def run_recsys_products_to_clients(data_path, k=15):
    data = recsys_clients_to_products_load_data(data_path)

    data, client_encoder, product_encoder = recsys_products_to_clients_encode_ids(data)

    client_ids = data['Client_ID'].unique()

    data = recsys_products_to_clients_aggregate_duplicates(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
    interaction_matrix = recsys_products_to_clients_create_interaction_matrix(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
    interaction_matrix_values = interaction_matrix.values

    num_clients, num_products = interaction_matrix.shape
    X = np.argwhere(interaction_matrix_values)
    y = interaction_matrix_values[X[:, 0], X[:, 1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mf_model = recsys_products_to_clients_build_matrix_factorization_model(num_clients, num_products)
    recsys_products_to_clients_train_matrix_factorization_model(mf_model, X_train, y_train)

    y_pred_test = mf_model.predict([X_test[:, 0], X_test[:, 1]]).flatten()
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Mean Squared Error (MSE) on the test set: {mse_test}")

    precisions = []
    recalls = []
    evaluation_results = []

    for client_id in client_ids:
        all_predictions = recsys_products_to_clients_generate_recommendations(client_encoder, client_id, mf_model, num_products)
        bought_products = data[data['Client_ID'] == client_id]['Product_ID'].unique()

        all_predictions = all_predictions.sort_values(by='Predicted_Rating', ascending=False)
        top_recommendations = all_predictions.nlargest(k, 'Predicted_Rating')

        # Calculate precision and recall at k
        top_k_products = top_recommendations['Product_ID'].values
        actual_products = bought_products[:k]  # Assuming ground truth contains top k bought products

        precision = precision_at_k(actual_products, top_k_products, k)
        recall = recall_at_k(actual_products, top_k_products, k)
        precisions.append(precision)
        recalls.append(recall)
        evaluation_results.append({
            'Client_ID': client_id,
            'Precision_at_k': precision,
            'Recall_at_k': recall
        })

    # Calculate mean precision and recall using Decimal for high precision
    mean_precision = sum(precisions) / Decimal(len(precisions))
    mean_recall = sum(recalls) / Decimal(len(recalls))

    # Save the results to a text file with high precision
    with open('results/Fusion.txt', 'w') as f:
        f.write(f"Mean Precision at {k}: {mean_precision}\n")
        f.write(f"Mean Recall at {k}: {mean_recall}\n\n")
        f.write("Precision and Recall for all clients:\n")
        f.write(pd.DataFrame(evaluation_results).to_string(index=False))

if __name__ == "__main__":
    data_path = "/Users/igormol/Desktop/Analytics_Fusion/sample/CPP.csv"
    run_recsys_products_to_clients(data_path)
