import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error

# File path
data_path = "/Users/igormol/Desktop/analytics2/sample/Sales_Gracia_modified.csv"

# Function to load data from a CSV file and handle missing Brand_Name values
def recsys_products_to_clients_load_data(path):
    data = pd.read_csv(path)
    data['Brand_Name'].fillna('Unknown', inplace=True)
    return data

# Function to encode Client_ID and Product_ID using LabelEncoder
def recsys_products_to_clients_encode_ids(data):
    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])
    return data, client_encoder, product_encoder

# Function to aggregate duplicate entries by summing rating values
def recsys_products_to_clients_aggregate_duplicates(data, client_col, product_col, rating_col):
    aggregated_data = data.groupby([client_col, product_col], as_index=False)[rating_col].sum()
    aggregated_data = aggregated_data.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on=product_col, how='left')
    return aggregated_data

# Function to create an interaction matrix (client-product rating matrix)
def recsys_products_to_clients_create_interaction_matrix(data, client_col, product_col, rating_col):
    # Remove any remaining duplicates
    data = data.drop_duplicates(subset=[client_col, product_col])
    interaction_matrix = data.pivot(index=client_col, columns=product_col, values=rating_col).fillna(0)
    return interaction_matrix

# Function to build a matrix factorization model using TensorFlow and Keras
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

# Function to train the matrix factorization model
def recsys_products_to_clients_train_matrix_factorization_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

# Function to generate product recommendations for a specific client
def recsys_products_to_clients_generate_recommendations(client_encoder, specific_client_id, model, num_products):
    specific_client_encoded_id = client_encoder.transform([specific_client_id])[0]
    client_product_combinations = np.array([(specific_client_encoded_id, product) for product in range(num_products)])

    all_client_ids = client_product_combinations[:, 0]
    all_product_ids = client_product_combinations[:, 1]
    predictions = model.predict([all_client_ids, all_product_ids]).flatten()

    scaler = MinMaxScaler(feature_range=(0, 100))
    predictions_normalized = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

    return pd.DataFrame({
        'Client_ID': [specific_client_id] * len(predictions_normalized),
        'Product_ID': all_product_ids,
        'Predicted_Rating': predictions_normalized
    })

# Function to compute precision and recall
def compute_precision_recall(recommendations, true_purchases):
    true_positives = len(set(recommendations) & set(true_purchases))
    precision = true_positives / len(recommendations) if len(recommendations) > 0 else 0
    recall = true_positives / len(true_purchases) if len(true_purchases) > 0 else 0
    return precision, recall

# Main function to run the recommender system
def run_recsys_products_to_clients(data_path):
    data = recsys_products_to_clients_load_data(data_path)
    data, client_encoder, product_encoder = recsys_products_to_clients_encode_ids(data)
    client_names = data['Client_ID'].unique()
    client_names = client_encoder.inverse_transform(client_names)

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

    recall_list = []
    precision_list = []

    for selected_client in client_names:
        all_predictions = recsys_products_to_clients_generate_recommendations(client_encoder, selected_client, mf_model, num_products)
        all_predictions = all_predictions.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on='Product_ID', how='left')
        all_predictions = all_predictions.sort_values(by='Predicted_Rating', ascending=False)
        top_recommendations = all_predictions.nlargest(15, 'Predicted_Rating')

        true_purchases = data[data['Client_ID'] == client_encoder.transform([selected_client])[0]]['Product_ID'].unique()
        recommended_products = top_recommendations['Product_ID'].tolist()

        precision, recall = compute_precision_recall(recommended_products, true_purchases)
        recall_list.append(recall)
        precision_list.append(precision)

    mean_recall = np.mean(recall_list)
    mean_precision = np.mean(precision_list)

    with open('results/Neural_Collab.txt', 'w') as f:
        f.write("Client, Recall, Precision\n")
        for client, recall, precision in zip(client_names, recall_list, precision_list):
            f.write(f"{client}, {recall:.10f}, {precision:.10f}\n")
        f.write(f"\nMean Recall: {mean_recall:.10f}\n")
        f.write(f"Mean Precision: {mean_precision:.10f}\n")

if __name__ == "__main__":
    run_recsys_products_to_clients(data_path)
