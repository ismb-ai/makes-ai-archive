import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.metrics import recall_score
from tabulate import tabulate

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV',
                           'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]
    return df

def create_interaction_matrix(data_frame):
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def build_autoencoder_model(input_size):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_size, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

def get_embeddings(encoder_model, X):
    return encoder_model.predict(X)

def recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    client_index = interaction_matrix.index.get_loc(client_id)
    client_embedding = embeddings[client_index]
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]
    return recommended_products

def recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5):
    product_index = interaction_matrix.columns.get_loc(product_id)
    product_embedding = embeddings[:, product_index]  # Extracting the product embedding
    
    # Transpose the embeddings matrix so that each row represents a client embedding
    transposed_embeddings = embeddings.T
    
    recommendations = []
    for client_embedding in transposed_embeddings:
        similarity = np.dot(client_embedding, product_embedding)
        recommendations.append(similarity)
    
    recommendations = np.array(recommendations)
    valid_indices = np.where(recommendations.argsort()[::-1] < len(interaction_matrix.columns))[0]
    client_indices = recommendations.argsort()[::-1][valid_indices][:k]
    return interaction_matrix.index[client_indices]


def calculate_recall_at_k(actual_products, recommended_products, k):
    return len(set(recommended_products)) / len(actual_products[actual_products > 0])

def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)
    interaction_matrix = create_interaction_matrix(df)
    train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)
    autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
    train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)
    encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)

    headers = ["Product ID", "Product Name", "Brand"]
    
    # Recommendations for Clients
    print("\nRecommendations for Clients:\n")
    for client_id in interaction_matrix.index:
        recommended_products = recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=15)
        actual_products = interaction_matrix.loc[client_id]
        recall_at_k = calculate_recall_at_k(actual_products, recommended_products, k=5)
        table_data = []
        for product_id in recommended_products:
            product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
            brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
            table_data.append([product_id, product_name, brand])
        print(f"Client {client_id} (Recall at K = 5: {recall_at_k:.2f}):")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    # Recommendations for Products
    print("\nRecommendations for Products:\n")
    for product_id in interaction_matrix.columns:
        recommended_clients = recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)
        print(f"\nProduct {product_id}:")
        if len(recommended_clients) > 0:
            table_data = []
            for client_id in recommended_clients:
                table_data.append([client_id])
            print(tabulate(table_data, headers=["Client ID"], tablefmt="grid"))
        else:
            print("No recommendations for this product.")


if __name__ == "__main__":
    main()

