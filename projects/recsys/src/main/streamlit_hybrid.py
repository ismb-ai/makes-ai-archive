# Unified Recsys/Forecast Web App
#
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
import streamlit as st
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load and preprocess data for the recommender system
def load_data(file_path):
    # Read data from the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group by 'Product_Name' and assign the first 'Product_ID' within each group
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')
    
    # Define substrings and brands to be excluded from the data
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']
    
    # Filter out rows containing excluded substrings in 'Product_Name'
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    
    # Filter out rows containing excluded brands in 'Brand_Name'
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]
    
    # Return the preprocessed DataFrame
    return df

# Create an interaction matrix from a DataFrame
def create_interaction_matrix(data_frame):   
    # Group by 'Client_ID' and 'Product_ID' and count interactions
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size()
    
    # Unstack the result, filling missing values with 0
    interaction_matrix = interaction_matrix.unstack(fill_value=0)
    
    # Return the resulting interaction matrix
    return interaction_matrix
    
# Build an autoencoder model
def build_autoencoder_model(input_size):
    # Create a Sequential model
    model = Sequential()

    # Add a dense layer with 128 neurons, ReLU activation, and specified input shape
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))

    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Add another dense layer with 64 neurons and ReLU activation
    model.add(Dense(64, activation='relu'))

    # Add another dense layer with 128 neurons and ReLU activation
    model.add(Dense(128, activation='relu'))

    # Add another dropout layer
    model.add(Dropout(0.5))

    # Add a dense output layer with input_size neurons and sigmoid activation
    model.add(Dense(input_size, activation='sigmoid'))

    # Compile the model using Adam optimizer and Mean Squared Error (MSE) loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the compiled autoencoder model
    return model

# Train the autoencoder model
def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    # Use the fit method to train the autoencoder model
    autoencoder_model.fit(
        X_train,                    # Training data (input and target are the same for autoencoder)
        X_train,                    # Target data (autoencoder reconstructs input)
        epochs=epochs,              # Number of training epochs
        batch_size=batch_size,      # Size of batches used for training
        shuffle=True,               # Shuffle the training data for each epoch
        validation_data=(X_test, X_test)  # Validation data (input and target are the same for validation)
    )

# Compute a similarity matrix using k-Nearest Neighbors
def compute_similarity_matrix(interaction_matrix, k=5):
    # Create a kNN model with cosine similarity and brute-force algorithm
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    
    # Fit the kNN model to the values of the interaction matrix
    knn.fit(interaction_matrix.values)
    
    # Compute the similarity matrix using the kNN distances
    similarity_matrix = 1 - knn.kneighbors_graph(mode='distance').toarray()
    
    # Fill the diagonal with zeros to avoid self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    
    # Return the computed similarity matrix
    return similarity_matrix

# Obtain embeddings using an encoder model
def get_embeddings(encoder_model, X):
    embeddings = encoder_model.predict(X)
    return embeddings

# Recommend products for a given client using a hybrid approach
def recommend_products_hybrid(embeddings, user_similarity_matrix, client_id, interaction_matrix, k=5):
    # Obtain the index of the target client in the interaction matrix
    client_index = interaction_matrix.index.get_loc(client_id)
    
    # Extract the embedding for the target client
    client_embedding = embeddings[client_index]

    # Calculate weighted embeddings based on user similarity matrix
    weighted_embeddings = np.dot(user_similarity_matrix[client_index], embeddings)
    
    # Combine the client's embedding with the weighted embeddings
    combined_embeddings = 0.5 * client_embedding + 0.5 * weighted_embeddings

    # Calculate cosine similarities between embeddings
    similarities = np.dot(embeddings, combined_embeddings) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(combined_embeddings))
    
    # Filter out invalid indices and get the indices of top-k similar products
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]

    # Get the client's previous purchases from the interaction matrix
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values

    # Filter out already purchased products and select the top-k recommendations
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]

    # Extract predicted scores for the recommended products
    predicted_scores = similarities[product_indices]

    # Return the recommended products and their predicted scores
    return recommended_products, predicted_scores

# Load and preprocess data for the sales forecast
def load_sales_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess data for training the sales forecast model
def preprocess_sales_data(data):
    # Extract time scale (X) and count (y) from the data
    X = data['TIME_SCALE'].values
    y = data['COUNT'].values

    # Split the data into training and testing sets (80-20 split)
    split_index = int(0.8 * len(X))
    X_train = X[:split_index]
    y_train = y[:split_index]

    # Reshape the training data for LSTM model input
    X_train = X_train.reshape((X_train.shape[0], 1, 1))

    # Return the preprocessed data
    return X_train, y_train, X, y

# Build the LSTM model for sales forecast
def build_sales_model():
    # Create a Sequential model
    model = Sequential()

    # Add the first LSTM layer with 100 units, ReLU activation, and input shape (1, 1)
    model.add(LSTM(units=100, activation='relu', input_shape=(1, 1), return_sequences=True))

    # Add the second LSTM layer with 200 units and ReLU activation
    model.add(LSTM(units=200, activation='relu', return_sequences=True))

    # Add the third LSTM layer with 300 units and ReLU activation
    model.add(LSTM(units=300, activation='relu', return_sequences=True))

    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.2))

    # Add the fourth LSTM layer with 200 units and ReLU activation
    model.add(LSTM(units=200, activation='relu'))

    # Add another dropout layer
    model.add(Dropout(0.2))

    # Add a Dense output layer with 1 unit (regression task)
    model.add(Dense(units=1))

    # Compile the model using Adam optimizer and Mean Squared Error (MSE) loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the compiled LSTM model
    return model

# Train the LSTM model for sales forecast
def train_sales_model(model, X_train, y_train, epochs=50, batch_size=1):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    return history

# Make predictions using the trained sales forecast model
def make_sales_predictions(model, X, smoothing_factor=0.1, randomness_factor=0.1):
    # Reshape input data for compatibility with the model
    X = X.reshape((X.shape[0], 1, 1))

    # Use the trained model to make initial predictions
    predictions = model.predict(X)

    # Initialize list to store smoothed predictions
    smoothed_predictions = [predictions[0]]

    # Apply smoothing to the predictions to reduce noise
    for i in range(1, len(predictions)):
        noise = np.random.normal(scale=randomness_factor)
        smoothed_value = smoothing_factor * (predictions[i] + noise) + (1 - smoothing_factor) * smoothed_predictions[-1]
        smoothed_predictions.append(smoothed_value)

    # Generate future time range for predictions
    future_time_range = np.arange(max(X), max(X) + 13, 1)
    future_X = future_time_range.reshape((len(future_time_range), 1, 1))

    # Make predictions for the future time range using the model
    future_predictions = model.predict(future_X)

    # Initialize list to store smoothed future predictions
    smoothed_future_predictions = [smoothed_predictions[-1]]

    # Apply smoothing to the future predictions
    for i in range(1, len(future_predictions)):
        noise = np.random.normal(scale=randomness_factor)
        smoothed_value = smoothing_factor * (future_predictions[i] + noise) + (1 - smoothing_factor) * smoothed_future_predictions[-1]
        smoothed_future_predictions.append(smoothed_value)

    # Return the smoothed predictions, future time range, and smoothed future predictions
    return np.array(smoothed_predictions), future_time_range, np.array(smoothed_future_predictions)

# Calculate Mean Absolute Error (MAE) for sales forecast
def calculate_sales_mae(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions.flatten())
    st.write(f'Mean Absolute Error (MAE) for Sales Forecast: {mae}')

# Plot results for both recommender system and sales forecast
def plot_results_recommender(data, recommendations_info):
    st.subheader("Product Recommendations for the Client:")

    product_ids = recommendations_info['recommended_products']
    product_names = [data.loc[data['Product_ID'] == product_id, 'Product_Name'].iloc[0] for product_id in product_ids]
    brands = [data.loc[data['Product_ID'] == product_id, 'Brand_Name'].iloc[0] for product_id in product_ids]
    predicted_scores = recommendations_info['predicted_scores']

    min_length = min(len(product_ids), len(product_names), len(brands), len(predicted_scores))
    product_ids = product_ids[:min_length]
    product_names = product_names[:min_length]
    brands = brands[:min_length]
    predicted_scores = predicted_scores[:min_length]

    recommendations_df = pd.DataFrame({
        'Product ID': product_ids,
        'Product Name': product_names,
        'Brand': brands,
        'Predicted Score': predicted_scores
    })

    st.table(recommendations_df)

def plot_results_sales(data, predictions, future_time_range, future_predictions, history):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    axs[0].plot(data['TIME_SCALE'], data['COUNT'], label='Original Data', marker='o', linestyle='-')
    axs[0].plot(data['TIME_SCALE'], predictions, label='Model Prediction', linestyle='dashed', color='orange')
    axs[0].plot(future_time_range, future_predictions, label='Future Prediction (12 months)', linestyle='dashed', color='green')
    axs[0].set_title('Sales Forecast vs Original Data (Extrapolation)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('COUNT')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history.history['loss'], label='Mean Squared Error (MSE)')
    axs[1].set_title('Mean Squared Error (MSE) Over Training Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MSE')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# Main function for the unified web app
def main():
    # Set the title for the Streamlit web app
    st.title("Unified Recommender System and Sales Forecast Web App")

    # Allow the user to upload CSV files for the recommender system and sales forecast
    file_path_recommender = st.file_uploader("Upload CSV file for Recommender System", type=["csv"])
    file_path_sales = st.file_uploader("Upload CSV file for Sales Forecast", type=["csv"])

    # Proceed if both files are uploaded
    if file_path_recommender is not None and file_path_sales is not None:
        # Load data for the recommender system
        df_recommender = load_data(file_path_recommender)
        interaction_matrix = create_interaction_matrix(df_recommender)

        # Train the autoencoder model for the recommender system
        train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)
        autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
        train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)
        encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
        embeddings = get_embeddings(encoder_model, interaction_matrix.values)
        user_similarity_matrix = compute_similarity_matrix(interaction_matrix, k=5)
        autoencoder_input = np.dot(user_similarity_matrix, embeddings)
        autoencoder_model = build_autoencoder_model(input_size=len(autoencoder_input[0]))
        train_autoencoder(autoencoder_model, autoencoder_input, autoencoder_input, epochs=10, batch_size=32)
        embeddings = get_embeddings(encoder_model, interaction_matrix.values)

        # Generate product recommendations for all clients
        top_recommendations_for_all_clients = {}
        for client_id in interaction_matrix.index:
            recommended_products, predicted_scores = recommend_products_hybrid(
                embeddings, user_similarity_matrix, client_id, interaction_matrix, k=5
            )
            top_recommendations_for_all_clients[int(client_id)] = {
                "recommended_products": recommended_products,
                "predicted_scores": predicted_scores
            }

        # Display recommendations for each client
        for client_id, recommendations_info in top_recommendations_for_all_clients.items():
            st.subheader(f"Top recommendations for Client {client_id}:")
            plot_results_recommender(df_recommender, recommendations_info)

        # Load data for the sales forecast
        df_sales = load_sales_data(file_path_sales)
        X_train_sales, y_train_sales, X_sales, y_sales = preprocess_sales_data(df_sales)

        # Train the LSTM model for the sales forecast
        sales_model = build_sales_model()
        history_sales = train_sales_model(sales_model, X_train_sales, y_train_sales)

        # Make sales predictions and display results
        predictions_sales, future_time_range_sales, future_predictions_sales = make_sales_predictions(sales_model, X_sales, randomness_factor=0.1)
        calculate_sales_mae(y_sales, predictions_sales)
        plot_results_sales(df_sales, predictions_sales, future_time_range_sales, future_predictions_sales, history_sales)
        
if __name__ == "__main__":
    main()
