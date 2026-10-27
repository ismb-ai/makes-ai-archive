
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tabulate import tabulate

# Load the original DataFrame
csv_file_path = '/Users/igormol/Downloads/online_retail_II.csv'
df = pd.read_csv(csv_file_path)

# Sum the 'Quantity' values for each unique pair of 'Customer ID' and 'StockCode'
user_item_interaction = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().reset_index()

# Create a product-client matrix (pivot table)
product_client_matrix = user_item_interaction.pivot_table(index='StockCode', columns='Customer ID', values='Quantity', fill_value=0)

# Convert the product-client matrix to a NumPy array
product_client_matrix_array = product_client_matrix.values

# Split the data into training and testing sets
train_data, test_data = train_test_split(product_client_matrix_array, test_size=0.2, random_state=42)

# Build the autoencoder model with dropout layers
input_layer = Input(shape=(product_client_matrix.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dropout(0.5)(encoded)  # Dropout layer added
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(product_client_matrix.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder with dropout
autoencoder.fit(train_data, train_data, epochs=50, batch_size=256, shuffle=True, validation_data=(test_data, test_data))

# Get recommendations for all products
all_product_ids = product_client_matrix.index

for product_id in all_product_ids:
    # Get clients who have not yet interacted with the product
    clients_not_interacted = product_client_matrix.columns[product_client_matrix.loc[product_id] == 0].tolist()

    # Predict ratings for clients not interacted
    predictions = autoencoder.predict(product_client_matrix.loc[[product_id]])

    # Extract predicted ratings for clients not interacted
    predicted_ratings = predictions[0]

    # Combine client IDs with predicted ratings
    recommendations = list(zip(clients_not_interacted, predicted_ratings))

    # Sort recommendations by predicted rating in descending order
    sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    # Get the top N recommendations (e.g., top 5)
    top_recommendations = sorted_recommendations[:5]

    # Print top recommendations for the current product
    print(f"\nTop recommendations for Product ID {product_id}:\n")
    print(tabulate(top_recommendations, headers=['Client ID', 'Predicted Rating'], tablefmt='grid'))
