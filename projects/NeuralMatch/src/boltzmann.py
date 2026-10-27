
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tabulate import tabulate

# Load the original DataFrame
csv_file_path = '/Users/igormol/Downloads/online_retail_II.csv'
df = pd.read_csv(csv_file_path)

# Sum the 'Quantity' values for each unique pair of 'Customer ID' and 'StockCode'
user_item_interaction = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().reset_index()

# Create a user-item matrix (pivot table)
user_item_matrix = user_item_interaction.pivot_table(index='Customer ID', columns='StockCode', values='Quantity', fill_value=0)

# Convert the user-item matrix to a NumPy array
user_item_matrix_array = user_item_matrix.values

# Split the data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix_array, test_size=0.2, random_state=42)

# Build the Boltzmann Machine model
input_layer = Input(shape=(user_item_matrix.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(user_item_matrix.shape[1], activation='sigmoid')(encoded)

boltzmann_machine = Model(inputs=input_layer, outputs=decoded)

# Compile the model
boltzmann_machine.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Train the Boltzmann Machine
boltzmann_machine.fit(train_data, train_data, epochs=50, batch_size=256, shuffle=True, validation_data=(test_data, test_data))

# Get recommendations for all clients
all_customer_ids = user_item_matrix.index

for customer_id in all_customer_ids:
    # Get items the customer has not yet interacted with
    items_not_interacted = user_item_matrix.columns[user_item_matrix.loc[customer_id] == 0].tolist()

    # Predict ratings for items not interacted
    predictions = boltzmann_machine.predict(user_item_matrix.loc[[customer_id]])

    # Extract predicted ratings for items not interacted
    predicted_ratings = predictions[0]

    # Combine item codes with predicted ratings
    recommendations = list(zip(items_not_interacted, predicted_ratings))

    # Sort recommendations by predicted rating in descending order
    sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    # Get the top N recommendations (e.g., top 5)
    top_recommendations = sorted_recommendations[:5]

    # Print top recommendations for the current customer
    print(f"\nTop recommendations for Customer ID {customer_id}:\n")
    print(tabulate(top_recommendations, headers=['StockCode', 'Predicted Rating'], tablefmt='grid'))
