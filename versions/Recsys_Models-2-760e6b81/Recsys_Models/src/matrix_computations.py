import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
file_path = '/Users/igormol/Desktop/Recsys_Models/samples/Marajo.csv'
data = pd.read_csv(file_path)

# Fill missing values
data.fillna('', inplace=True)

# Extract relevant columns
data = data[['Client_ID', 'Product_ID', 'Monetary_Value']]

# Create the user-item interaction matrix M
M = data.pivot_table(index='Client_ID', columns='Product_ID', values='Monetary_Value', aggfunc='sum', fill_value=0)

# Compute the user similarity matrix S using cosine similarity
user_sim_matrix = cosine_similarity(M)
S = pd.DataFrame(user_sim_matrix, index=M.index, columns=M.index)

# Compute the item similarity matrix T using cosine similarity
item_sim_matrix = cosine_similarity(M.T)
T = pd.DataFrame(item_sim_matrix, index=M.columns, columns=M.columns)

# Save matrices to CSV files
M.to_csv('/Users/igormol/Desktop/Recsys_Models/samples/user_item_matrix.csv')
S.to_csv('/Users/igormol/Desktop/Recsys_Models/samples/user_similarity_matrix.csv')
T.to_csv('/Users/igormol/Desktop/Recsys_Models/samples/item_similarity_matrix.csv')

print("Matrices have been successfully saved to CSV files.")
