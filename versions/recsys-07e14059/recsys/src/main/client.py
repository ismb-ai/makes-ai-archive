# Non-negative Matrix Factorization Recommender System (Clients to Products)

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import average_precision_score
from tabulate import tabulate

def load_data(file_path):
    # Load your data into a pandas DataFrame.
    return pd.read_csv(file_path)

def merge_dataframes(df, clients_df):
    # Merge the two DataFrames based on the 'Client_ID' in 'df' and 'id' in 'clients_df'.
    return pd.merge(df, clients_df, left_on='Client_ID', right_on='id', how='inner')

def select_relevant_columns(data):
    # Select only the relevant columns.
    return data[['Client_ID', 'Product_ID', 'Product_Name', 'Brand_Name', 'nome']]

def normalize_data(data):
    # Normalize the data frame.
    standard_product_ids = data.groupby('Product_Name')['Product_ID'].first().reset_index()
    data = pd.merge(data, standard_product_ids, on='Product_Name', how='left', suffixes=('', '_standard'))
    data['Product_ID'] = data['Product_ID_standard']
    data = data.drop('Product_ID_standard', axis=1)
    data = data[~data['Product_Name'].str.upper().str.contains('SERV')]
    return data

def create_user_item_matrix(data):
    # Create a user-item interaction matrix with purchase frequency.
    return data.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def apply_nmf(user_item_matrix):
    # Apply Non-Negative Matrix Factorization (NMF) with increased max_iter.
    model = NMF(n_components=10, init='random', random_state=42, max_iter=500)
    user_factors = model.fit_transform(user_item_matrix)
    product_factors = model.components_
    return user_factors, product_factors

def reconstruct_matrix(user_factors, product_factors):
    # Reconstruct the original matrix.
    return np.dot(user_factors, product_factors)

def get_top_recommendations(user_item_matrix, predicted_matrix, data):
    # Get top recommendations and compute MAP at K for all clients.
    client_ids = data['Client_ID'].unique()
    top_recommendations = {}

    for client_id in client_ids:
        products_not_bought = set(data['Product_ID'].unique()) - set(user_item_matrix.columns[user_item_matrix.loc[client_id] > 0])

        recommended_products_for_client = {product_id for product_id, _ in top_recommendations.get(client_id, [])}

        products_not_bought = [product_id for product_id in products_not_bought
                               if product_id not in recommended_products_for_client
                               and "INSTALACAO" not in data[data['Product_ID'] == product_id]['Product_Name'].values[0].upper()
                               and "OLEO" not in data[data['Product_ID'] == product_id]['Product_Name'].values[0].upper()
                               and "CAPA" not in data[data['Product_ID'] == product_id]['Product_Name'].values[0].upper()
                               and "DEERE" not in data[data['Product_ID'] == product_id]['Brand_Name'].values[0].upper()
                               and "KIT" not in data[data['Product_ID'] == product_id]['Product_Name'].values[0].upper()
                               and "DIVERSOS" not in data[data['Product_ID'] == product_id]['Brand_Name'].values[0].upper()
                               and "SERV" not in data[data['Product_ID'] == product_id]['Product_Name'].values[0].upper()]

        if not products_not_bought:
            continue

        product_indices = [user_item_matrix.columns.get_loc(product_id) for product_id in products_not_bought]
        client_index = np.where(user_item_matrix.index == client_id)[0][0]

        predictions_for_client = [(product_id, predicted_matrix[client_index, product_idx]) for product_id, product_idx in zip(products_not_bought, product_indices)]

        top_recommendations[client_id] = sorted(predictions_for_client, key=lambda x: x[1], reverse=True)[:5]

    return top_recommendations

def write_recommendations_to_file(top_recommendations, data, file_path):
    # Write top recommendations and MAP at K for each client to the file.
    with open(file_path, "w") as file:
        for client_id in top_recommendations.keys():
            recommended_items_with_names = get_recommended_items_with_names(top_recommendations[client_id], data)

            file.write(f"Top recommendations for Client {client_id} ({get_client_name(client_id, data)}):\n")
            table_headers = ["Product ID", "Product Name", "Brand Name", "Predicted Score"]
            table_data = [(product_id, product_name, brand_name, score) for product_id, product_name, brand_name, score in recommended_items_with_names]
            file.write(tabulate(table_data, headers=table_headers, tablefmt="grid"))
            file.write("\n\n")

def get_recommended_items_with_names(recommended_items, data):
    # Get the recommended items with names and formatted scores.
    return [(product_id,
             data[data['Product_ID'] == product_id]['Product_Name'].values[0],
             data[data['Product_ID'] == product_id]['Brand_Name'].values[0],
             format(score, '.3g'))
            for product_id, score in recommended_items]

def get_client_name(client_id, data):
    # Get the client's name.
    return data[data['Client_ID'] == client_id]['nome'].values[0]

def main():
    # Load your data into a pandas DataFrame.
    file_path = "/Volumes/Untitled/cpp-complete.csv"
    df = load_data(file_path)

    # Load clients data into another DataFrame.
    clients_file_path = "/Volumes/Untitled/account_202401081403.csv"
    clients_df = load_data(clients_file_path)

    # Merge the two DataFrames based on the 'Client_ID' in 'df' and 'id' in 'clients_df'.
    merged_df = merge_dataframes(df, clients_df)

    # Select only the relevant columns.
    data = select_relevant_columns(merged_df)

    # Normalize the data frame.
    data = normalize_data(data)

    # Create a user-item interaction matrix with purchase frequency.
    user_item_matrix = create_user_item_matrix(data)

    # Apply Non-Negative Matrix Factorization (NMF) with increased max_iter.
    user_factors, product_factors = apply_nmf(user_item_matrix)

    # Reconstruct the original matrix.
    predicted_matrix = reconstruct_matrix(user_factors, product_factors)

    # Get top recommendations and compute MAP at K for all clients.
    top_recommendations = get_top_recommendations(user_item_matrix, predicted_matrix, data)

    # Write top recommendations and MAP at K for each client to the file.
    write_recommendations_to_file(top_recommendations, data, "/Volumes/Untitled/cpp-recsys-client.txt")

if __name__ == "__main__":
    main()
