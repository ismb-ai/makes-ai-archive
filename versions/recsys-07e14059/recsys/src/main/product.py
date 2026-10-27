# Non-negative Matrix Factorization Recommender System (Products to Clients)

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from tabulate import tabulate

def load_data(file_path):
    # Load the main CSV file into a data frame.
    return pd.read_csv(file_path)

def load_clients_data(file_path):
    # Load the clients CSV file into a data frame.
    return pd.read_csv(file_path, low_memory=False)

def merge_data_frames(main_data, clients_data):
    # Merge data frames on Client_ID to get client names in the main data frame.
    return pd.merge(main_data, clients_data[['id', 'nome']], left_on='Client_ID', right_on='id', how='left')

def filter_substrings(data, column_name, substrings_to_remove):
    # Remove lines with specified sub-strings in a given column.
    conditions = data[column_name].str.upper().str.contains('|'.join(substrings_to_remove))
    return data[~conditions]

def create_item_user_matrix(data):
    # Create an item-user interaction matrix based on the frequency of purchase.
    return pd.crosstab(index=data['Product_ID'], columns=data['Client_ID'])

def apply_nmf(item_user_matrix):
    # Fit NMF model to the item-user matrix with an increased maximum number of iterations.
    model = NMF(n_components=5, init='random', random_state=42, max_iter=1000)  # Increased max_iter value
    W = model.fit_transform(item_user_matrix)
    H = model.components_
    return W, H

def get_top_recommendations(data, item_user_matrix, predicted_matrix):
    # Get top recommendations for each product.
    product_ids = item_user_matrix.index
    top_recommendations = {}

    for product_id in product_ids:
        clients_not_bought = set(data['Client_ID'].unique()) - set(item_user_matrix.columns[item_user_matrix.loc[product_id] > 0])

        recommended_clients_for_product = {client_id for _, _, _, client_id, _ in top_recommendations.get(product_id, [])}

        if not clients_not_bought:
            continue

        client_indices = [item_user_matrix.columns.get_loc(client_id) for client_id in clients_not_bought]
        product_index = np.where(item_user_matrix.index == product_id)[0][0]

        predictions_for_product = [(client_id, data[data['Client_ID'] == client_id]['nome'].values[0], predicted_matrix[product_index, client_idx]) for client_id, client_idx in zip(clients_not_bought, client_indices)]

        sorted_predictions = sorted(predictions_for_product, key=lambda x: x[2], reverse=True)
        filtered_predictions = [prediction for prediction in sorted_predictions if prediction[0] not in recommended_clients_for_product]

        top_recommendations[product_id] = filtered_predictions
        recommended_clients_for_product.update(prediction[0] for prediction in filtered_predictions)

    return top_recommendations

def write_recommendations_to_file(top_recommendations, data, file_path):
    # Write top recommendations and MAP at K for each product to the file.
    with open(file_path, "w") as file:
        for product_id in top_recommendations.keys():
            recommended_clients_with_names = get_recommended_clients_with_names(top_recommendations[product_id])

            file.write(f"Top recommendations for Product {product_id} ({get_product_info(product_id, data)}):\n")
            table_headers = ["Client ID", "Client Name", "Predicted Score"]
            table_data = [(client_id, client_name, score) for client_id, client_name, score in recommended_clients_with_names]
            file.write(tabulate(table_data, headers=table_headers, tablefmt="grid"))
            file.write("\n\n")

def get_recommended_clients_with_names(recommended_clients):
    # Get the recommended clients with names and formatted scores.
    return [(client_id, client_name, format(score, '.3g')) for client_id, client_name, score in recommended_clients]

def get_product_info(product_id, data):
    # Get the product's name and brand.
    product_name = data[data['Product_ID'] == product_id]['Product_Name'].values[0]
    brand_name = data[data['Product_ID'] == product_id]['Brand_Name'].values[0]
    return f"{product_name} - {brand_name}"

def main():
    # Load the main CSV file into a data frame.
    main_data = load_data("/Volumes/Untitled/cpp-complete.csv")

    # Load the clients CSV file into a data frame.
    clients_data = load_clients_data("/Volumes/Untitled/account_202401081403.csv")

    # Merge data frames on Client_ID to get client names in the main data frame.
    merged_data = merge_data_frames(main_data, clients_data)

    # Remove lines with specified sub-strings in "Product_Name" column.
    merged_data = filter_substrings(merged_data, 'Product_Name', ["SERVICOS", "DIVERSOS", "OLEO", "KIT", "CAPA", "INSTALACAO"])

    # Remove lines with specified sub-string in "Brand_Name" column.
    merged_data = filter_substrings(merged_data, 'Brand_Name', ["DEERE"])

    # Create an item-user interaction matrix based on the frequency of purchase.
    item_user_matrix = create_item_user_matrix(merged_data)

    # Fit NMF model to the item-user matrix with an increased maximum number of iterations.
    W, H = apply_nmf(item_user_matrix)

    # Reconstruct the original matrix.
    predicted_matrix = np.dot(W, H)

    # Get top recommendations for each product.
    top_recommendations = get_top_recommendations(merged_data, item_user_matrix, predicted_matrix)

    # Write top recommendations and MAP at K for each product to the file.
    write_recommendations_to_file(top_recommendations, merged_data, "/Volumes/Untitled/cpp-recsys-product.txt")

if __name__ == "__main__":
    main()
