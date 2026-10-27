# Restricted Boltzmann Machine Recommender System

# Igor Mol <igor.mol@makes.ai>

# In a collaborative recommender system, restricted Boltzmann machines are
# utilized to understand patterns in user-item interactions. RBMs learn from user
# preferences and item features, capturing relationships in the data. By modeling
# these interactions, RBMs can predict a user's interest in items they haven't
# encountered. This capability allows RBMs to offer personalized recommendations
# based on similarities among users, enhancing the collaborative filtering
# mechanism in recommender systems.

import pandas as pd
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The function load_data loads a CSV file into a DataFrame and performs data 
# cleaning operations.
# - Input: The function takes a file path as its parameter, representing the 
# path to the CSV file to be loaded.
# - Return: The function returns a DataFrame after applying data cleaning 
# operations.

def load_data(file_path):
    # Load the CSV file into a DataFrame.
    df = pd.read_csv(file_path)

    # Use the first occurrence of "Product_ID" for each unique "Product_Name".
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')

    # Filter lines based on specified conditions.
    substrings_to_exclude = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS',
                             'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    brand_substring_to_exclude = 'DEERE'

    # Exclude lines based on "Product_Name" containing specified substrings.
    df = df[~df['Product_Name'].str.contains('|'.join(substrings_to_exclude), case=False)]

    # Exclude lines based on "Product_Brand" containing specified substring.
    df = df[~df['Brand_Name'].str.contains(brand_substring_to_exclude, case=False)]

    # Return the cleaned DataFrame.
    return df


# The function create_interaction_matrix creates an interaction matrix from a 
# DataFrame, where rows represent unique "Client_ID," columns represent unique 
# "Product_ID," and the values indicate the interaction count between clients 
# and products.
# - Input: The function takes a DataFrame as its parameter, expecting it to have 
# columns 'Client_ID' and 'Product_ID'.
# - Return: The function returns an interaction matrix with clients as rows, 
# products as columns, and the matrix filled with interaction counts between 
# clients and products.

def create_interaction_matrix(data_frame):
    # Group the DataFrame by 'Client_ID' and 'Product_ID,' then calculate the size of
    # each group, and unstack the result. Fill missing values with 0.
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)


# The function instantiate_rbm_model creates a Restricted Boltzmann Machine
# based on the provided interaction matrix with optional parameters for maximum
# components, learning rate, and number of iterations.
# - Input: The function takes an interaction matrix, an optional parameter 'max_
# components' (default is None), a learning rate, and the number of iterations.
# - Return: The function returns an instantiated RBM model trained on the input
# interaction matrix.

def instantiate_rbm_model(interaction_matrix, max_components=None, learning_rate=0.01, n_iter=20):
    # Set the maximum number of components based on the number of products.
    if max_components is None:
        max_components = min(100, interaction_matrix.shape[1])

    # Limit the number of components to the specified maximum.
    n_components = min(max_components, interaction_matrix.shape[1])

    # Instantiate the RBM model with increased complexity and regularization.
    rbm_model = make_pipeline(StandardScaler(), BernoulliRBM(n_components=n_components, 
                                                             learning_rate=learning_rate, 
                                                             n_iter=n_iter, verbose=True))
    rbm_model.fit(interaction_matrix.values)

    # Return the instantiated and trained RBM model.
    return rbm_model

# The next function generates product recommendations for a given client using a
# Restricted Boltzmann Machine (RBM) model based on an interaction matrix.
# - Input: The function takes an interaction matrix, an RBM model, a specific 
# client ID, and an optional parameter 'k' (default is 5) for the number of 
# recommendations.
# - Return: The function returns a list of top 'k' recommended product IDs for 
# the specified client.

def get_recommendations_from_rbm(interaction_matrix, rbm_model, client_id, k=5):
    # Find the index corresponding to the given client ID in the interaction matrix.
    client_index = interaction_matrix.index.get_loc(client_id)
    
    # Extract the interaction data for the specified client and reshape it.
    client_data = interaction_matrix.iloc[client_index].values.reshape(1, -1)
    
    # Transform the client data using the RBM model to get recommendations.
    recommendations = rbm_model.transform(client_data)
    
    # Get the indices of the top 'k' recommendations and retrieve corresponding product IDs.
    top_indices = recommendations.argsort()[0, ::-1][:k]
    
    # Return the list of top 'k' recommended product IDs for the specified client.
    return interaction_matrix.columns[top_indices]

# The following function generates product recommendations for all clients using
# a pre-trained Restricted Boltzmann Machine (RBM) and an interaction matrix.
# - Input: The function takes an interaction matrix, a pre-trained RBM, and an
# optional parameter 'k' (default is 5) for the number of recommendations.
# - Return: The function returns a dictionary containing client IDs as keys and 
# their respective top recommended products as values.

def recommend_top_products_for_all_clients(interaction_matrix, rbm_model, k=5):
    # Get all client IDs from the interaction matrix.
    all_clients = interaction_matrix.index
    # Initialize an empty dictionary to store recommendations for all clients.
    all_recommendations = {}

    # Iterate through all clients and get top product recommendations.
    for client_id in all_clients:
        recommended_products = get_recommendations_from_rbm(interaction_matrix, rbm_model, client_id, k)
        
        # Only include clients with at least one recommendation.
        if len(recommended_products) > 0:
            all_recommendations[client_id] = {"recommended_products": recommended_products}

    # Return the dictionary with client IDs and their top recommended products.
    return all_recommendations


# The function compute_recall_at_k calculates Recall at K for client based on
# actual purchases and a list of recommended products.
# - Input: The function takes an interaction matrix, a specific client ID, a 
# list of recommended products, and an optional parameter 'k' (default is 15)
# for the number of recommendations considered in the recall calculation.
# - Return: The function returns the Recall at K value for the specified client.

def compute_recall_at_k(interaction_matrix, client_id, recommended_products, k=15):
    # Retrieve the actual purchases made by the given client.
    client_purchases = interaction_matrix.loc[client_id]
    
    # Calculate Recall at K: the ratio of correctly recommended products to total
    # actual purchases by the client.
    recall_at_k = len(set(recommended_products)) / len(client_purchases[client_purchases > 0])
    
    # Return the calculated Recall at K value.
    return recall_at_k

# The main routine organizes the recommendation system workflow. It loads
# sales data from a CSV file, creates an interaction matrix, trains a Restricted
# Boltzmann Machine, generates top product recommendations for all clients,
# computes Recall at K for each client, and writes the results to an output file.

def main():
    # File path to the sales data CSV file.
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    
    # Load sales data into a DataFrame.
    df = load_data(file_path)
    
    # Create an interaction matrix from the DataFrame.
    interaction_matrix = create_interaction_matrix(df)
    
    # Instantiate an RBM model with adjusted hyperparameters.
    rbm_model = instantiate_rbm_model(interaction_matrix, max_components=200, learning_rate=0.01, n_iter=20)
    
    # Generate top product recommendations for all clients using the RBM model.
    top_recommendations_for_all_clients = recommend_top_products_for_all_clients(interaction_matrix, rbm_model, k=15)

    # Output file path to store the results.
    output_file_path = "cpp-recsys-clients-rbm.txt"
    
    # Write the recommendations and recall information to the output file.
    with open(output_file_path, 'w') as output_file:
        for client_id, recommendations_info in top_recommendations_for_all_clients.items():
            recommended_products = recommendations_info["recommended_products"]
            
            # Calculate Recall at K for each client.
            recall_at_k = compute_recall_at_k(interaction_matrix, client_id, recommended_products, k=15)

            # Write client-specific information to the output file.
            output_file.write(f"Top recommendations for Client {client_id} (Recall at K: {recall_at_k:.4f}):\n")
            output_file.write(f"{'Product ID':<15} {'Product Name':<30} {'Brand':<20}\n")
            output_file.write("-" * 80 + "\n")
            
            # Write recommended products' details to the output file.
            for product_id in recommended_products:
                product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                output_file.write(f"{product_id:<15} {product_name:<30} {brand:<20}\n")
            
            output_file.write("\n")


if __name__ == "__main__":
    main()
