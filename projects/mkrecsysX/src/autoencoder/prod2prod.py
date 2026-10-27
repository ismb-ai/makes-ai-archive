import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

def load_data(file_path):
    return pd.read_csv(file_path)

def create_interaction_matrix(data_frame):
    return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

def instantiate_similarity_matrix(interaction_matrix):
    similarity_matrix = cosine_similarity(interaction_matrix.T)
    return similarity_matrix

def get_similar_products(similarity_matrix, interaction_matrix, product_id, k=5):
    product_index = interaction_matrix.columns.get_loc(product_id)
    similar_indices = similarity_matrix[product_index].argsort()[:-k-1:-1]
    similar_products = interaction_matrix.columns[similar_indices].tolist()[1:]
    return similar_products

def recommend_products_for_product(similarity_matrix, interaction_matrix, product_id, k=5):
    similar_products = get_similar_products(similarity_matrix, interaction_matrix, product_id, k)
    
    # Calculate recall at k
    relevant_items = interaction_matrix[interaction_matrix[product_id] > 0].index.tolist()
    recommended_relevant = sum(1 for item in similar_products if item in relevant_items)
    recall_at_k = len(relevant_items)
    
    return similar_products, recall_at_k

def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = load_data(file_path)
    interaction_matrix = create_interaction_matrix(df)
    similarity_matrix = instantiate_similarity_matrix(interaction_matrix)
    
    all_recommendations = {}
    all_recall_at_k = {}
    
    for product_id in interaction_matrix.columns:
        recommended_products, recall_at_k = recommend_products_for_product(similarity_matrix, interaction_matrix, product_id, k=5)
        all_recommendations[product_id] = recommended_products
        all_recall_at_k[product_id] = recall_at_k
    
    # Display recommendations for all products along with recall at k
    for product_id, recommended_products in all_recommendations.items():
        recall = all_recall_at_k[product_id]
        recommended_product_info = df[df['Product_ID'].isin(recommended_products)][['Product_ID', 'Product_Name', 'Brand_Name']].drop_duplicates()
        print(f"Recommendations for Product ID: {product_id} (Recall at K: {recall:.2f})")
        print(tabulate(recommended_product_info, headers='keys', tablefmt='grid'))
        print("\n")

if __name__ == "__main__":
    main()
