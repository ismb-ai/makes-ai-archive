import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def custom_standard_scaler(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    scaled_X = (X - mean) / std_dev
    return scaled_X

def custom_cosine_similarity(X):
    dot_product = X.dot(X.T)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    cosine_similarity = dot_product / (norms * norms.T)
    np.fill_diagonal(cosine_similarity, 0)
    return cosine_similarity

def custom_nearest_neighbors(X, n_neighbors=5, metric='cosine'):
    if metric == 'cosine':
        # Calculate cosine similarity matrix
        cosine_similarity = custom_cosine_similarity(X)

        # Find indices of top-k neighbors for each data point
        indices = np.argsort(cosine_similarity, axis=1)[:, ::-1][:, :n_neighbors]

        # Retrieve the distances (cosine similarities) and return them
        distances = cosine_similarity[np.arange(len(X))[:, None], indices]
        return distances.tolist()

    # Add other distance metrics as needed

def get_recommendations(article_id, features_scaled, df):
    article_idx = df[df['click_article_id'] == article_id].index.values[0]
    neighbors = custom_nearest_neighbors(features_scaled)
    recommended_articles = [df.loc[idx, 'click_article_id'] for idx, _ in enumerate(neighbors[article_idx])]
    return recommended_articles

def print_top_recommendations_for_all_users(df, features_scaled):
    unique_users = df['user_id'].unique()

    user_articles = df.groupby('user_id')['click_article_id'].unique()

    for user_id, user_articles_list in zip(unique_users, user_articles):
        print(f"\nTop recommendations for User {user_id}:")

        recommendations = get_recommendations(user_articles_list[0], features_scaled, df)

        # Filter out articles the user has already interacted with
        recommendations = np.setdiff1d(recommendations, user_articles_list)

        if recommendations.size > 0:
            print(f"Article {user_articles_list[0]}: {recommendations.tolist()}")

def main():
    # Load data
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
    df = load_data(file_path)

    # Preprocess data using custom StandardScaler
    features = df[['click_os', 'click_region', 'click_country', 'click_referrer_type']]
    features_scaled = custom_standard_scaler(features.values)

    # Call the function to print top recommendations for all users
    print_top_recommendations_for_all_users(df, features_scaled)

if __name__ == "__main__":
    main()
