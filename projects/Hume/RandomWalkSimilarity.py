import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Load the CSV file into a pandas DataFrame
file_path = '/Volumes/KINGSTON/archive/ratings_small.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at path: {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Extract relevant columns
ratings_df = df[['userId', 'movieId', 'rating']]

# Create user-item matrix
user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Initialize the Nearest Neighbors model
k_neighbors = 5  # You can adjust this number based on your requirements
nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', algorithm='brute')
nn_model.fit(train_data)

# Function to get top N recommendations for a given user
def get_top_n_recommendations(user_id, user_item_matrix, nn_model, n=5):
    user_index = user_item_matrix.index.get_loc(user_id)
    
    # Find the k-nearest neighbors of the user
    _, neighbor_indices = nn_model.kneighbors([user_item_matrix.iloc[user_index]])

    # Extract movies rated by neighbors but not by the user
    unrated_movies = user_item_matrix.iloc[user_index][user_item_matrix.columns.isin(neighbor_indices.flatten())]
    recommendations = unrated_movies[unrated_movies == 0].index.tolist()

    return recommendations[:n]

# Print recommendations for all users
all_users_recommendations = {}
for user_id in user_item_matrix.index:
    top_recommendations = get_top_n_recommendations(user_id, user_item_matrix, nn_model)
    all_users_recommendations[user_id] = top_recommendations

# Print recommendations for all users
for user_id, recommendations in all_users_recommendations.items():
    print(f"Top Recommendations for User {user_id}: {recommendations}")
