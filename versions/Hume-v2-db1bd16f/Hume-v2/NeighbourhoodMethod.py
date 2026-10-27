# Neighborhood-based Recommendation Method
#
# Igor Mol <igor.mol@makes.ai>
#
# The following program utilizes a class-based approach to implement a movie 
# recommendation system. In technical terms, a class named "MovieRecommender" is
# created to encapsulate the entire recommendation process. This class contains 
# methods for loading data from a specified CSV file, extracting relevant ratings 
# information, creating a user-item matrix based on user and movie ratings, 
# standardizing the ratings using z-scores, and computing user similarity through
# cosine similarity metrics. The use of classes enhances code organization and encapsulation,
# facilitating modular development and maintenance.
# 	Upon instantiation of the "MovieRecommender" class with the file path of the
# movie ratings dataset, the program executes a sequence of methods to perform the 
# recommendation process. It begins by loading the data, checking for file existence
# and handling potential errors. Subsequently, the program extracts user ratings and
# constructs a user-item matrix, which forms the basis for the recommendation 
# calculations. Z-scores are applied to standardize the ratings, and cosine 
# similarity is computed between users to determine their likeness in terms of 
# movie preferences. This systematic approach improves code readability and makes
# it easier to understand and modify the recommendation system.
#	The recommendation process is finalized with the creation of a method to
# retrieve the top N movie recommendations for a given user. The method considers
# user similarity scores, filters out movies the user has already rated, and sorts
# the recommendations based on their predicted preference. Finally, the program 
# iterates through all users in the dataset, calling the recommendation method for
# each user and printing the top recommendations. This organized structure enhances
# the program's scalability, making it adaptable to larger datasets and facilitating
# the extension of functionalities for future improvements in the movie recommendation
# system.

import pandas as pd
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File not found at path: {self.file_path}")
            exit()
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()

    def extract_ratings(self):
        self.ratings_df = self.df[['userId', 'movieId', 'rating']]

    def create_user_item_matrix(self):
        self.user_item_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    def standardize_ratings(self):
        self.user_item_matrix_zscore = self.user_item_matrix.apply(zscore, axis=1)

    def compute_user_similarity(self):
        self.user_similarity = cosine_similarity(self.user_item_matrix_zscore)

    def get_top_n_recommendations(self, user_id, n=5):
        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_similarity_scores = self.user_similarity[user_index]

        # Get indices of users with highest similarity scores (excluding the user itself)
        similar_users_indices = user_similarity_scores.argsort()[:-n-1:-1]

        # Get movie recommendations from the most similar users
        user_movies = self.user_item_matrix.iloc[user_index]
        recommendations = self.user_item_matrix.iloc[similar_users_indices].mean(axis=0)
        recommendations = recommendations[~user_movies.astype(bool)]  # Filter out movies the user has already rated
        recommendations = recommendations.sort_values(ascending=False)[:n].index.tolist()

        return recommendations

    def print_recommendations_for_all_users(self):
        for user_id in self.user_item_matrix.index:
            top_recommendations = self.get_top_n_recommendations(user_id)
            print(f"Top Recommendations for User {user_id}: {top_recommendations}")

# File path for the CSV file
file_path = '/Volumes/KINGSTON/archive/ratings_small.csv'

# Create an instance of MovieRecommender
movie_recommender = MovieRecommender(file_path)

# Perform the recommendation process
movie_recommender.extract_ratings()
movie_recommender.create_user_item_matrix()
movie_recommender.standardize_ratings()
movie_recommender.compute_user_similarity()
movie_recommender.print_recommendations_for_all_users()
