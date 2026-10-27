# Restricted Boltzmann Machine Recommendation System
#
# Igor Mol <igor.mol@makes.ai>
#
# The following program implements a movie recommendation system using a class-based
# structure for clear organization and functionality. In simple terms, the code 
# follows a step-by-step process. Firstly, it loads movie ratings data from a 
# CSV file into a table-like structure using the pandas library. Relevant columns,
# such as user IDs, movie IDs, and ratings, are isolated, and a user-item matrix 
# is created to represent how users have rated different movies. The ratings are 
# then normalized, ensuring they fall within a standardized range.
# 	Secondly, the program splits the data into training and testing sets, a 
# common practice in machine learning. It constructs a recommendation model known 
# as a Restricted Boltzmann Machine (RBM) using TensorFlow and Keras. The RBM model
# is trained with the normalized data, optimizing its parameters to accurately predict
# user preferences. After training, the system utilizes the model to generate 
# personalized movie recommendations for each user. These recommendations are sorted
# based on predicted ratings, and the top suggestions are printed, showcasing a 
# user-friendly approach to movie recommendations through a well-organized class
# structure.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Define a class for movie recommendation
class MovieRecommender:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        # Load data from the specified CSV file into a pandas DataFrame
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File not found at path: {self.file_path}")
            exit()
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()

    def preprocess_data(self):
        # Extract relevant columns and create a user-item matrix
        ratings_df = self.df[['userId', 'movieId', 'rating']]
        self.user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        # Normalize the ratings to be between 0 and 1 using MinMaxScaler
        scaler = MinMaxScaler()
        self.user_item_matrix_scaled = scaler.fit_transform(self.user_item_matrix)

    def split_data(self):
        # Split the data into training and testing sets
        self.train_data, self.test_data = train_test_split(self.user_item_matrix_scaled, test_size=0.2, random_state=42)

    def build_rbm_model(self, hidden_units=50):
        # Build a Restricted Boltzmann Machine (RBM) model using TensorFlow and Keras
        visible_units = self.train_data.shape[1]
        self.rbm_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=hidden_units, activation='relu', input_shape=(visible_units,), name='hidden'),
            tf.keras.layers.Dense(units=visible_units, activation='sigmoid', name='visible')
        ])
        self.rbm_model.compile(optimizer='adam', loss='mean_squared_error')

    def train_rbm_model(self, num_epochs=10, batch_size=64):
        # Train the RBM model with the training data
        self.rbm_model.fit(self.train_data, self.train_data, batch_size=batch_size, epochs=num_epochs, verbose=1)

    def get_recommendations_for_all_users(self, num_recommendations=5):
        # Generate movie recommendations for all users
        all_users_recommendations = {}
        for user_id in self.user_item_matrix.index:
            user_index = self.user_item_matrix.index.get_loc(user_id)
            user_ratings = self.user_item_matrix_scaled[user_index:user_index + 1]
            predicted_ratings = self.rbm_model.predict(user_ratings)
            unrated_movies = np.where(user_ratings == 0)[1]
            sorted_movies = unrated_movies[np.argsort(predicted_ratings[0, unrated_movies])[::-1]]
            top_recommendations = sorted_movies[:num_recommendations]
            all_users_recommendations[user_id] = top_recommendations
        return all_users_recommendations

    def print_recommendations(self, recommendations_dict):
        # Print the recommendations for all users
        for user_id, recommendations in recommendations_dict.items():
            print(f"Top Recommendations for User {user_id}: {recommendations}")

# File path for the CSV file
file_path = '/Volumes/KINGSTON/archive/ratings_small.csv'

# Create an instance of MovieRecommender
movie_recommender = MovieRecommender(file_path)

# Perform the recommendation process
movie_recommender.preprocess_data()
movie_recommender.split_data()
movie_recommender.build_rbm_model()
movie_recommender.train_rbm_model()
recommendations = movie_recommender.get_recommendations_for_all_users()
movie_recommender.print_recommendations(recommendations)

