# Attention Mechanism Approach for Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The program is designed to implement a movie recommender system using deep learning
# techniques with an attention mechanism. The recommender system is organized into 
# a class named RecommenderSystem to enhance code structure and maintainability. 
# The main functionalities, such as loading data, splitting it into training and 
# testing sets, building the recommendation model, training the model, making 
# predictions, and extracting top recommendations, are encapsulated within the class.
# 	In the main function, the recommender system is instantiated with the path 
# to the movie ratings dataset. The data is loaded, preprocessed, and split into 
# training and testing sets. The model architecture, involving embedding layers, 
# attention mechanism, and dense layers, is defined and compiled. The model is then
# trained using the training data. Subsequently, predictions are generated for the
# test set, and the top 5 movie recommendations for each user are extracted and 
# displayed. The refactoring of the code into a class-based structure contributes
# to code organization and readability.

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

class RecommenderSystem:
    def __init__(self, file_path, embedding_size=50):
        self.file_path = file_path
        self.embedding_size = embedding_size
        self.user_mapping = None
        self.movie_mapping = None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.file_path)
        self.user_mapping = {user_id: index for index, user_id in enumerate(df['userId'].unique())}
        self.movie_mapping = {movie_id: index for index, movie_id in enumerate(df['movieId'].unique())}
        df['user'] = df['userId'].map(self.user_mapping)
        df['movie'] = df['movieId'].map(self.movie_mapping)
        return df

    def split_data(self, data):
        return train_test_split(data, test_size=0.2, random_state=42)

    def build_model(self):
        user_input = keras.Input(shape=(1,), name='user_input')
        movie_input = keras.Input(shape=(1,), name='movie_input')

        user_embedding = layers.Embedding(input_dim=len(self.user_mapping), output_dim=self.embedding_size)(user_input)
        movie_embedding = layers.Embedding(input_dim=len(self.movie_mapping), output_dim=self.embedding_size)(movie_input)

        attention = layers.Attention()([user_embedding, movie_embedding])

        user_flat = layers.Flatten()(user_embedding)
        movie_flat = layers.Flatten()(movie_embedding)
        attention_flat = layers.Flatten()(attention)

        concatenated = layers.Concatenate()([user_flat, movie_flat, attention_flat])

        dense1 = layers.Dense(128, activation='relu')(concatenated)
        output = layers.Dense(1)(dense1)

        self.model = keras.Model(inputs=[user_input, movie_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, train_data):
        self.model.fit([train_data['user'], train_data['movie']], train_data['rating'], epochs=5, batch_size=64, validation_split=0.2)

    def predict_ratings(self, test_data):
        return self.model.predict([test_data['user'], test_data['movie']]).flatten()

    def recommend_top_k_movies(self, user_data, k=5):
        top_recommendations = user_data.nlargest(k, 'predicted_rating')
        return top_recommendations[['userId', 'movieId', 'predicted_rating']]

def main():
    file_path = "/Volumes/KINGSTON/archive/ratings_small.csv"

    recommender = RecommenderSystem(file_path)
    
    # Load data
    data = recommender.load_data()

    # Split data
    train_data, test_data = recommender.split_data(data)

    # Build and train the model
    recommender.build_model()
    recommender.train_model(train_data)

    # Make predictions on the test set
    test_data['predicted_rating'] = recommender.predict_ratings(test_data)

    # Get top 5 recommendations for each user
    top_recommendations = recommender.recommend_top_k_movies(test_data, k=5)

    # Display top recommendations
    print(top_recommendations)

if __name__ == "__main__":
    main()
