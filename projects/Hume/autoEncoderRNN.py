# autoEnconder/RNN Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The following code, based on the framework developed by arXiv:1807.01473, im-
# plements a hybrid recommendation system using deep learning techniques. The 
# code is organized into a class named RecommendationSystem, encapsulating the 
# functionality for building, training, and generating recommendations. The system
# is designed to handle user-item interactions stored in a CSV file. It utilizes 
# the TensorFlow and Keras libraries for constructing a recommendation model that
# combines the power of embeddings, a variational autoencoder (VAE) approach, and 
# a recurrent neural network (RNN) with Long Short-Term Memory (LSTM) cells.
# 	In the first part, the class initializes by loading the user-item interaction
# data from a CSV file. Two mapping dictionaries are created to convert user and 
# movie IDs into numerical indices for efficient processing. These indices are added 
# to the DataFrame to represent users and movies. The class also defines the number
# of unique users and movies in the dataset.
# 	Moving on to the second part, the code describes the model architecture
# within the build_model method. It employs embeddings to represent users and movies
# as dense vectors. These vectors are concatenated and fed into a hybrid structure,
# combining a dense layer, a variational autoencoder (VAE), and a recurrent neural
# network (RNN) with LSTM cells. The model is then compiled using the Adam optimizer
# and Mean Squared Error loss function.
#	The final paragraph explains the training and recommendation generation
# process. The train_model method trains the recommendation model using user and 
# movie indices and corresponding ratings. The generate_recommendations method 
# utilizes the trained model to predict ratings for all possible user-movie com-
# binations. These predictions are then organized into a DataFrame containing User
# IDs, Movie IDs, and the predicted ratings. The system, encapsulated in the class,
# is executed in the main function, providing a clear entry point for users to run the
# entire recommendation process.

import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, LSTM, RepeatVector, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

class RecommendationSystem:
    def __init__(self, csv_file_path):
        # Load the CSV file into a DataFrame and create mappings for users and movies
        self.df = pd.read_csv(csv_file_path)
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(self.df['User_ID'].unique())}
        self.movie_mapping = {movie_id: idx for idx, movie_id in enumerate(self.df['Movie_ID'].unique())}
        self.num_users = len(self.user_mapping)
        self.num_movies = len(self.movie_mapping)
        
    def map_ids_to_indices(self):
        # Map user and movie IDs to indices in the DataFrame
        self.df['User_Index'] = self.df['User_ID'].map(self.user_mapping)
        self.df['Movie_Index'] = self.df['Movie_ID'].map(self.movie_mapping)
        
    def build_model(self):
        # Build a hybrid recommendation model using Embeddings, LSTM, and Dense layers
        embedding_size = 50
        
        user_input = Input(shape=(1,))
        movie_input = Input(shape=(1,))
        
        user_embedding = Embedding(input_dim=self.num_users, output_dim=embedding_size)(user_input)
        movie_embedding = Embedding(input_dim=self.num_movies, output_dim=embedding_size)(movie_input)
        
        user_flatten = Flatten()(user_embedding)
        movie_flatten = Flatten()(movie_embedding)
        
        merged = Concatenate()([user_flatten, movie_flatten])
        
        latent_dim = 20
        
        h = Dense(latent_dim, activation='relu')(merged)
        h = RepeatVector(embedding_size)(h)
        
        lstm = LSTM(embedding_size, return_sequences=True)(h)
        lstm_flat = Flatten()(lstm)
        
        combined = Concatenate()([user_flatten, lstm_flat])
        
        output = Dense(1)(combined)
        
        model = Model(inputs=[user_input, movie_input], outputs=output)
        
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss=MeanSquaredError())
        
        return model
    
    def train_model(self, model, epochs=10, batch_size=64, validation_split=0.2):
        # Train the recommendation model
        X = [self.df['User_Index'].values, self.df['Movie_Index'].values]
        y = self.df['Rating'].values
        model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        
    def generate_recommendations(self, model):
        # Generate recommendations using the trained model
        user_indices = np.array([self.user_mapping[user_id] for user_id in self.df['User_ID'].unique()])
        movie_indices = np.array([self.movie_mapping[movie_id] for movie_id in self.df['Movie_ID'].unique()])
        
        user_movie_combinations = np.array(np.meshgrid(user_indices, movie_indices)).T.reshape(-1, 2)
        predictions = model.predict([user_movie_combinations[:, 0], user_movie_combinations[:, 1]])
        
        recommendations_df = pd.DataFrame({
            'User_ID': [list(self.user_mapping.keys())[idx] for idx in user_movie_combinations[:, 0]],
            'Movie_ID': [list(self.movie_mapping.keys())[idx] for idx in user_movie_combinations[:, 1]],
            'Rating_Prediction': predictions.flatten()
        })
        
        return recommendations_df


def main():
    # Define the path to the CSV file
    csv_file_path = "/Users/igormol/Desktop/recsys-light.csv"
    
    # Create an instance of the RecommendationSystem class
    rec_sys = RecommendationSystem(csv_file_path)
    
    # Map user and movie IDs to indices in the DataFrame
    rec_sys.map_ids_to_indices()
    
    # Build the recommendation model
    model = rec_sys.build_model()
    
    # Train the recommendation model
    rec_sys.train_model(model)
    
    # Generate recommendations using the trained model
    recommendations = rec_sys.generate_recommendations(model)
    
    # Display the recommendations
    print("Recommendations:")
    print(recommendations.head())


if __name__ == "__main__":
    main()
