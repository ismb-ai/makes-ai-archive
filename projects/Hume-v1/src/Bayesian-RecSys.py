# Bayesian Recommendation System
#
# Igor Mol <igor.mol@makes.ai>
#
# The following code implements a Hybrid Recommender System, designed to predict 
# user ratings for movies. The code is organized into a class called HybridRecommender,
# encapsulating the functionality of loading and preprocessing data, building the hybrid
# model architecture, training the model, and evaluating its performance. The class 
# utilizes various layers such as Embedding, Conv1D, Dense, Lambda, Concatenate, and
# Attention from the TensorFlow and Keras libraries to construct a sophisticated neural
# network. This network combines collaborative filtering (BPR) with convolutional and 
# attention-based features, providing a comprehensive approach to capturing user-movie
# interactions.
# 	The load_and_preprocess_data method reads a CSV file containing user-movie
# interaction data, maps user and movie IDs to integer indices, and prepares the
# data for training. The build_model method defines the architecture of the hybrid
# model, incorporating embedding layers for users and movies, a convolutional neural
# network (CNN) for capturing spatial patterns, Bayesian Personalized Ranking (BPR) 
# for personalized recommendations, and an attention mechanism to focus on relevant 
# information. The model is trained using the train_model method, which leverages the
# training data to optimize the model's parameters. Finally, the evaluate_model method
# assesses the model's performance on a separate test set, and the predict_ratings 
# method generates predicted ratings for movies in the test set, allowing for a detailed
# analysis of the model's effectiveness.
# 	The example usage section demonstrates how to instantiate the HybridRecommender 
# class, load and preprocess data, split it into training and testing sets, train the 
# hybrid model, evaluate its performance, and make predictions on the test set. This
# modular and structured approach enhances code readability, maintainability, and
# reusability, providing a foundation for building and experimenting with sophisticated
# recommender systems in the field of collaborative filtering.

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dense, Flatten, Lambda, Concatenate, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Class definition for the Hybrid Recommender System
class HybridRecommender:
    def __init__(self, user_vocab_size, movie_vocab_size):
        self.user_vocab_size = user_vocab_size
        self.movie_vocab_size = movie_vocab_size
        self.model = self.build_model()  # Initialize the hybrid model when an instance is created

    # Load and preprocess data method
    def load_and_preprocess_data(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        user_mapping = {id: i for i, id in enumerate(df['User_ID'].unique())}
        movie_mapping = {id: i for i, id in enumerate(df['Movie_ID'].unique())}
        df['User_ID'] = df['User_ID'].map(user_mapping)
        df['Movie_ID'] = df['Movie_ID'].map(movie_mapping)
        return df, user_mapping, movie_mapping

    # Method to build the hybrid model
    def build_model(self):
        user_input = Input(shape=(1,))
        movie_input = Input(shape=(1,))

        # Embedding layers
        user_embedding = Embedding(self.user_vocab_size, 50)(user_input)
        movie_embedding = Embedding(self.movie_vocab_size, 50)(movie_input)
        movie_embedding_reshaped = Lambda(lambda x: K.expand_dims(x, axis=-1))(movie_embedding)

        # Convolutional Neural Network (CNN)
        cnn_output = Conv1D(32, kernel_size=3, activation='relu')(movie_embedding_reshaped)
        cnn_output = Flatten()(cnn_output)

        # Bayesian Personalized Ranking (BPR)
        positive_movie_embedding = Embedding(self.movie_vocab_size, 50)(Lambda(lambda x: x[:, 0])(movie_input))
        negative_movie_embedding = Embedding(self.movie_vocab_size, 50)(Lambda(lambda x: K.expand_dims(x[:, 0], axis=-1))(movie_input))

        positive_interaction = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([Flatten()(user_embedding), Flatten()(positive_movie_embedding)])
        negative_interaction = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([Flatten()(user_embedding), Flatten()(negative_movie_embedding)])

        bpr_loss = Lambda(lambda x: K.log(K.sigmoid(x[0] - x[1]) + K.epsilon()))([positive_interaction, negative_interaction])

        # Attention Mechanism
        attention_output = Attention()([user_embedding, movie_embedding])

        # Concatenate all components
        merged_output = Concatenate()([Flatten()(user_embedding), cnn_output, Flatten()(attention_output)])
        fc1 = Dense(100, activation='relu')(merged_output)
        output = Dense(1, activation='sigmoid')(fc1)

        # Model definition
        model = Model(inputs=[user_input, movie_input], outputs=output)
        model.add_loss(bpr_loss)
        model.compile(optimizer=Adam(learning_rate=0.001))

        return model

    # Method to train the hybrid model
    def train_model(self, train_data, epochs=10, batch_size=32, validation_split=0.2):
        self.model.fit([train_data['User_ID'], train_data['Movie_ID']], epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Method to evaluate the hybrid model on the test set
    def evaluate_model(self, test_data):
        test_loss = self.model.evaluate([test_data['User_ID'], test_data['Movie_ID']])
        print(f'Test Loss: {test_loss}')

    # Method to predict ratings on the test set
    def predict_ratings(self, test_data):
        predictions = self.model.predict([test_data['User_ID'], test_data['Movie_ID']])
        result_df = pd.DataFrame({'User_ID': test_data['User_ID'], 'Movie_ID': test_data['Movie_ID'],
                                  'Actual_Rating': test_data['Rating'], 'Predicted_Rating': predictions.flatten()})
        return result_df


# Example usage
csv_file_path = "/Users/igormol/Desktop/recsys-light.csv"
user_vocab_size =  # Define user_vocab_size
movie_vocab_size =  # Define movie_vocab_size

# Create an instance of HybridRecommender
hybrid_recommender = HybridRecommender(user_vocab_size, movie_vocab_size)

# Load and preprocess data
df, _, _ = hybrid_recommender.load_and_preprocess_data(csv_file_path)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train the hybrid model
hybrid_recommender.train_model(train_df)

# Evaluate the model on the test set
hybrid_recommender.evaluate_model(test_df)

# Predict ratings on the test set and print the results
result_df = hybrid_recommender.predict_ratings(test_df)
print(result_df)

