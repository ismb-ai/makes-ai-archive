# Wide & Deep Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The following code achieves the development of a recommendation system that 
# provides personalized movie ratings. It combines two key components: a wide 
# component, using a Generalized Linear Model (GLM), and an embedding component,
# incorporating a Recurrent Neural Network (RNN) with an Attention Mechanism. The
# wide component captures global patterns, while the embedding component captures
# intricate user preferences. The hybrid model is trained on a dataset containing
# user and movie interactions, mapping User_ID and Movie_ID to numerical indices 
# for efficient processing.
# 	The code begins by loading and preprocessing the dataset, creating mappings
# for unique User_IDs and Movie_IDs. It then builds the wide component, which is 
# essentially a linear model capturing general relationships between users and movies.
# Simultaneously, the embedding model captures nuanced patterns through the use of
# embeddings and recurrent layers. The two components are concatenated to form a
# comprehensive hybrid model. This model is trained using the Mean Squared Error
# loss function and the Adam optimizer, iterating over the dataset for a specified
# number of epochs. The trained model is evaluated on a test set, and its performance
# metrics, such as loss and predicted ratings, are printed.
# 	The architecture consists of three main components: the wide component,
# the embedding model, and the hybrid model. The wide component uses a linear model
# to combine user and movie information. The embedding model utilizes embeddings 
# and recurrent layers to understand sequential dependencies and attention mechanisms
# to focus on relevant information. The hybrid model merges the outputs of the wide
# and embedding components, creating a comprehensive model capable of capturing both
# global and intricate patterns. The model is compiled and trained using Keras, a 
# high-level neural networks API, and TensorFlow, an open-source machine learning
# framework. The training process involves adjusting model weights to minimize the
# difference between predicted and actual movie ratings. The final model is capable
# of making personalized movie rating predictions for users based on their historica
#l interactions.

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, Flatten
from keras.models import Model

class RecommendationSystem:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def load_and_preprocess_data(self):
        # Load data from CSV file
        df = pd.read_csv(self.csv_file_path)

        # Create mappings for User_ID and Movie_ID
        user_mapping = {id: i for i, id in enumerate(df['User_ID'].unique())}
        movie_mapping = {id: i for i, id in enumerate(df['Movie_ID'].unique())}

        # Apply the mappings to the DataFrame
        df['User_ID'] = df['User_ID'].map(user_mapping)
        df['Movie_ID'] = df['Movie_ID'].map(movie_mapping)

        return df, user_mapping, movie_mapping

    def build_wide_component(self, user_input, movie_input):
        # Build the wide component (Generalized Linear Model)
        wide_branch = Concatenate()([user_input, movie_input])
        return Dense(1, activation='linear')(wide_branch)

    def build_embedding_model(self, user_input, movie_input, user_vocab_size, movie_vocab_size):
        # Build the embedding model (Recurrent Neural Network with Attention Mechanism)
        user_embedding = Embedding(user_vocab_size, 50)(user_input)
        movie_embedding = Embedding(movie_vocab_size, 50)(movie_input)
        rnn_output = LSTM(50)(user_embedding)
        attention_output = Attention()([tf.expand_dims(rnn_output, axis=1), movie_embedding])
        return Flatten()(attention_output)

    def build_hybrid_model(self, user_vocab_size, movie_vocab_size):
        # Build the hybrid model combining wide and embedding components
        user_input = Input(shape=(1,))
        movie_input = Input(shape=(1,))
        wide_component = self.build_wide_component(user_input, movie_input)
        embedding_model = self.build_embedding_model(user_input, movie_input, user_vocab_size, movie_vocab_size)
        merged_output = Concatenate()([wide_component, embedding_model])
        fc1 = Dense(100, activation='relu')(merged_output)
        output = Dense(1)(fc1)
        model = Model(inputs=[user_input, movie_input], outputs=output)
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train_and_evaluate_model(self, model, train_data, test_data, epochs=10, batch_size=32, validation_split=0.2):
        # Train the hybrid model and evaluate on the test set
        model.fit([train_data['User_ID'], train_data['Movie_ID']], train_data['Rating'],
                  epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        test_loss = model.evaluate([test_data['User_ID'], test_data['Movie_ID']], test_data['Rating'])
        print(f'Test Loss: {test_loss}')

        predictions = model.predict([test_data['User_ID'], test_data['Movie_ID']])
        result_df = pd.DataFrame({'User_ID': test_data['User_ID'], 'Movie_ID': test_data['Movie_ID'],
                                  'Actual_Rating': test_data['Rating'], 'Predicted_Rating': predictions.flatten()})
        print(result_df)

if __name__ == "__main__":
    # Initialize the RecommendationSystem with the CSV file path
    rec_sys = RecommendationSystem("/Users/igormol/Desktop/recsys-light.csv")

    # Load and preprocess data
    df, user_mapping, movie_mapping = rec_sys.load_and_preprocess_data()

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Calculate vocabulary sizes
    user_vocab_size = len(user_mapping) + 1
    movie_vocab_size = len(movie_mapping) + 1

    # Build and train the hybrid model
    hybrid_model = rec_sys.build_hybrid_model(user_vocab_size, movie_vocab_size)
    rec_sys.train_and_evaluate_model(hybrid_model, train_df, test_df, epochs=10, batch_size=32, validation_split=0.2)
