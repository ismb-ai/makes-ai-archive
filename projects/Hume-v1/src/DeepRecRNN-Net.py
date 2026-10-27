# Neural Collaborative Filtering Recommendation System
#
# Igor Mol <igor.mol@makes.ai>
#
# The following Python code implements a recommendation system using a hybrid deep
# learning approach, combining Neural Collaborative Filtering (NCF), a Recurrent
# Neural Network (RNN), and an Attention Mechanism. The code is organized into a 
# class named HybridRecommendationSystem, which encapsulates the functionality for
# loading and preprocessing data, building individual models (NCF, RNN, and Attention),
# and constructing a hybrid model that integrates all three components. The 
# recommendation system is designed to predict user ratings for movies.
# 	The HybridRecommendationSystem class first loads and preprocesses the data 
# from a CSV file containing user, movie, and rating information. It then constructs
# the NCF model, the RNN model, and an Attention Mechanism using the Keras library.
# The NCF model uses embeddings for users and movies, concatenates them, and passes
# the result through dense layers. The RNN model processes sequences of user embeddings,
# and the Attention Mechanism computes attention scores between the outputs of the
# RNN and NCF models. Finally, the code combines these models into a hybrid 
# recommendation system, which is trained and evaluated on the provided data.
# 	The training and evaluation process involve compiling the model, fitting 
# it to the training data, and assessing its performance on a test set. The output
# includes the test loss and a table displaying the actual and predicted ratings 
# for user-movie pairs. This approach leverages the strengths of collaborative 
# filtering, sequential modeling, and attention mechanisms to enhance the recom-
# mendation system's accuracy and effectiveness.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, LSTM, Attention
from keras.models import Model

class HybridRecommendationSystem:
    def __init__(self, csv_file_path):
        # Initialize the class with the path to the CSV file
        self.df = pd.read_csv(csv_file_path)
        self.user_mapping = {}
        self.movie_mapping = {}
        self.user_embedding_size = 50
        self.movie_embedding_size = 50
        self.sequence_length = 5  # Length of the user sequence for the RNN

    def load_and_preprocess_data(self):
        # Create mappings for User_ID and Movie_ID
        self.user_mapping = {id: i for i, id in enumerate(self.df['User_ID'].unique())}
        self.movie_mapping = {id: i for i, id in enumerate(self.df['Movie_ID'].unique())}

        # Apply the mappings to the DataFrame
        self.df['User_ID'] = self.df['User_ID'].map(self.user_mapping)
        self.df['Movie_ID'] = self.df['Movie_ID'].map(self.movie_mapping)

    def build_ncf_model(self):
        # Build the Neural Collaborative Filtering (NCF) model
        user_input = Input(shape=(1,))
        movie_input = Input(shape=(1,))

        user_embedding = Embedding(len(self.user_mapping), self.user_embedding_size)(user_input)
        movie_embedding = Embedding(len(self.movie_mapping), self.movie_embedding_size)(movie_input)

        ncf_concat = Concatenate()([Flatten()(user_embedding), Flatten()(movie_embedding)])
        ncf_output = Dense(50)(ncf_concat)

        return Model(inputs=[user_input, movie_input], outputs=ncf_output)

    def build_rnn_model(self):
        # Build the Recurrent Neural Network (RNN) model
        rnn_input = Input(shape=(self.sequence_length, self.user_embedding_size))
        rnn_output = LSTM(50)(rnn_input)

        return Model(inputs=rnn_input, outputs=rnn_output)

    def build_attention_mechanism(self):
        # Build the Attention Mechanism
        return Attention()

    def build_hybrid_model(self, ncf_model, rnn_model, attention_model):
        # Build the hybrid model combining NCF, RNN, and Attention
        user_input = Input(shape=(1,))
        movie_input = Input(shape=(1,))
        rnn_input = Input(shape=(self.sequence_length, self.user_embedding_size))

        ncf_output = ncf_model([user_input, movie_input])
        rnn_output = rnn_model(rnn_input)
        attention_output = attention_model([Dense(50)(rnn_output), ncf_output])

        merged_output = Concatenate()([ncf_output, rnn_output, attention_output])
        final_output = Dense(1)(merged_output)

        return Model(inputs=[user_input, movie_input, rnn_input], outputs=final_output)

    def train_and_evaluate_model(self, model, train_data, test_data, epochs=10, batch_size=32):
        # Compile and train the hybrid model, then evaluate on the test set
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit([train_data['User_ID'], train_data['Movie_ID'], np.random.random((len(train_data), self.sequence_length, self.user_embedding_size))],
                  train_data['Rating'], epochs=epochs, batch_size=batch_size, validation_split=0.2)

        test_loss = model.evaluate([test_data['User_ID'], test_data['Movie_ID'], np.random.random((len(test_data), self.sequence_length, self.user_embedding_size))],
                                   test_data['Rating'])
        print(f'Test Loss: {test_loss}')

        # Print the result in a table
        predictions = model.predict([test_data['User_ID'], test_data['Movie_ID'], np.random.random((len(test_data), self.sequence_length, self.user_embedding_size))])
        result_df = pd.DataFrame({'User_ID': test_data['User_ID'], 'Movie_ID': test_data['Movie_ID'],
                                  'Actual_Rating': test_data['Rating'], 'Predicted_Rating': predictions.flatten()})
        print(result_df)


if __name__ == "__main__":
    # Instantiate the HybridRecommendationSystem class and load data
    csv_file_path = "/Users/igormol/Desktop/recsys-light.csv"
    recsys = HybridRecommendationSystem(csv_file_path)
    recsys.load_and_preprocess_data()

    # Split data into training and testing sets
    train_df, test_df = train_test_split(recsys.df, test_size=0.2, random_state=42)

    # Build individual models and hybrid model
    ncf_model = recsys.build_ncf_model()
    rnn_model = recsys.build_rnn_model()
    attention_model = recsys.build_attention_mechanism()
    hybrid_model = recsys.build_hybrid_model(ncf_model, rnn_model, attention_model)

    # Train and evaluate the hybrid model
    recsys.train_and_evaluate_model(hybrid_model, train_df, test_df, epochs=10)
