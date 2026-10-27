# Deep Learning Approach to Rec-sys
#
# Igor Mol <igor.mol@makes.ai>
#
# The following code implements a collaborative filtering recommendation system 
# using a neural network. The code begins by loading and preprocessing data from
# a CSV file, creating mappings for User_ID and Movie_ID, and applying these 
# mappings to the DataFrame. The user and movie IDs are then mapped to continuous
# indices to facilitate neural network processing. The collaborative filtering 
# model is built using Keras, with embeddings for user and movie IDs, concatenated
# and passed through dense layers with dropout to predict the rating. The model 
# is compiled with the mean squared error loss function and the Adam optimizer.
# 	The train_and_evaluate_model function is defined to train the collaborative
# filtering model on the training data and evaluate its performance on the test 
# data. The model is trained for a specified number of epochs, and the test loss
# is printed. Additionally, the function prints a table of actual and predicted 
# ratings for each user and movie combination in the test set. The training and
# evaluation process is executed in the __main__ block, where the CSV file path 
# is provided, data is preprocessed, and the collaborative model is trained and 
# evaluated.
# 	The collaborative filtering model architecture includes an embedding 
# layer for both user and movie IDs, followed by dense layers with dropout for
# increased complexity. The training process involves minimizing the mean squared
# error between predicted and actual ratings. The model's predictions and evaluation
# results are then printed for further analysis. This code provides a foundational
# structure for building and training collaborative filtering models for recommendation
# systems using neural networks.

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from keras.models import Model

def load_and_preprocess_data(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Create mappings for User_ID and Movie_ID
    user_mapping = {id: i for i, id in enumerate(df['User_ID'].unique())}
    movie_mapping = {id: i for i, id in enumerate(df['Movie_ID'].unique())}

    # Apply the mappings to the DataFrame
    df['User_ID'] = df['User_ID'].map(user_mapping)
    df['Movie_ID'] = df['Movie_ID'].map(movie_mapping)

    return df, user_mapping, movie_mapping

def build_collaborative_model(user_vocab_size, movie_vocab_size):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    user_embedding = Embedding(user_vocab_size, 100, input_length=1)(user_input)
    movie_embedding = Embedding(movie_vocab_size, 100, input_length=1)(movie_input)

    user_flat = Flatten()(user_embedding)
    movie_flat = Flatten()(movie_embedding)

    concatenated = Concatenate()([user_flat, movie_flat])
    fc1 = Dense(200, activation='relu')(concatenated)
    dropout1 = Dropout(0.3)(fc1)
    fc2 = Dense(100, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(fc2)
    output = Dense(1)(dropout2)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def train_and_evaluate_model(model, train_data, test_data, epochs=20):
    model.fit([train_data['User_ID'], train_data['Movie_ID']], train_data['Rating'],
              epochs=epochs, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    test_loss = model.evaluate([test_data['User_ID'], test_data['Movie_ID']], test_data['Rating'])
    print(f'Test Loss: {test_loss}')

    # Print the result in a table
    predictions = model.predict([test_data['User_ID'], test_data['Movie_ID']])
    result_df = pd.DataFrame({'User_ID': test_data['User_ID'], 'Movie_ID': test_data['Movie_ID'],
                              'Actual_Rating': test_data['Rating'], 'Predicted_Rating': predictions.flatten()})
    print(result_df)

if __name__ == "__main__":
    csv_file_path = "/Users/igormol/Desktop/recsys-light.csv"
    df, user_mapping, movie_mapping = load_and_preprocess_data(csv_file_path)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    user_vocab_size = len(user_mapping) + 1  # Add 1 for 0-based indexing
    movie_vocab_size = len(movie_mapping) + 1

    collaborative_model = build_collaborative_model(user_vocab_size, movie_vocab_size)

    # Train and evaluate the model with increased complexity
    train_and_evaluate_model(collaborative_model, train_df, test_df, epochs=30)
