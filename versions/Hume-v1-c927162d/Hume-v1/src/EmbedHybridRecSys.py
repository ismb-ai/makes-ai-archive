# Deep Learning Recommendation System: An Embedding Approach
#
# Igor Mol <igor.mol@makes.ai>
#
# The following code implements a hybrid recommendation system using deep learning 
# techniques, particularly incorporating embeddings, LSTM (Long Short-Term Memory)
# layers, and attention mechanisms. The program starts by loading and preprocessing
# data from a CSV file, creating mappings for User_ID and Movie_ID, and applying 
# these mappings to the dataset. It then defines a hybrid model architecture using 
# the Keras library. The model comprises user and movie embeddings, an LSTM layer 
# for sequential data processing, an attention mechanism to capture relevant 
# information, and a dense neural network for prediction. The training and 
# evaluation of the model are performed on a dataset split into training and 
# testing sets, with the final predictions displayed in a table.
# 	The load_and_preprocess_data function is responsible for reading the 
# data from a specified CSV file, creating unique mappings for users and movies,
# and transforming the dataset accordingly. The core functionality lies in the
# build_hybrid_model function, where the hybrid recommendation model is constructed
# with user and movie embeddings, an LSTM layer, attention mechanism, and dense 
# layers. The model is trained and evaluated using the training and testing sets,
# and the predictions are presented in a structured table. The code showcases a 
# practical implementation of a recommendation system leveraging deep learning 
# techniques, providing insights into how these components work together to make
# predictions based on user-movie interactions.

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, Flatten
from keras.models import Model

def load_and_preprocess_data(csv_file_path):
    df = pd.read_csv(csv_file_path)

    user_mapping = {id: i for i, id in enumerate(df['User_ID'].unique())}
    movie_mapping = {id: i for i, id in enumerate(df['Movie_ID'].unique())}

    df['User_ID'] = df['User_ID'].map(user_mapping)
    df['Movie_ID'] = df['Movie_ID'].map(movie_mapping)

    return df, user_mapping, movie_mapping

def build_hybrid_model(user_vocab_size, movie_vocab_size):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    user_embedding = Embedding(user_vocab_size, 50)(user_input)
    movie_embedding = Embedding(movie_vocab_size, 50)(movie_input)

    rnn_output = LSTM(50)(user_embedding)

    attention_output = Attention()([tf.expand_dims(rnn_output, axis=1), movie_embedding])
    attention_output = Flatten()(attention_output)

    merged_output = Concatenate()([rnn_output, attention_output])
    fc1 = Dense(100, activation='relu')(merged_output)
    output = Dense(1)(fc1)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def train_and_evaluate_model(model, train_data, test_data, epochs=10):
    model.fit([train_data['User_ID'], train_data['Movie_ID']], train_data['Rating'],
              epochs=epochs, batch_size=32, validation_split=0.2)

    test_loss = model.evaluate([test_data['User_ID'], test_data['Movie_ID']], test_data['Rating'])
    print(f'Test Loss: {test_loss}')

    predictions = model.predict([test_data['User_ID'], test_data['Movie_ID']])
    result_df = pd.DataFrame({'User_ID': test_data['User_ID'], 'Movie_ID': test_data['Movie_ID'],
                              'Actual_Rating': test_data['Rating'], 'Predicted_Rating': predictions.flatten()})
    print(result_df)

if __name__ == "__main__":
    csv_file_path = "/Users/igormol/Desktop/recsys-light.csv"
    df, user_mapping, movie_mapping = load_and_preprocess_data(csv_file_path)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    user_vocab_size = len(user_mapping) + 1
    movie_vocab_size = len(movie_mapping) + 1

    hybrid_model = build_hybrid_model(user_vocab_size, movie_vocab_size)
    train_and_evaluate_model(hybrid_model, train_df, test_df, epochs=10)

