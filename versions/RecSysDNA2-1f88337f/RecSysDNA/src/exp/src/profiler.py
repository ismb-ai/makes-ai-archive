import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, concatenate
from keras.utils import to_categorical

def profiler_load_data(data_path):
    return pd.read_csv(data_path)

def profiler_encode_categorical(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df, label_encoders

def profiler_prepare_features_and_target(df):
    X = df.drop(['product_id', 'timestamp', 'user_id'], axis=1)
    y = df['product_id']
    return X, y

def profiler_standardize_numerical(X):
    scaler = StandardScaler()
    X[['temperature']] = scaler.fit_transform(X[['temperature']])
    return X

def profiler_split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def profiler_convert_target_to_categorical(y_train, y_test):
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return y_train, y_test

def profiler_build_model(X, label_encoders, y_train):
    input_layers = []
    embeddings = []
    for column in X.columns:
        input_layer = Input(shape=(1,))
        input_layers.append(input_layer)
        if column in label_encoders:
            vocab_size = df[column].nunique()
            embed_dim = min(50, vocab_size // 2)
            embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)(input_layer)
            embeddings.append(Flatten()(embedding))
        else:
            embeddings.append(input_layer)

    merged = concatenate(embeddings)
    dense1 = Dense(128, activation='relu')(merged)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(y_train.shape[1], activation='softmax')(dense2)

    model = Model(inputs=input_layers, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, input_layers

def profiler_prepare_input_data(X, X_train, X_test):
    input_data_train = [X_train[column].values for column in X.columns]
    input_data_test = [X_test[column].values for column in X.columns]
    return input_data_train, input_data_test

def profiler_train_model(model, input_data_train, y_train):
    model.fit(input_data_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

def profiler_evaluate_model(model, input_data_test, y_test):
    loss, accuracy = model.evaluate(input_data_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

def profiler_determine_typical_profiles(df, label_encoders):
    product_profiles = {}
    for product_id in df['product_id'].unique():
        product_data = df[df['product_id'] == product_id]
        profile_means = product_data.drop(['product_id', 'timestamp', 'user_id'], axis=1).mean()
        product_profiles[product_id] = profile_means

    for product_id, profile in product_profiles.items():
        print(f'Product ID: {product_id}')
        for column, value in profile.items():
            if column in label_encoders:
                value = label_encoders[column].inverse_transform([int(value)])[0]
            print(f'  {column}: {value}')
        print()

def main():
    data_path = '/Users/igormol/large_contextual_data.csv'
    df = profiler_load_data(data_path)
    df, label_encoders = profiler_encode_categorical(df)
    X, y = profiler_prepare_features_and_target(df)
    X = profiler_standardize_numerical(X)
    X_train, X_test, y_train, y_test = profiler_split_data(X, y)
    y_train, y_test = profiler_convert_target_to_categorical(y_train, y_test)
    model, input_layers = profiler_build_model(X, label_encoders, y_train)
    input_data_train, input_data_test = profiler_prepare_input_data(X, X_train, X_test)
    profiler_train_model(model, input_data_train, y_train)
    profiler_evaluate_model(model, input_data_test, y_test)
    profiler_determine_typical_profiles(df, label_encoders)

if __name__ == "__main__":
    main()
