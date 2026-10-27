# profiler.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the program that profiles the characteristics
# of users U who buys a product P according to the contextual variables associa-
# ted to the pair (U, P). Our implementation utilizes a deep learning approach
# for multi-classification. The neural network is trained in the contextual
# variables and targets the products' identifiers.

# In sec. 1, we define functions for data manipulation. In sec. 2, we build the
# neural network structure. In sec. 3, one finds the core routines for model's
# training and evaluation. In sec. 4, we define the post-training functions to
# extract results from the model.

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from keras.models import Model, Sequential
from keras.optimizers import Adam
import streamlit as st

# Section 1: Pre-training phase II. Data manipulation.

def profiler_encode_categorical(df):
    # Encodes categorical columns in a DataFrame using LabelEncoder.
    # It takes the following parameter:
    # - 'df': The DataFrame containing categorical columns to be encoded.

    # Dictionary to store label encoders for each categorical column
    label_encoders = {}

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the column's data type is object (categorical)
        if df[column].dtype == object:
            # Initialize a LabelEncoder for the column
            le = LabelEncoder()

            # Encode the column values using LabelEncoder
            df[column] = le.fit_transform(df[column])

            # Store the LabelEncoder in the dictionary with column name as key
            label_encoders[column] = le

    # Return the encoded DataFrame and the dictionary of label encoders
    return df, label_encoders

def profiler_prepare_features_and_target(df):
    # Prepares features (X) and target variable (y) from the given DataFrame.
    # It drops specified columns from the DataFrame to create the feature matrix (X),
    # and extracts the target variable (y) from the 'product_id' column.
    # It takes the following parameter:
    # - 'df': The DataFrame containing the dataset.

    # Drop specified columns ('product_id', 'timestamp', 'user_id') to create the feature matrix (X)
    X = df.drop(['product_id', 'timestamp', 'user_id'], axis=1)

    # Extract the target variable (y) from the 'product_id' column
    y = df['product_id']

    # Return the feature matrix (X) and the target variable (y)
    return X, y

def profiler_standardize_numerical(X):
    # Standardizes numerical features in the DataFrame using StandardScaler.
    # It takes the following parameter:
    # - 'X': The DataFrame containing numerical features to be standardized.

    # Initialize a StandardScaler object
    scaler = StandardScaler()

    # Standardize the 'temperature' column using the StandardScaler
    # The double square brackets [['temperature']] are used to pass a DataFrame with a single column to scaler.fit_transform
    # and assign the standardized values back to the 'temperature' column
    X[['temperature']] = scaler.fit_transform(X[['temperature']])

    # Return the DataFrame with standardized 'temperature' column
    return X

def profiler_split_data(X, y):
    # Splits the data into training and testing sets.
    # It takes the following parameters:
    # - 'X': The feature matrix.
    # - 'y': The target variable.
    # It returns four arrays: X_train, X_test, y_train, y_test.

    # Split the data into training and testing sets with a test size of 20% and a fixed random state for reproducibility.
    return train_test_split(X, y, test_size=0.2, random_state=42)

def profiler_convert_target_to_categorical(y_train, y_test):
    # Converts target variables to categorical format.
    # It takes the following parameters:
    # - 'y_train': The training set target variable.
    # - 'y_test': The testing set target variable.
    # It returns the categorical representations of y_train and y_test.

    # Convert y_train and y_test to categorical format using one-hot encoding.
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Return the categorical representations of y_train and y_test.
    return y_train, y_test

# Section 2: Pre-training Phase II. Create neural network architecture.

def create_input_layer(column):
    # Creates an input layer for a neural network.
    # It takes the following parameter:
    # - 'column': The name of the input column.
    # It returns an Input layer with the specified column name.

    # Create an Input layer with a shape of (1,) for the specified column.
    return Input(shape=(1,), name=f"{column}_input")

def create_embedding_layer(input_layer, vocab_size):
    # Creates an embedding layer for a neural network.
    # It takes the following parameters:
    # - 'input_layer': The input layer.
    # - 'vocab_size': The size of the vocabulary (number of unique categories).
    # It returns the embedding layer.

    # Calculate the embedding dimension based on the vocabulary size.
    embed_dim = min(50, vocab_size // 2)

    # Create an Embedding layer with input dimension equal to the vocabulary size
    # and output dimension equal to the embedding dimension.
    embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)(input_layer)

    # Flatten the embedding layer to connect it with a Dense layer.
    return Flatten()(embedding)

def build_embeddings_and_inputs(X, label_encoders):
    # Builds embedding layers and input layers for each feature in the dataset.
    # It takes the following parameters:
    # - 'X': The DataFrame containing the dataset.
    # - 'label_encoders': A dictionary containing label encoders for categorical features.
    # It returns two lists: input_layers and embeddings.

    # Initialize lists to store input layers and embeddings
    input_layers = []
    embeddings = []

    # Iterate over each column in the DataFrame
    for column in X.columns:
        # Create an input layer for the column
        input_layer = create_input_layer(column)

        # Append the input layer to the list of input layers
        input_layers.append(input_layer)

        # Check if the column is categorical and has a corresponding label encoder
        if column in label_encoders:
            # Calculate the vocabulary size for the column
            vocab_size = X[column].nunique()

            # Create an embedding layer for the categorical column
            embedding = create_embedding_layer(input_layer, vocab_size)

            # Append the embedding layer to the list of embeddings
            embeddings.append(embedding)
        else:
            # If the column is not categorical, use the input layer as the embedding
            embeddings.append(input_layer)

    # Return the lists of input layers and embeddings
    return input_layers, embeddings

def build_sequential_model(merged_input_shape, output_shape):
    # Builds a sequential neural network model using Keras.
    # It takes the following parameters:
    # - 'merged_input_shape': The shape of the input data after merging all input layers.
    # - 'output_shape': The number of classes for the output layer.
    # It returns the constructed sequential model.

    # Initialize a sequential model
    model = Sequential()

    # Add a dense layer with 128 units and ReLU activation function as the first hidden layer.
    # The input shape is determined by 'merged_input_shape'.
    model.add(Dense(128, activation='relu', input_shape=merged_input_shape))

    # Add a second dense layer with 64 units and ReLU activation function.
    model.add(Dense(64, activation='relu'))

    # Add the output layer with 'output_shape' units and softmax activation function.
    # Softmax activation is used for multi-class classification.
    model.add(Dense(output_shape, activation='softmax'))

    # Return the constructed sequential model
    return model

def profiler_build_model(X, label_encoders, y_train):
    # Builds a functional model using input layers, embeddings, and a sequential model.
    # It takes the following parameters:
    # - 'X': The DataFrame containing the dataset.
    # - 'label_encoders': A dictionary containing label encoders for categorical features.
    # - 'y_train': The training set target variable (after one-hot encoding).
    # It returns the constructed functional model and input layers.

    # Build input layers and embeddings based on the dataset and label encoders
    input_layers, embeddings = build_embeddings_and_inputs(X, label_encoders)

    # Concatenate the embeddings to merge them into a single tensor
    merged = concatenate(embeddings)

    # Create a dummy model to determine the shape after merge
    temp_model = Model(inputs=input_layers, outputs=merged)
    merged_input_shape = (temp_model.output_shape[1],)

    # Build the sequential model using the merged input shape and the number of output classes
    sequential_model = build_sequential_model(merged_input_shape, y_train.shape[1])

    # Create the final functional model by combining input layers and the sequential model
    final_model = Model(inputs=input_layers, outputs=sequential_model(merged))

    # Compile the final model with Adam optimizer, categorical crossentropy loss, and accuracy metric
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed functional model and input layers
    return final_model, input_layers

def profiler_prepare_input_data(X, X_train, X_test):
    # Prepares input data for training and testing the model.
    # It takes the following parameters:
    # - 'X': The DataFrame containing the dataset.
    # - 'X_train': The training set features.
    # - 'X_test': The testing set features.
    # It returns input data for training and testing.

    # Extract the values of each feature column for the training and testing sets
    input_data_train = [X_train[column].values for column in X.columns]
    input_data_test = [X_test[column].values for column in X.columns]

    # Return input data for training and testing
    return input_data_train, input_data_test

# Section 3: Training phase. Core functions for model training and evaluation.

def profiler_train_model(model, input_data_train, y_train):
    # Trains the model using the input data and target variable.
    # It takes the following parameters:
    # - 'model': The model to be trained.
    # - 'input_data_train': Input data for training the model.
    # - 'y_train': The training set target variable.
    # It trains the model for 5 epochs with a batch size of 32 and a validation split of 20%.

    # Train the model using the input data and target variable
    model.fit(input_data_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

def profiler_evaluate_model(model, input_data_test, y_test):
    # Evaluates the trained model using the test data.
    # It takes the following parameters:
    # - 'model': The trained model to be evaluated.
    # - 'input_data_test': Input data for testing the model.
    # - 'y_test': The true labels for the testing set.
    # It prints and returns the test accuracy of the model.

    # Evaluate the model using the test data
    loss, accuracy = model.evaluate(input_data_test, y_test)

    # Print the test accuracy
    st.write(f'Test Accuracy: {accuracy:.2f}')

    # Return the test accuracy
    return accuracy

# Section 4: Post-training phase. Routines to extract results from the model.

def extract_product_data(df, product_id):
    # Extracts data corresponding to a specific product ID from the DataFrame.
    # It takes the following parameters:
    # - 'df': The DataFrame containing the dataset.
    # - 'product_id': The ID of the product to extract data for.
    # It returns a subset of the DataFrame containing data for the specified product ID.

    # Filter the DataFrame to extract data for the specified product ID
    return df[df['product_id'] == product_id]

def calculate_profile_means(product_data):
    # Calculates the mean values of profile features for a product.
    # It takes the following parameter:
    # - 'product_data': The DataFrame containing data for a specific product.
    # It calculates the mean values of profile features (excluding 'product_id', 'timestamp', 'user_id') and returns them.

    # Drop 'product_id', 'timestamp', 'user_id' columns and calculate the mean values of the remaining columns
    return product_data.drop(['product_id', 'timestamp', 'user_id'], axis=1).mean()

def calculate_product_profiles(df):
    # Calculates profile means for each product in the dataset.
    # It takes the following parameter:
    # - 'df': The DataFrame containing the dataset.
    # It returns a dictionary where keys are product IDs and values are profile means for each product.

    # Initialize an empty dictionary to store product profiles
    product_profiles = {}

    # Iterate over unique product IDs in the DataFrame
    for product_id in df['product_id'].unique():
        # Extract data for the current product ID
        product_data = extract_product_data(df, product_id)

        # Calculate profile means for the product
        profile_means = calculate_profile_means(product_data)

        # Store the profile means in the dictionary with the product ID as key
        product_profiles[product_id] = profile_means

    # Return the dictionary containing product profiles
    return product_profiles

def decode_value(column, value, label_encoders):
    # Decodes a categorical value using the corresponding label encoder.
    # It takes the following parameters:
    # - 'column': The name of the column.
    # - 'value': The encoded value to decode.
    # - 'label_encoders': A dictionary containing label encoders for categorical columns.
    # It returns the decoded value.

    # Check if the column has a corresponding label encoder
    if column in label_encoders:
        # Use the label encoder to inverse transform the encoded value
        return label_encoders[column].inverse_transform([int(value)])[0]

    # If the column is not categorical, return the original value
    return value

def format_profile_entry(column, value):
    # Formats a profile entry as HTML for display.
    # It takes the following parameters:
    # - 'column': The name of the column.
    # - 'value': The value of the column.
    # It returns the formatted profile entry as HTML.

    # Format the profile entry as a styled div element with background color, padding, margin, and border-radius
    return f"<div style='background-color: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; display: inline-block;'>{column}: {value}</div>"

def display_profile(product_id, profile, label_encoders):
    # Displays the profile of a product in an expander.
    # It takes the following parameters:
    # - 'product_id': The ID of the product.
    # - 'profile': The profile of the product containing column-value pairs.
    # - 'label_encoders': A dictionary containing label encoders for categorical columns.

    # Use an expander to display the profile with the product ID as the title
    with st.expander(f"Product ID: {product_id}", expanded=True):
        # Iterate over column-value pairs in the profile
        for column, value in profile.items():
            # Decode the value if the column is categorical
            decoded_value = decode_value(column, value, label_encoders)

            # Format the profile entry and display it as HTML
            st.markdown(format_profile_entry(column, decoded_value), unsafe_allow_html=True)

def display_product_profiles(product_profiles, label_encoders):
    # Displays the profiles of multiple products.
    # It takes the following parameters:
    # - 'product_profiles': A dictionary where keys are product IDs and values are profiles of products.
    # - 'label_encoders': A dictionary containing label encoders for categorical columns.

    # Iterate over product IDs and profiles in the product_profiles dictionary
    for product_id, profile in product_profiles.items():
        # Display the profile of each product
        display_profile(product_id, profile, label_encoders)

def profiler_determine_typical_profiles(df, label_encoders):
    # Determines typical profiles of products in the dataset and displays them.
    # It takes the following parameters:
    # - 'df': The DataFrame containing the dataset.
    # - 'label_encoders': A dictionary containing label encoders for categorical columns.

    # Calculate product profiles
    product_profiles = calculate_product_profiles(df)

    # Display product profiles
    display_product_profiles(product_profiles, label_encoders)
