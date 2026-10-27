import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import xgboost as xgb

def load_data(filepath):
    # Loads data from a CSV file located at the specified filepath
    return pd.read_csv(filepath)

def encode_categorical_variables(df, columns):
    # Initializes an empty dictionary to store label encoders
    label_encoders = {}

    # Iterates through each specified column
    for column in columns:
        # Initializes a label encoder for the current column
        label_encoders[column] = LabelEncoder()

        # Transforms the categorical values in the column to numerical values
        df[column] = label_encoders[column].fit_transform(df[column])

    # Returns the dataframe with encoded columns and the dictionary of label encoders
    return df, label_encoders

def normalize_numerical_variables(df, columns):
    # Initializes a MinMaxScaler for normalizing numerical variables
    scaler = MinMaxScaler()

    # Applies the scaler to the specified columns in the dataframe
    df[columns] = scaler.fit_transform(df[columns])

    # Returns the dataframe with normalized columns
    return df

def aggregate_duplicates(df, groupby_columns, agg_dict):
    # Groups the dataframe by the specified columns and aggregates using the provided dictionary
    return df.groupby(groupby_columns).agg(agg_dict).reset_index()

def create_interaction_matrix(df, index_col, columns_col, values_col):
    # Creates a pivot table (interaction matrix) with specified index, columns, and values, filling NaNs with 0
    interaction_matrix = df.pivot(index=index_col, columns=columns_col, values=values_col).fillna(0)

    # Converts the interaction matrix to a compressed sparse row matrix
    user_item_matrix = csr_matrix(interaction_matrix.values)

    # Returns the interaction matrix and the sparse matrix
    return interaction_matrix, user_item_matrix

def define_autoencoder(input_dim, encoding_dim):
    # Define a sequential model using a dictionary
    autoencoder = Sequential()

    # Adding the input layer (shape should be specified for the first layer)
    # The Input layer is implicitly created when the first Dense layer is added in Sequential
    autoencoder.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))

    # Adding the encoder layer with the specified encoding dimension and ReLU activation
    # This acts as the encoded representation
    # autoencoder.add(Dense(encoding_dim, activation="relu"))

    # Adding the decoder layer with the same dimension as the input and sigmoid activation
    # This reconstructs the input from the encoded representation
    autoencoder.add(Dense(input_dim, activation="sigmoid"))

    # Compile the model with Adam optimizer and binary crossentropy loss
    autoencoder.compile(optimizer=Adam(), loss=BinaryCrossentropy())

    # Return the constructed autoencoder model
    return autoencoder

def train_autoencoder(autoencoder, interaction_matrix, epochs, batch_size):
    # Fit the autoencoder model on the interaction matrix
    autoencoder.fit(
        interaction_matrix.values,  # Input data: interaction matrix values
        interaction_matrix.values,  # Output data: same as input for autoencoder
        epochs=epochs,              # Number of training epochs
        batch_size=batch_size,      # Size of each batch for gradient descent
        shuffle=True,               # Shuffle the data before each epoch
        validation_split=0.2        # Use 20% of the data for validation
    )

def get_encoded_representation(autoencoder, interaction_matrix):
    # Create a model that maps inputs to the output of the encoder layer
    encoder_model = Model(
        inputs=autoencoder.input,                   # Input to the autoencoder
        outputs=autoencoder.get_layer(index=1).output  # Output of the second layer (encoder output)
    )
    # Use the encoder model to get the encoded representation of the interaction matrix
    return encoder_model.predict(interaction_matrix.values)

def prepare_xgboost_data(train_df, test_df, target_column):
    # Splitting train data into features and target
    X_train = train_df.drop(columns=[target_column])  # Dropping the target column from training data to get features
    y_train = train_df[target_column]  # Extracting the target column as the target variable for training data

    # Splitting test data into features and target
    X_test = test_df.drop(columns=[target_column])  # Dropping the target column from testing data to get features
    y_test = test_df[target_column]  # Extracting the target column as the target variable for testing data

    # Creating DMatrix for XGBoost from training and testing datasets
    train_matrix = xgb.DMatrix(X_train, label=y_train)  # Converting training data to DMatrix
    test_matrix = xgb.DMatrix(X_test, label=y_test)  # Converting testing data to DMatrix

    return train_matrix, test_matrix  # Returning the DMatrix for training and testing data

def train_xgboost_model(dtrain, dtest, params, num_boost_round, early_stopping_rounds):
    # Trains an XGBoost model.
    # Parameters:
    # - dtrain: DMatrix for training data.
    # - dtest: DMatrix for validation data.
    # - params: Dictionary of parameters for XGBoost.
    # - num_boost_round: Number of boosting rounds.
    # - early_stopping_rounds: Rounds of early stopping.
    # Returns:
    # - Trained XGBoost model.

    # Train the XGBoost model using provided parameters, training data, and validation data
    return xgb.train(
        params=params,                # Parameters for the XGBoost model
        dtrain=dtrain,                # DMatrix containing the training data
        num_boost_round=num_boost_round,  # Number of boosting rounds
        evals=[(dtest, 'test')],      # List of evaluation datasets and their names
        early_stopping_rounds=early_stopping_rounds,  # Early stopping rounds
        verbose_eval=False            # Disable verbose evaluation
    )

def get_xgboost_predictions(bst, df_aggregated, target_column):
    # Generates predictions using the trained XGBoost model.
    # Parameters:
    # - bst: Trained XGBoost model.
    # - df_aggregated: DataFrame containing aggregated data.
    # - target_column: Name of the target column in the DataFrame.
    # Returns:
    # - Array of predictions.

    # Create a DMatrix from the aggregated DataFrame, excluding the target column
    df_dmatrix = xgb.DMatrix(df_aggregated.drop(columns=[target_column]))

    # Generate predictions using the trained XGBoost model
    return bst.predict(df_dmatrix)

def reshape_xgboost_predictions(xgb_predictions, df_aggregated, interaction_matrix):
    # Initialize an empty matrix with the same shape as the interaction matrix
    xgb_predictions_matrix = np.zeros(interaction_matrix.shape)

    # Iterate over the predictions and corresponding client and product IDs
    for i, (client_id, product_id) in enumerate(zip(df_aggregated['Client_ID'], df_aggregated['Product_ID'])):
        # Find the row index in the interaction matrix for the current client ID
        client_idx = interaction_matrix.index.get_loc(client_id)

        # Find the column index in the interaction matrix for the current product ID
        product_idx = interaction_matrix.columns.get_loc(product_id)

        # Assign the XGBoost prediction to the appropriate position in the prediction matrix
        xgb_predictions_matrix[client_idx, product_idx] = xgb_predictions[i]

    # Return the reshaped predictions matrix
    return xgb_predictions_matrix

def combine_predictions(autoencoder, interaction_matrix, xgb_predictions_matrix):
    # Predict the interactions using the autoencoder model
    autoencoder_predictions = autoencoder.predict(interaction_matrix.values)

    # Combine the autoencoder predictions and XGBoost predictions by averaging them
    combined_predictions = (autoencoder_predictions + xgb_predictions_matrix) / 2

    # Convert the combined predictions to a DataFrame with the same index and columns as the interaction matrix
    return pd.DataFrame(combined_predictions, index=interaction_matrix.index, columns=interaction_matrix.columns)

def get_top_k_recommendations(combined_df, interaction_matrix, user_id, k=5):
    # Find the row index in the interaction matrix corresponding to the specified user ID
    user_row_number = interaction_matrix.index.get_loc(user_id)

    # Sort the predictions for this user in descending order
    sorted_user_predictions = combined_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the top k product IDs and their corresponding predicted ratings
    top_k_products = sorted_user_predictions.head(k)
    top_k_product_ids = top_k_products.index.tolist()
    top_k_ratings = top_k_products.values.tolist()

    # Return the top k product IDs and their ratings as lists
    return top_k_product_ids, top_k_ratings

def create_recommendations_table(interaction_matrix, label_encoders, combined_df, k=5):
    # Generate a list of recommendations for all users in the interaction matrix
    recommendations_list = []

    for user_id in interaction_matrix.index:
        # Get the client name for the current user_id
        client_name = decode_user_id(user_id, label_encoders)

        # Get the top k product recommendations and their predicted ratings
        recommended_product_ids, predicted_ratings = get_top_k_recommendations(combined_df, interaction_matrix, user_id, k)

        # Decode the recommended product ids to product names
        recommended_product_names = decode_product_ids(recommended_product_ids, label_encoders)

        # Append the recommendations for the current user to the list
        recommendations_list.extend(create_user_recommendations(client_name, recommended_product_ids, recommended_product_names, predicted_ratings))

    # Convert the list of recommendations to a DataFrame and return
    return pd.DataFrame(recommendations_list)

def decode_user_id(user_id, label_encoders):
    # Decode the user_id to get the client name
    return label_encoders['Client_ID'].inverse_transform([user_id])[0]

def decode_product_ids(product_ids, label_encoders):
    # Decode the list of product_ids to get the product names
    return label_encoders['Product_ID'].inverse_transform(product_ids)

def create_user_recommendations(client_name, product_ids, product_names, ratings):
    # Create a list of recommendations for a single user
    return [
        {'Client_ID': client_name, 'Product_ID': product_name, 'Predicted_Rating': rating}
        for product_id, product_name, rating in zip(product_ids, product_names, ratings)
    ]

def recall_at_k(actual, predicted, k):
    if len(actual) == 0:
        return np.nan  # If there are no actual purchases, return NaN
    act_set = set(actual)
    pred_set = set(predicted[:k])
    return len(act_set & pred_set) / float(len(act_set))

def get_actual_purchases(test_df, user_id):
    # Retrieves the actual purchases of a user from the test dataset.
    # Parameters:
    # - test_df: DataFrame containing test data.
    # - user_id: ID of the user.
    # Returns:
    # - List of actual purchases made by the user.
    return test_df[test_df['Client_ID'] == user_id]['Product_ID'].tolist()

def calculate_recall_scores(interaction_matrix, test_df, combined_df, k):
    # Calculate recall scores for each user and store them in a list and dictionary.

    # Initialize an empty list to store recall scores and an empty dictionary for user-specific recall scores
    recall_scores = []
    user_recalls = {}

    # Iterate over each user in the interaction matrix
    for user_id in interaction_matrix.index:
        # Get the actual purchases made by the user from the test DataFrame
        actual_purchases = get_actual_purchases(test_df, user_id)

        # Get the top-k recommended items for the user
        predicted_purchases, _ = get_top_k_recommendations(combined_df, interaction_matrix, user_id, k)

        # Calculate the recall at k for the user
        recall = recall_at_k(actual_purchases, predicted_purchases, k)

        # Append the recall score to the list and dictionary if it is not NaN
        if not np.isnan(recall):
            recall_scores.append(recall)
            user_recalls[user_id] = recall

    return recall_scores, user_recalls

def calculate_mean_recall(recall_scores):
    # Calculate the mean of the recall scores provided
    return np.mean(recall_scores)

def calculate_recall_at_k(interaction_matrix, test_df, combined_df, k=5):
    # Calculate recall scores for each user
    recall_scores, user_recalls = calculate_recall_scores(interaction_matrix, test_df, combined_df, k)
    # Calculate the mean recall score from the individual recall scores
    mean_recall = calculate_mean_recall(recall_scores)
    # Return the mean recall score and the dictionary of individual user recalls
    return mean_recall, user_recalls

def print_recommendations_and_recall(recommendations_df, user_recalls, interaction_matrix, label_encoders, k=5):
    # Iterate over each user in the interaction matrix
    for user_id in interaction_matrix.index:
        # Get the client's name using the label encoder for 'Client_ID'
        client_name = label_encoders['Client_ID'].inverse_transform([user_id])[0]
        # Check if the user has a recall score calculated
        if user_id in user_recalls:
            # Print the recall score at k for the client
            print(f'Recall at {k} for client {client_name}: {user_recalls[user_id]}')

        # Get the recommendations for the current user
        user_recommendations = recommendations_df[recommendations_df['Client_ID'] == client_name]
        # Sort the recommendations by predicted rating in descending order
        user_recommendations = user_recommendations.sort_values(by='Predicted_Rating', ascending=False)
        # Print the top recommendations for the client
        print(f"Top recommendations for client {client_name}:")
        print(user_recommendations)
        # Print a new line for better readability
        print("\n")

def main():
    # Define the filepath to the data
    filepath = '/Users/igormol/Desktop/analytics/src/exp/recsysDualCore/sample/Vaz.csv'

    # Load the data from the specified filepath
    df = load_data(filepath)

    # Encode categorical variables for specified columns
    df, label_encoders = encode_categorical_variables(df, ['Client_ID', 'Product_ID', 'Product_Name', 'Brand_Name'])

    # Normalize the specified numerical variables
    df = normalize_numerical_variables(df, ['Monetary_Value', 'Discount'])

    # Aggregate duplicate entries by summing 'Monetary_Value' and averaging 'Discount'
    df_aggregated = aggregate_duplicates(df, ['Client_ID', 'Product_ID'], {
        'Monetary_Value': 'sum',
        'Discount': 'mean',
        'Product_Name': 'first',
        'Brand_Name': 'first'
    })

    # Create interaction matrices from the aggregated DataFrame
    interaction_matrix, user_item_matrix = create_interaction_matrix(df_aggregated, 'Client_ID', 'Product_ID', 'Monetary_Value')

    # Split the aggregated DataFrame into training and test sets
    train_df, test_df = train_test_split(df_aggregated, test_size=0.2, random_state=42)

    # Define and train an autoencoder model
    autoencoder = define_autoencoder(interaction_matrix.shape[1], 50)
    train_autoencoder(autoencoder, interaction_matrix, epochs=50, batch_size=256)

    # Obtain the encoded (latent) representation of users from the autoencoder
    user_latent_factors = get_encoded_representation(autoencoder, interaction_matrix)

    # Prepare data for XGBoost training and testing
    dtrain, dtest = prepare_xgboost_data(train_df, test_df, 'Monetary_Value')

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,        # Increased max depth for the trees
        'eta': 0.01,            # Decreased learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    num_boost_round = 100000   # Increased number of boosting rounds significantly
    early_stopping_rounds = 1000

    # Train the XGBoost model
    bst = train_xgboost_model(dtrain, dtest, params, num_boost_round, early_stopping_rounds)

    # Generate predictions using the trained XGBoost model
    xgb_predictions = get_xgboost_predictions(bst, df_aggregated, 'Monetary_Value')

    # Reshape XGBoost predictions to match the interaction matrix
    xgb_predictions_matrix = reshape_xgboost_predictions(xgb_predictions, df_aggregated, interaction_matrix)

    # Combine autoencoder and XGBoost predictions
    combined_df = combine_predictions(autoencoder, interaction_matrix, xgb_predictions_matrix)

    # Create a recommendations table based on the combined predictions
    recommendations_df = create_recommendations_table(interaction_matrix, label_encoders, combined_df)

    # Set the number of top recommendations to consider
    k = 5

    # Calculate the average recall at k and recall scores for each user
    average_recall_at_k, user_recalls = calculate_recall_at_k(interaction_matrix, test_df, combined_df, k)

    # Print the average recall at k
    print(f'Average Recall at {k}: {average_recall_at_k}')

    # Print recommendations and recall scores for users
    print_recommendations_and_recall(recommendations_df, user_recalls, interaction_matrix, label_encoders)

if __name__ == "__main__":
    main()
