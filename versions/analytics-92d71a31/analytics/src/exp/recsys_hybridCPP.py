
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import xgboost as xgb

def load_data(filepath):
    return pd.read_csv(filepath)

def encode_categorical_variables(df, columns):
    label_encoders = {}
    for column in columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders

def normalize_numerical_variables(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def aggregate_duplicates(df, groupby_columns, agg_dict):
    return df.groupby(groupby_columns).agg(agg_dict).reset_index()

def create_interaction_matrix(df, index_col, columns_col, values_col):
    interaction_matrix = df.pivot(index=index_col, columns=columns_col, values=values_col).fillna(0)
    user_item_matrix = csr_matrix(interaction_matrix.values)
    return interaction_matrix, user_item_matrix

def define_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def train_autoencoder(autoencoder, interaction_matrix, epochs, batch_size):
    autoencoder.fit(interaction_matrix.values, interaction_matrix.values,
                    epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)

def get_encoded_representation(autoencoder, interaction_matrix):
    encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
    return encoder_model.predict(interaction_matrix.values)

def prepare_xgboost_data(train_df, test_df, target_column):
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    return xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_test, label=y_test)

def train_xgboost_model(dtrain, dtest, params, num_boost_round, early_stopping_rounds):
    return xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, 'test')],
                     early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

def get_xgboost_predictions(bst, df_aggregated, target_column):
    df_dmatrix = xgb.DMatrix(df_aggregated.drop(columns=[target_column]))
    return bst.predict(df_dmatrix)

def reshape_xgboost_predictions(xgb_predictions, df_aggregated, interaction_matrix):
    xgb_predictions_matrix = np.zeros(interaction_matrix.shape)
    for i, (client_id, product_id) in enumerate(zip(df_aggregated['Client_ID'], df_aggregated['Product_ID'])):
        client_idx = interaction_matrix.index.get_loc(client_id)
        product_idx = interaction_matrix.columns.get_loc(product_id)
        xgb_predictions_matrix[client_idx, product_idx] = xgb_predictions[i]
    return xgb_predictions_matrix

def combine_predictions(autoencoder, interaction_matrix, xgb_predictions_matrix):
    autoencoder_predictions = autoencoder.predict(interaction_matrix.values)
    combined_predictions = (autoencoder_predictions + xgb_predictions_matrix) / 2
    return pd.DataFrame(combined_predictions, index=interaction_matrix.index, columns=interaction_matrix.columns)

def get_top_k_recommendations(combined_df, interaction_matrix, user_id, k=5):
    user_row_number = interaction_matrix.index.get_loc(user_id)
    sorted_user_predictions = combined_df.iloc[user_row_number].sort_values(ascending=False)
    top_k_products = sorted_user_predictions.head(k)
    top_k_product_ids = top_k_products.index.tolist()
    top_k_ratings = top_k_products.values.tolist()
    return top_k_product_ids, top_k_ratings

def create_recommendations_table(interaction_matrix, label_encoders, combined_df, k=5):
    recommendations_list = []

    for user_id in interaction_matrix.index:
        client_name = label_encoders['Client_ID'].inverse_transform([user_id])[0]
        recommended_product_ids, predicted_ratings = get_top_k_recommendations(combined_df, interaction_matrix, user_id, k)
        recommended_product_names = label_encoders['Product_ID'].inverse_transform(recommended_product_ids)

        for product_id, product_name, rating in zip(recommended_product_ids, recommended_product_names, predicted_ratings):
            recommendations_list.append({
                'Client_ID': client_name,
                'Product_ID': product_name,
                'Predicted_Rating': rating
            })

    return pd.DataFrame(recommendations_list)

def recall_at_k(actual, predicted, k):
    if len(actual) == 0:
        return np.nan  # If there are no actual purchases, return NaN
    act_set = set(actual)
    pred_set = set(predicted[:k])
    return len(act_set & pred_set) / float(len(act_set))

def calculate_recall_at_k(interaction_matrix, test_df, combined_df, k=5):
    recall_scores = []
    user_recalls = {}

    for user_id in interaction_matrix.index:
        actual_purchases = test_df[test_df['Client_ID'] == user_id]['Product_ID'].tolist()
        predicted_purchases, _ = get_top_k_recommendations(combined_df, interaction_matrix, user_id, k)
        recall = recall_at_k(actual_purchases, predicted_purchases, k)
        if not np.isnan(recall):  # Only append recall scores that are not NaN
            recall_scores.append(recall)
            user_recalls[user_id] = recall

    return np.mean(recall_scores), user_recalls

def print_recommendations_and_recall(recommendations_df, user_recalls, interaction_matrix, label_encoders, k=5):
    for user_id in interaction_matrix.index:
        client_name = label_encoders['Client_ID'].inverse_transform([user_id])[0]
        if user_id in user_recalls:
            print(f'Recall at {k} for client {client_name}: {user_recalls[user_id]}')

        user_recommendations = recommendations_df[recommendations_df['Client_ID'] == client_name]
        user_recommendations = user_recommendations.sort_values(by='Predicted_Rating', ascending=False)
        print(f"Top recommendations for client {client_name}:")
        print(user_recommendations)
        print("\n")

def main():
    filepath = '/Users/igormol/Desktop/analytics/sample/CPP.csv'
    df = load_data(filepath)

    df, label_encoders = encode_categorical_variables(df, ['Client_ID', 'Product_ID', 'Product_Name', 'Brand_Name'])
    df = normalize_numerical_variables(df, ['Monetary_Value', 'Discount'])

    df_aggregated = aggregate_duplicates(df, ['Client_ID', 'Product_ID'], {
        'Monetary_Value': 'sum',
        'Discount': 'mean',
        'Product_Name': 'first',
        'Brand_Name': 'first'
    })

    interaction_matrix, user_item_matrix = create_interaction_matrix(df_aggregated, 'Client_ID', 'Product_ID', 'Monetary_Value')

    train_df, test_df = train_test_split(df_aggregated, test_size=0.2, random_state=42)

    autoencoder = define_autoencoder(interaction_matrix.shape[1], 50)
    train_autoencoder(autoencoder, interaction_matrix, 50, 256)

    user_latent_factors = get_encoded_representation(autoencoder, interaction_matrix)

    dtrain, dtest = prepare_xgboost_data(train_df, test_df, 'Monetary_Value')
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,        # Increase the maximum depth of the trees
        'eta': 0.01,            # Decrease the learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    num_boost_round = 100000   # Increase the number of boosting rounds significantly
    early_stopping_rounds = 1000

    bst = train_xgboost_model(dtrain, dtest, params, num_boost_round, early_stopping_rounds)

    xgb_predictions = get_xgboost_predictions(bst, df_aggregated, 'Monetary_Value')
    xgb_predictions_matrix = reshape_xgboost_predictions(xgb_predictions, df_aggregated, interaction_matrix)

    combined_df = combine_predictions(autoencoder, interaction_matrix, xgb_predictions_matrix)

    recommendations_df = create_recommendations_table(interaction_matrix, label_encoders, combined_df)

    k = 5  # Define k here
    average_recall_at_k, user_recalls = calculate_recall_at_k(interaction_matrix, test_df, combined_df, k)
    print(f'Average Recall at {k}: {average_recall_at_k}')

    print_recommendations_and_recall(recommendations_df, user_recalls, interaction_matrix, label_encoders, k)

if __name__ == "__main__":
    main()
