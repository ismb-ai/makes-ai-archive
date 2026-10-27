import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tabulate import tabulate

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 7'])
    df = df.dropna()
    return df

def encode_features(df):
    # Initialize an empty dictionary to store the label encoders for each column
    label_encoders = {}

    # Iterate over the columns that need to be label encoded
    for column in ['Client_ID', 'Product_ID', 'Product_Name', 'Brand_Name']:
        # Create a LabelEncoder object for the current column
        le = LabelEncoder()

        # Fit the label encoder to the column data and transform the data to integer labels
        df[column] = le.fit_transform(df[column])

        # Store the label encoder object in the dictionary with the column name as the key
        label_encoders[column] = le

    # Return the transformed DataFrame and the dictionary of label encoders
    return df, label_encoders

def create_interaction_matrix(df):
    interaction_matrix = df.pivot_table(index='Client_ID', columns='Product_ID', values='Monetary_Value', aggfunc='sum', fill_value=0)
    return interaction_matrix

# Import necessary library
from sklearn.model_selection import train_test_split

def prepare_train_test_data(interaction_matrix):
    # Convert the interaction matrix into a DataFrame suitable for train/test splitting
    # 1. stack(): Stacks the columns into a single column, converting the wide format to a long format
    # 2. reset_index(): Resets the index of the resulting DataFrame, converting the multi-index into columns
    # 3. rename(): Renames the resulting column from 0 to 'Rating'
    interaction_df = interaction_matrix.stack().reset_index().rename(columns={0: 'Rating'})

    # Split the interaction DataFrame into training and testing sets
    # - test_size=0.2 means 20% of the data will be used for testing
    # - random_state=42 ensures the split is reproducible
    train_data, test_data = train_test_split(interaction_df, test_size=0.2, random_state=42)

    # Extract features (Client_ID and Product_ID) and target (Rating) for the training set
    X_train = train_data[['Client_ID', 'Product_ID']]
    y_train = train_data['Rating']

    # Extract features (Client_ID and Product_ID) and target (Rating) for the testing set
    X_test = test_data[['Client_ID', 'Product_ID']]
    y_test = test_data['Rating']

    # Return the training and testing sets
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")

def create_mappings(df, original_df):
    product_name_mapping = dict(zip(df['Product_ID'], original_df['Product_Name']))
    brand_name_mapping = dict(zip(df['Product_ID'], original_df['Brand_Name']))
    return product_name_mapping, brand_name_mapping

def get_recommendations(model, user_id, interaction_matrix, product_name_mapping, brand_name_mapping, num_recommendations=5):
    all_product_ids = interaction_matrix.columns.values
    user_data = pd.DataFrame({'Client_ID': [user_id] * len(all_product_ids), 'Product_ID': all_product_ids})
    predicted_ratings = model.predict(user_data)
    recommendations = pd.DataFrame({'Product_ID': all_product_ids, 'Predicted_Rating': predicted_ratings})
    top_recommendations = recommendations.sort_values(by='Predicted_Rating', ascending=False).head(num_recommendations)
    top_recommendations['Product_Name'] = top_recommendations['Product_ID'].map(product_name_mapping)
    top_recommendations['Brand_Name'] = top_recommendations['Product_ID'].map(brand_name_mapping)
    return top_recommendations

def generate_recommendations(model, interaction_matrix, clients, product_name_mapping, brand_name_mapping):
    recommendations = {client: get_recommendations(model, client, interaction_matrix, product_name_mapping, brand_name_mapping) for client in clients}
    return recommendations

def save_recommendations_to_file(recommendations, file_path):
    output = ""
    for client, recs in recommendations.items():
        client_name = label_encoders['Client_ID'].inverse_transform([client])[0]
        output += f"Top recommendations for client {client_name}:\n"
        table = tabulate(recs[['Product_Name', 'Brand_Name', 'Predicted_Rating']], headers=['Product', 'Brand', 'Predicted Rating'], tablefmt='pretty', showindex=False)
        output += table + "\n\n"
    with open(file_path, "w") as file:
        file.write(output)
    print(f"Recommendations saved to {file_path}")

if __name__ == "__main__":
    file_path = "/Users/igormol/Downloads/Vaz.csv"

    df = load_and_preprocess_data(file_path)
    original_df = df.copy()
    df, label_encoders = encode_features(df)
    interaction_matrix = create_interaction_matrix(df)
    X_train, y_train, X_test, y_test = prepare_train_test_data(interaction_matrix)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    product_name_mapping, brand_name_mapping = create_mappings(df, original_df)
    clients = interaction_matrix.index
    recommendations = generate_recommendations(model, interaction_matrix, clients, product_name_mapping, brand_name_mapping)

    save_recommendations_to_file(recommendations, "recommendations.txt")
