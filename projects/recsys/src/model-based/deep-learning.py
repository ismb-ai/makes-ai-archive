import pandas as pd
import numpy as np

def load_data(file_path, time_window):
    df = pd.read_csv(file_path)
    user_ids = df['user_id'].astype('category').cat.codes.values
    item_ids = df['item_id'].astype('category').cat.codes.values
    ratings = df['rating'].values / 5.0

    # Convert timestamps to datetime format, handling any invalid formats
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with NaT (Not a Timestamp) values
    df = df.dropna(subset=['timestamp'])

    df['timestamp_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    df['relative_time'] = df['timestamp_seconds'] % time_window.total_seconds()
    return user_ids, item_ids, ratings, df


def create_embedding_matrices(num_users, num_items, embedding_dim):
    user_embedding_matrix = np.random.normal(size=(num_users, embedding_dim))
    item_embedding_matrix = np.random.normal(size=(num_items, embedding_dim))
    user_bias = np.zeros(num_users)
    item_bias = np.zeros(num_items)
    global_bias = 0.0
    return user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias

def collaborative_filtering_model(user_ids, item_ids, user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias):
    user_embed = user_embedding_matrix[user_ids]
    item_embed = item_embedding_matrix[item_ids]
    user_bias_term = user_bias[user_ids]
    item_bias_term = item_bias[item_ids]
    user_item_interaction = np.sum(user_embed * item_embed, axis=1)
    output = user_item_interaction + user_bias_term + item_bias_term + global_bias
    return output

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def update_parameters_with_time_window(user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias,
                                      user_ids, item_ids, ratings, learning_rate, timestamps, time_window):
    user_embed = user_embedding_matrix[user_ids]
    item_embed = item_embedding_matrix[item_ids]
    user_bias_term = user_bias[user_ids]
    item_bias_term = item_bias[item_ids]

    # Convert DatetimeProperties to numerical representation of seconds
    timestamp_seconds = timestamps.astype(int) // 10**9  # Convert nanoseconds to seconds

    # Apply modulo operation after converting to numerical representation
    weight = np.exp(-(timestamp_seconds % int(time_window.total_seconds())) / time_window.total_seconds())
    
    error = (ratings - (np.sum(user_embed * item_embed, axis=1) + user_bias_term + item_bias_term + global_bias)) * weight
    user_grad = -2 * np.dot(error, item_embed) / len(error)
    item_grad = -2 * np.dot(error, user_embed) / len(error)
    user_bias_grad = -2 * np.sum(error) / len(error)
    item_bias_grad = -2 * np.sum(error) / len(error)
    global_bias_grad = -2 * np.sum(error) / len(error)

    # Update parameters using gradient descent.
    user_embedding_matrix -= learning_rate * user_grad
    item_embedding_matrix -= learning_rate * item_grad
    user_bias -= learning_rate * user_bias_grad
    item_bias -= learning_rate * item_bias_grad
    global_bias -= learning_rate * global_bias_grad

    return user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias


def training_cycle(X_train, y_train, user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias,
                   learning_rate, epochs, batch_size, time_window, df):
    num_samples = len(X_train)
    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        batch_indices = np.array_split(indices, range(batch_size, num_samples, batch_size))
        for batch_indices in batch_indices:
            batch_user_ids = X_train[batch_indices, 0]
            batch_item_ids = X_train[batch_indices, 1]
            batch_ratings = y_train[batch_indices]
            batch_timestamps = df['timestamp'].iloc[batch_indices]
            user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias = update_parameters_with_time_window(
                user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias,
                batch_user_ids, batch_item_ids, batch_ratings, learning_rate, batch_timestamps, time_window)

def main():
    file_path = 'recommender_data.csv'
    embedding_dim = 10
    epochs = 10
    batch_size = 64
    learning_rate = 0.001
    time_window = pd.Timedelta(days=30)  # Adjust the time window as needed

    user_ids, item_ids, ratings, df = load_data(file_path, time_window)
    num_users, num_items = len(df['user_id'].unique()), len(df['item_id'].unique())

    user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias = create_embedding_matrices(
        num_users, num_items, embedding_dim)

    X = np.column_stack((user_ids, item_ids))
    y = ratings

    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)

    X_train, y_train = X[:split_index], y[:split_index]

    training_cycle(X_train, y_train, user_embedding_matrix, item_embedding_matrix, user_bias, item_bias, global_bias,
                   learning_rate, epochs, batch_size, time_window, df)

    predictions = collaborative_filtering_model(user_ids, item_ids, user_embedding_matrix, item_embedding_matrix,
                                                user_bias, item_bias, global_bias)

    df['predicted_rating'] = predictions * 5.0

    users = df['user_id'].unique()
    for user in users:
        user_df = df[df['user_id'] == user]
        top_recommendations = user_df.sort_values(by='predicted_rating', ascending=False).head(3)
        print(f"\nTop recommendations for {user}:")
        print(top_recommendations[['item_id', 'predicted_rating']])

if __name__ == "__main__":
    main()
