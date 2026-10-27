import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def encode_categorical_variables(data):
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    return data, label_encoders

def normalize_numerical_variables(data):
    scaler = StandardScaler()
    data['temperature'] = scaler.fit_transform(data[['temperature']])
    return data, scaler

def perform_clustering(data, contextual_columns, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data[contextual_columns])
    data_clustered = data.copy()
    data_clustered['cluster'] = cluster_labels
    cluster_centers = kmeans.cluster_centers_
    variance_within_clusters = cluster_centers.std(axis=0)
    return data_clustered, variance_within_clusters

def create_sequences(data, contextual_columns, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def build_model(seq_length, input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(input_shape[1])  # Output layer should match the number of features
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def forecast_user(user_data, model, seq_length, forecast_steps, label_encoders, scaler, contextual_columns):
    predictions = []
    current_seq = user_data[-seq_length:]
    for _ in range(forecast_steps):
        pred = model.predict(np.expand_dims(current_seq, axis=0))[0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [pred], axis=0)
    pred_df = pd.DataFrame(predictions, columns=contextual_columns)
    for col, le in label_encoders.items():
        if col in pred_df.columns:
            pred_df[col] = le.inverse_transform(pred_df[col].astype(int))
    pred_df['temperature'] = scaler.inverse_transform(pred_df[['temperature']])
    return pred_df

def forecast(file_path, seq_length=5, n_clusters=3, forecast_steps=4):
    data = load_data(file_path)
    data, label_encoders = encode_categorical_variables(data)
    data, scaler = normalize_numerical_variables(data)
    contextual_columns = ["weather_condition", "temperature", "operating_system", "device_type", "age_group",
                          "gender", "income_level", "mood", "activity", "time_of_day", "day_of_week", "season"]
    data_clustered, variance_within_clusters = perform_clustering(data, contextual_columns, n_clusters)
    X, y = create_sequences(data[contextual_columns].values, contextual_columns, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(seq_length, (seq_length, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)
    users = data['user_id'].unique()
    user_forecasts = {}
    for user_id in users:
        user_data = data[data['user_id'] == user_id][contextual_columns].values
        if len(user_data) > seq_length:
            predictions = forecast_user(user_data, model, seq_length, forecast_steps, label_encoders, scaler, contextual_columns)
            user_forecasts[user_id] = predictions
    return data_clustered, variance_within_clusters, user_forecasts

def main():
    data_clustered, variance_within_clusters, user_forecasts = forecast("/Users/igormol/Desktop/RecSysDNA/sample/large_contextual_data.csv")

    # Print forecast table for each user
    for user_id, forecast_df in user_forecasts.items():
        print(f"Forecast for User {user_id}:")
        print(forecast_df)
        print("\n")

    # Plot contextual variables sensitivity with respect to time
    plt.figure(figsize=(10, 6))
    contextual_columns = ["weather_condition", "temperature", "operating_system", "device_type", "age_group",
                          "gender", "income_level", "mood", "activity", "time_of_day", "day_of_week", "season"]
    for i, variable in enumerate(contextual_columns):
        plt.scatter(variance_within_clusters[i], np.arange(len(contextual_columns))[i], label=variable)

    plt.xlabel('Sensitivity of Variable with Respect to Time')
    plt.ylabel('Contextual Variable')
    plt.title('Contextual Variables Sensitivity Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
