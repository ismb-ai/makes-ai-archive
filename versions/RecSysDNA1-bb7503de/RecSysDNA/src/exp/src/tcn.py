import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load data
file_path = "/Users/igormol/Desktop/RecSysDNA/sample/large_contextual_data.csv"
data = pd.read_csv(file_path)

# Select contextual variables columns
contextual_columns = ["weather_condition", "temperature", "operating_system", "device_type", "age_group", "gender",
                      "income_level", "mood", "activity", "time_of_day", "day_of_week", "season"]

# Encode categorical variables
label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Normalize numerical variables
scaler = StandardScaler()
data['temperature'] = scaler.fit_transform(data[['temperature']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 5  # Number of previous time steps to use for prediction
X, y = create_sequences(data[contextual_columns].values, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build TCN model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, X_train.shape[2])),
    Dropout(0.2),
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(X_train.shape[2])  # Output layer should match the number of features
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)

# Make predictions for each user
def forecast_user(user_id, user_data, model, seq_length, forecast_steps):
    predictions = []
    current_seq = user_data[-seq_length:]
    for _ in range(forecast_steps):
        pred = model.predict(np.expand_dims(current_seq, axis=0))[0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [pred], axis=0)
    return np.array(predictions)

# Forecast future values for each user
forecast_steps = 4  # Number of weeks ahead to forecast
users = data['user_id'].unique()
user_forecasts = {}

for user_id in users:
    user_data = data[data['user_id'] == user_id][contextual_columns].values
    if len(user_data) > seq_length:
        predictions = forecast_user(user_id, user_data, model, seq_length, forecast_steps)

        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions, columns=contextual_columns)

        # Inverse transform encoded categorical variables
        for col, le in label_encoders.items():
            if col in pred_df.columns:
                pred_df[col] = le.inverse_transform(pred_df[col].astype(int))

        # Inverse transform scaled numerical variables
        pred_df['temperature'] = scaler.inverse_transform(pred_df[['temperature']])

        user_forecasts[user_id] = pred_df

# Print forecast table for each user
for user_id, forecast_df in user_forecasts.items():
    print(f"Forecast for User {user_id}:")
    print(forecast_df)
    print("\n")
