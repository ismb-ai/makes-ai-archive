import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# Load the CSV file
file_path = "/Users/igormol/Desktop/RecSysDNA/sample/large_contextual_data.csv"
data = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['weather_condition', 'operating_system', 'device_type', 'age_group', 'gender', 'income_level', 'mood', 'activity', 'time_of_day', 'day_of_week', 'season']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Extract features and target
X = data[categorical_columns + ['temperature']]
y = (data['product_id'] > 0).astype(int)  # Assuming 'product_id' > 0 means sold

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train['temperature'] = scaler.fit_transform(X_train[['temperature']])
X_test['temperature'] = scaler.transform(X_test[['temperature']])

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Calculate SHAP values
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test, nsamples=100)

# Convert SHAP values to a DataFrame
shap_df = pd.DataFrame(shap_values[0], columns=X_test.columns)

# Print feature importance for each product
results = pd.DataFrame({'product_id': data.iloc[X_test.index]['product_id'], 'prediction': model.predict(X_test).flatten()})
results = pd.concat([results, shap_df], axis=1)

print("Feature Importance per Product:")
print(results)

# If you want to print it in a more readable format:
for index, row in results.iterrows():
    print(f"Product ID: {row['product_id']}, Prediction: {row['prediction']}")
    for col in X_test.columns:
        print(f"  {col}: {row[col]}")
    print("\n")
