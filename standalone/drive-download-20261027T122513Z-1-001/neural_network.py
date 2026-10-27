import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tabulate import tabulate
import squarify
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/Users/igormol/Downloads/inadimplentes_parcelas_2024-06-12_renamed_english.csv"
data = pd.read_csv(file_path)

# Handle missing values (if any)
data = data.replace("0", np.nan)  # Replace "0" with NaN to properly handle missing dates
data = data.fillna(0)  # Fill NaN values with 0 or another appropriate value

# Encode categorical variables
label_encoder_client = LabelEncoder()
data['Client'] = label_encoder_client.fit_transform(data['Client'])

label_encoder_agreement = LabelEncoder()
data['Agreement'] = label_encoder_agreement.fit_transform(data['Agreement'])

# Convert columns to appropriate data types
data['Cpf'] = data['Cpf'].astype(str)
data['Contract'] = data['Contract'].astype(str)

# Handle date columns separately
data['Reference_Date'] = pd.to_datetime(data['Reference_Date'], errors='coerce')
data['Due_Date_Installment'] = pd.to_datetime(data['Due_Date_Installment'], errors='coerce')

# Fill any remaining NaN values in date columns with a default date or handle appropriately
data['Reference_Date'] = data['Reference_Date'].fillna(pd.Timestamp('1970-01-01'))
data['Due_Date_Installment'] = data['Due_Date_Installment'].fillna(pd.Timestamp('1970-01-01'))

# Extract useful time features
data['Reference_Year'] = data['Reference_Date'].dt.year
data['Reference_Month'] = data['Reference_Date'].dt.month
data['Due_Year'] = data['Due_Date_Installment'].dt.year
data['Due_Month'] = data['Due_Date_Installment'].dt.month

# Drop original date columns
data = data.drop(columns=['Reference_Date', 'Due_Date_Installment'])

# Standardize numerical features
numerical_features = ['Installment_Amount', 'Amount_Paid_Month', 'Delay_Of_The_Month', 'Overdue_Balance', 'Days_Overdue', 'Included_Charges']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define features and target
features = ['Client', 'Cpf', 'Contract', 'Agreement', 'Installment_Number', 'Reference_Year', 'Reference_Month', 'Due_Year', 'Due_Month'] + numerical_features
X = data[features]

# Encode 'Cpf' and 'Contract' separately to ensure original values are retained
label_encoder_cpf = LabelEncoder()
X['Cpf'] = label_encoder_cpf.fit_transform(data['Cpf'])

label_encoder_contract = LabelEncoder()
X['Contract'] = label_encoder_contract.fit_transform(data['Contract'])

# Define the target variable
y = (data['Days_Overdue'] > 0).astype(int)  # Binary target for default

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=len(features), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Predict default probabilities
y_pred_prob = model.predict(X_test)
y_pred_prob_percentage = y_pred_prob.flatten() * 100

# Perform hierarchical clustering to classify clients into risk classes
clustering_model = AgglomerativeClustering(n_clusters=3)
risk_classes = clustering_model.fit_predict(y_pred_prob)

# Map cluster labels to risk categories
risk_category_mapping = {0: 'C', 1: 'B', 2: 'A'}
risk_categories = np.vectorize(risk_category_mapping.get)(risk_classes)

# Combine results into a DataFrame
results = pd.DataFrame({
    'Client': label_encoder_client.inverse_transform(X_test['Client']),
    'Cpf': label_encoder_cpf.inverse_transform(X_test['Cpf']),
    'Default_Probability (%)': y_pred_prob_percentage,
    'Risk_Class': risk_categories
})

# Sort results by Default Probability in descending order
results = results.sort_values(by='Default_Probability (%)', ascending=False)

# Print the complete table of results using tabulate
print(tabulate(results, headers='keys', tablefmt='pretty'))

# Save the sorted results to a CSV file
results.to_csv('/Users/igormol/Downloads/credit_risk_scores.csv', index=False)

# Calculate cluster statistics
cluster_stats = results.groupby('Risk_Class').agg({
    'Client': 'count',
    'Default_Probability (%)': 'mean'
}).reset_index()

cluster_stats['Percentage_of_Clients'] = (cluster_stats['Client'] / cluster_stats['Client'].sum()) * 100

# Plot treemap
sizes = cluster_stats['Client']
labels = [f"Cluster {row['Risk_Class']}\n{row['Percentage_of_Clients']:.2f}% of clients\nAvg Default Prob: {row['Default_Probability (%)']:.2f}%" for idx, row in cluster_stats.iterrows()]

plt.figure(figsize=(12, 8))
squarify.plot(sizes=sizes, label=labels, alpha=.8)
plt.axis('off')
plt.title('Treemap of Client Clusters by Risk Class')
plt.show()
