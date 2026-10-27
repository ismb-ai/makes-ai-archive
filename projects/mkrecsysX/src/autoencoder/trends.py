import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Read the CSV file
df = pd.read_csv("/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv")

# Convert 'Sales_Date' to datetime format
df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])

# Compute Recency, Frequency, and Monetary Value
current_date = df['Sales_Date'].max()
rfm = df.groupby(['Client_ID', pd.Grouper(key='Sales_Date', freq='M')]).agg({
    'Sales_Date': lambda x: (current_date - x.max()).days,
    'Sales_Date': 'count',
    'Monetary_Value': 'sum'
}).rename(columns={'Sales_Date': 'Recency', 'Monetary_Value': 'Monetary'}).reset_index()

# Create a table to store angular coefficients
angular_coefficients = []

# Create a plot for each client and calculate angular coefficients
for client_id, group in rfm.groupby('Client_ID'):
    if len(group) < 2:  # Skip clients with insufficient data
        continue
    
    # Prepare data for linear regression
    X = group.index.values.reshape(-1, 1)
    y = group['Recency'].values
    
    # Create and fit linear regression model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Calculate angular coefficient (slope)
    angular_coefficient = lr.coef_[0]
    angular_coefficients.append([client_id, angular_coefficient])

# Convert angular coefficients to DataFrame
angular_coefficients_df = pd.DataFrame(angular_coefficients, columns=['Client_ID', 'Angular_Coefficient'])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Suppress the warning by setting n_init explicitly
kmeans.fit(angular_coefficients_df[['Angular_Coefficient']])
angular_coefficients_df['Cluster'] = kmeans.labels_

# Define cluster labels
cluster_labels = {
    0: "Loyal customer",
    1: "Stable customer",
    2: "Customer at risk"
}

# Map cluster labels to cluster numbers
angular_coefficients_df['Cluster_Label'] = angular_coefficients_df['Cluster'].map(cluster_labels)

# Plot clustering results
plt.figure(figsize=(10, 6))
for label, color in zip(angular_coefficients_df['Cluster_Label'].unique(), ['blue', 'green', 'red']):
    subset = angular_coefficients_df[angular_coefficients_df['Cluster_Label'] == label]
    plt.scatter(subset['Client_ID'], subset['Angular_Coefficient'], label=label, color=color)

plt.title('Clustering of Clients based on Angular Coefficients')
plt.xlabel('Client ID')
plt.ylabel('Angular Coefficient')
plt.legend()

plt.show()

# Print clustering results
print("\nClustering Results:")
print(angular_coefficients_df)

# Plot the RFM forecast for each client
for client_id, group in rfm.groupby('Client_ID'):
    if len(group) < 2:  # Skip clients with insufficient data
        continue
    
    # Prepare data for linear regression
    X = group.index.values.reshape(-1, 1)
    y = group['Recency'].values
    
    # Create and fit linear regression model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Make predictions
    y_pred = lr.predict(X)
    
    # Plot historical data and forecast
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(group.index, group['Recency'], label='Historical Data')
    ax.plot(group.index, y_pred, label='Forecast', linestyle='--')
    
    ax.set_title(f'RFM Score Forecast for Client ID: {client_id}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Recency')
    ax.legend()
    ax.annotate(f'Angular Coefficient: {angular_coefficients_df[angular_coefficients_df["Client_ID"] == client_id]["Angular_Coefficient"].values[0]}', 
                xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
