import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Load the CSV file
df = pd.read_csv('/Users/igormol/Desktop/analytics/sample/CPP.csv')

# RFM Analysis Functions
def compute_rfm(df):
    current_date = pd.to_datetime(df['Sales_Date']).max()
    rfm_data = df.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - pd.to_datetime(x).max()).days,
        'Product_ID': 'count',
        'Monetary_Value': 'sum'
    }).reset_index()

    rfm_data['Recency'] = pd.qcut(rfm_data['Sales_Date'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
    rfm_data['Monetary'] = pd.qcut(rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1

    rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']
    rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / \
                                       (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
    return rfm_data

# Churn Analysis Functions
def load_and_preprocess_churn_data(df):
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    df = df.dropna()

    le = LabelEncoder()
    df['Product_Name'] = le.fit_transform(df['Product_Name'])
    df['Brand_Name'] = le.fit_transform(df['Brand_Name'])

    df['Total_Monetary_Value'] = df.groupby('Client_ID')['Monetary_Value'].transform('sum')
    df['Total_Discount'] = df.groupby('Client_ID')['Discount'].transform('sum')
    df['Unique_Products'] = df.groupby('Client_ID')['Product_ID'].transform('nunique')
    df['Days_Since_Last_Purchase'] = (df['Sales_Date'].max() - df.groupby('Client_ID')['Sales_Date'].transform('max')).dt.days
    df['Churn'] = (df['Days_Since_Last_Purchase'] > 30).astype(int)
    df = df.drop_duplicates(subset=['Client_ID'])

    return df

def train_rf_model(df, features, target):
    X, y = prepare_data(df, features, target)
    X_train, X_test, y_train, y_test = split_data(X, y)
    rf = train_model(X_train, y_train)
    evaluate_model(rf, X_test, y_test)
    return rf

def prepare_data(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def predict_churn_risk(rf, df, features):
    df['Churn_Risk'] = rf.predict_proba(df[features])[:, 1]
    churn_risk_table = df[['Client_ID', 'Churn_Risk']].sort_values(by='Churn_Risk', ascending=False)
    print("Churn Risk Table:")
    print(churn_risk_table)

# Recommender System Functions
def create_interaction_matrix(data_frame):
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)
    return interaction_matrix

def build_autoencoder_model(input_size):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_size,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(input_size, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

def get_embeddings(encoder_model, X):
    embeddings = encoder_model.predict(X)
    return embeddings

def recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    client_index = interaction_matrix.index.get_loc(client_id)
    client_embedding = embeddings[client_index]
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]
    return recommended_products

# Main Function
def main():
    print("Client Product Recommendation System")

    # Load and preprocess data
    churn_data = load_and_preprocess_churn_data(df)
    rfm_data = compute_rfm(df)
    combined_data = pd.merge(rfm_data, churn_data, on='Client_ID')

    # Train Random Forest model for churn prediction
    features = ['Total_Monetary_Value', 'Total_Discount', 'Unique_Products', 'Days_Since_Last_Purchase']
    target = 'Churn'
    rf_model = train_rf_model(churn_data, features, target)
    combined_data['Churn_Risk'] = rf_model.predict_proba(churn_data[features])[:, 1]

    # Train Autoencoder model for product recommendation
    interaction_matrix = create_interaction_matrix(df)
    X_train, X_test = train_test_split(interaction_matrix, test_size=0.2, random_state=42)
    autoencoder_model = build_autoencoder_model(X_train.shape[1])
    train_autoencoder(autoencoder_model, X_train, X_test)
    embeddings = get_embeddings(autoencoder_model, interaction_matrix.values)

    # Get user input for RFM and Churn Scores
    recency_threshold = int(input("Enter Recency Threshold (1 to 10): "))
    frequency_threshold = int(input("Enter Frequency Threshold (1 to 10): "))
    monetary_threshold = int(input("Enter Monetary Threshold (1 to 10): "))
    churn_threshold = float(input("Enter Churn Risk Threshold (0.0 to 1.0): "))

    # Filter clients based on RFM and Churn Scores
    filtered_clients = combined_data[
        (combined_data['Recency'] <= recency_threshold) &
        (combined_data['Frequency'] >= frequency_threshold) &
        (combined_data['Monetary'] >= monetary_threshold) &
        (combined_data['Churn_Risk'] <= churn_threshold)
    ]

    print(f"Filtered Clients (RFM and Churn Risk):")
    print(filtered_clients[['Client_ID', 'Recency', 'Frequency', 'Monetary', 'Churn_Risk']])

    # Recommend products to filtered clients
    for client_id in filtered_clients['Client_ID']:
        recommended_products = recommend_products_autoencoder(embeddings, client_id, interaction_matrix)
        print(f"Recommended Products for Client {client_id}: {recommended_products}")

if __name__ == "__main__":
    main()
