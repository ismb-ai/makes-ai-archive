import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from tabulate import tabulate

def bayes_load_data(filepath):
    df = pd.read_csv(filepath)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    return df

def bayes_compute_rfm(df):
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

def bayes_preprocess_churn_data(df):
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

def bayes_check_class_imbalance(y):
    class_counts = y.value_counts()
    return class_counts[0] / class_counts[1] > 2 or class_counts[1] / class_counts[0] > 2

def bayes_handle_class_imbalance(df, target):
    class_counts = df[target].value_counts()
    df_majority = df[df[target] == class_counts.idxmax()]
    df_minority = df[df[target] == class_counts.idxmin()]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)
    return pd.concat([df_majority, df_minority_upsampled])

def bayes_train_rf_model(df, features, target):
    X, y = df[features], df[target]
    if bayes_check_class_imbalance(y):
        df = bayes_handle_class_imbalance(df, target)
        X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    bayes_evaluate_model(rf, X_test, y_test)
    return rf

def bayes_evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    bayes_plot_confusion_matrix(y_test, y_pred)
    bayes_plot_roc_curve(y_test, y_prob)

def bayes_plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def bayes_plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def bayes_train_bayesian_decision_tree(df, features, target):
    X, y = df[features], df[target]
    model = make_pipeline(StandardScaler(), GaussianNB())
    model.fit(X, y)
    return model

def bayes_create_interaction_matrix(data_frame):
    interaction_matrix = data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)
    return interaction_matrix

def bayes_recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=5):
    client_index = interaction_matrix.index.get_loc(client_id)
    client_embedding = embeddings[client_index]
    similarities = np.dot(embeddings, client_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(client_embedding))
    valid_indices = np.where(similarities.argsort()[::-1] < len(interaction_matrix.columns))[0]
    product_indices = similarities.argsort()[::-1][valid_indices][:k]
    client_purchases = interaction_matrix.loc[client_id]
    purchased_product_ids = client_purchases[client_purchases > 0].index.values
    recommended_products = [product_id for product_id in interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]
    return recommended_products

def bayes_build_autoencoder_model(input_size):
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

def bayes_train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32):
    autoencoder_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1)

def bayes_get_embeddings(autoencoder_model, interaction_matrix):
    get_embedding_layer = K.function([autoencoder_model.layers[0].input], [autoencoder_model.layers[2].output])
    embeddings = get_embedding_layer([interaction_matrix.to_numpy()])[0]
    return embeddings

def bayes_display_results_table(combined_df):
    print(tabulate(combined_df, headers='keys', tablefmt='psql'))

def main():
    filepath = "/Users/igormol/Desktop/analytics/sample/CPP.csv"
    df = bayes_load_data(filepath)

    rfm_data = bayes_compute_rfm(df)
    churn_df = bayes_preprocess_churn_data(df)

    features = ['Total_Monetary_Value', 'Total_Discount', 'Unique_Products', 'Days_Since_Last_Purchase']
    target = 'Churn'
    rf_model = bayes_train_rf_model(churn_df, features, target)

    churn_df['Churn_Risk'] = rf_model.predict_proba(churn_df[features])[:, 1]
    combined_df = pd.merge(rfm_data, churn_df[['Client_ID', 'Churn_Risk']], on='Client_ID')

    combined_df['RFM_Category'] = pd.qcut(combined_df['RFM_Score_Normalized'], q=5, labels=False)

    bayesian_tree = bayes_train_bayesian_decision_tree(combined_df, ['RFM_Score_Normalized', 'Churn_Risk'], 'RFM_Category')
    combined_df['Attention_Needed'] = bayesian_tree.predict(combined_df[['RFM_Score_Normalized', 'Churn_Risk']])

    bayes_display_results_table(combined_df[['Client_ID', 'RFM_Score_Normalized', 'Churn_Risk', 'Attention_Needed']])

    interaction_matrix = bayes_create_interaction_matrix(df)
    X_train, X_test = train_test_split(interaction_matrix, test_size=0.2, random_state=42)
    autoencoder = bayes_build_autoencoder_model(X_train.shape[1])
    bayes_train_autoencoder(autoencoder, X_train, X_test, epochs=10, batch_size=32)

    embeddings = bayes_get_embeddings(autoencoder, interaction_matrix)
    clients_attention = combined_df[combined_df['Attention_Needed'] == 1]['Client_ID']

    for client_id in clients_attention:
        recommended_products = bayes_recommend_products_autoencoder(embeddings, client_id, interaction_matrix)
        print(f'Recommended Products for Client {client_id}: {recommended_products}')

if __name__ == "__main__":
    main()
