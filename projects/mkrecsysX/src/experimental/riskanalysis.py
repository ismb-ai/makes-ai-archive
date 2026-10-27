import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    df = df.dropna()

    le = LabelEncoder()
    df['Product_Name'] = le.fit_transform(df['Product_Name'])
    df['Brand_Name'] = le.fit_transform(df['Brand_Name'])

    df['Total_Monetary_Value'] = df.groupby('Client_ID')['Monetary_Value'].transform('sum')
    df['Total_Discount'] = df.groupby('Client_ID')['Discount'].transform('sum')
    df['Unique_Products'] = df.groupby('Client_ID')['Product_ID'].transform('nunique')
    df['Days_Since_Last_Purchase'] = (df['Sales_Date'].max() - df.groupby('Client_ID')['Sales_Date'].transform('max')).dt.days

    df['Churn'] = df['Days_Since_Last_Purchase'] > 30
    df = df.drop_duplicates(subset=['Client_ID'])

    print("Data loading and preprocessing completed.\n")
    return df

def train_gbm_model(df, features, target):
    print("Splitting the data into training and testing sets...")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split completed.\n")

    print("Training the Gradient Boosting Machine (GBM) model...")
    gbm = GradientBoostingClassifier(random_state=42)
    gbm.fit(X_train, y_train)
    print("Model training completed.\n")

    print("Evaluating the model...")
    y_pred = gbm.predict(X_test)
    y_prob = gbm.predict_proba(X_test)[:, 1]

    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    print("Model evaluation completed.\n")
    return gbm

def predict_churn_risk(gbm, df, features):
    print("Predicting churn risk for each client...")
    df['Churn_Risk'] = gbm.predict_proba(df[features])[:, 1]
    churn_risk_table = df[['Client_ID', 'Churn_Risk']].sort_values(by='Churn_Risk', ascending=False)
    print("Churn risk prediction completed.\n")

    print("Displaying churn risk table:")
    print(tabulate(churn_risk_table, headers='keys', tablefmt='grid'))

def plot_churn_risk(df):
    print("Plotting churn risk distribution and scatter plot...")

    plt.figure(figsize=(10, 6))
    plt.hist(df['Churn_Risk'], bins=20, color='blue', edgecolor='k', alpha=0.7)
    plt.title('Distribution of Churn Risk')
    plt.xlabel('Churn Risk')
    plt.ylabel('Number of Clients')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Client_ID'], df['Churn_Risk'], alpha=0.6, color='red')
    plt.title('Churn Risk of Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Churn Risk')
    plt.show()

    print("Plotting completed.\n")

def main():
    print("Starting the churn prediction process...")
    file_path = "/Users/igormol/Desktop/mkrecsysX/sample/sales-CPP-updated.csv"
    df = load_and_preprocess_data(file_path)

    features = ['Total_Monetary_Value', 'Total_Discount', 'Unique_Products', 'Days_Since_Last_Purchase']
    target = 'Churn'

    gbm = train_gbm_model(df, features, target)
    predict_churn_risk(gbm, df, features)
    plot_churn_risk(df)

    print("Churn prediction process completed.")

if __name__ == "__main__":
    main()
