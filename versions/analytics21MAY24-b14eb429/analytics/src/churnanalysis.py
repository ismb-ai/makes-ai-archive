# churnanalysis.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains the functions responsible for the customer retention
# analysis, also referred to as "Churn Analysis." We utilize a machine learning
# approach based upon a Random Forest decision tree.

# In sec. 1, we declare functions to load and prepare user's data and define the
# architecture of the Random Forest. In sec. 2, we write the routines
# responsible for the training of the model. Finally, in sec. 3, we define the
# functions to plot and print the results into the web interface.

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns

# Section 1: Load and Prepare user's data and builds decision-tree architecture.

def prepare_data(df, features, target):
    # Extract the feature columns (X) from the DataFrame using the specified feature names
    X = df[features]
    # Extract the target column (y) from the DataFrame using the specified target name
    y = df[target]
    # Return the feature matrix (X) and target vector (y)
    return X, y

def load_and_preprocess_churn_data(df):
    # Convert 'Sales_Date' column to datetime format
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])

    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical variables 'Product_Name' and 'Brand_Name' using LabelEncoder
    le = LabelEncoder()
    df['Product_Name'] = le.fit_transform(df['Product_Name'])
    df['Brand_Name'] = le.fit_transform(df['Brand_Name'])

    # Create new features based on client purchase history
    df['Total_Monetary_Value'] = df.groupby('Client_ID')['Monetary_Value'].transform('sum')
    df['Total_Discount'] = df.groupby('Client_ID')['Discount'].transform('sum')
    df['Unique_Products'] = df.groupby('Client_ID')['Product_ID'].transform('nunique')
    df['Days_Since_Last_Purchase'] = (df['Sales_Date'].max() - df.groupby('Client_ID')['Sales_Date'].transform('max')).dt.days

    # Define churn based on the number of days since last purchase (assuming churn if no purchase within last 30 days)
    df['Churn'] = (df['Days_Since_Last_Purchase'] > 30).astype(int)

    # Remove duplicate rows based on 'Client_ID'
    df = df.drop_duplicates(subset=['Client_ID'])

    # Return the preprocessed DataFrame
    return df

# Section 2: Routines to build and train the Random Forest model.

def train_rf_model(df, features, target):
    # Prepare the feature matrix X and the target vector y from the DataFrame
    X, y = prepare_data(df, features, target)

    # Check if there is a class imbalance in the target variable
    if check_class_imbalance(y):
        # If there is class imbalance, handle it by oversampling the minority class
        df = handle_class_imbalance(df, target)
        # Re-prepare the feature matrix X and the target vector y after handling class imbalance
        X, y = prepare_data(df, features, target)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the Random Forest model using the training data
    rf = train_model(X_train, y_train)

    # Evaluate the trained model using the testing data
    evaluate_model(rf, X_test, y_test)

    # Return the trained Random Forest model
    return rf

def check_class_imbalance(y):
    # Calculate the counts of each class in the target vector (y)
    class_counts = y.value_counts()
    # Display the class distribution in the Streamlit app
    st.write("Class distribution:", class_counts)
    # Check if there is a significant class imbalance (one class is more than twice the size of the other)
    return class_counts[0] / class_counts[1] > 2 or class_counts[1] / class_counts[0] > 2

def handle_class_imbalance(df, target):
    # Display a message indicating that class imbalance is being handled
    st.write("Handling class imbalance")

    # Calculate the counts of each class in the target column
    class_counts = df[target].value_counts()

    # Separate the majority and minority class samples from the DataFrame
    df_majority = df[df[target] == class_counts.idxmax()]
    df_minority = df[df[target] == class_counts.idxmin()]

    # Upsample the minority class to have the same number of samples as the majority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)

    # Combine the upsampled minority class with the majority class
    return pd.concat([df_majority, df_minority_upsampled])

def split_data(X, y):
    # Split the feature matrix (X) and target vector (y) into training and testing sets
    # The test set will be 20% of the data, and the split will be random but reproducible with random_state=42
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Initialize a Random Forest classifier with a fixed random state for reproducibility
    rf = RandomForestClassifier(random_state=42)
    # Train the Random Forest classifier using the training data
    rf.fit(X_train, y_train)
    # Return the trained Random Forest model
    return rf

def evaluate_model(model, X_test, y_test):
    # Use the trained model to predict the target values for the test data
    y_pred = model.predict(X_test)
    # Use the trained model to predict the probabilities for the positive class for the test data
    y_prob = model.predict_proba(X_test)[:, 1]

    # Display the accuracy of the model on the test data in the Streamlit app
    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
    # Display the classification report, which includes precision, recall, and F1-score
    st.write("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot and display the confusion matrix for the test data predictions
    plot_confusion_matrix(y_test, y_pred)
    # Plot and display the ROC curve for the test data predictions
    plot_roc_curve(y_test, y_prob)

def predict_churn_risk(rf, df, features):
    # Predict churn risk probabilities for each customer using the trained Random Forest model
    df['Churn_Risk'] = rf.predict_proba(df[features])[:, 1]
    # Create a DataFrame with client IDs and their corresponding churn risk probabilities, sorted by descending churn risk
    churn_risk_table = df[['Client_ID', 'Churn_Risk']].sort_values(by='Churn_Risk', ascending=False)

    # Display the churn risk table in the Streamlit app
    st.write("Churn Risk Table:")
    st.table(churn_risk_table)

# Section 3: Functions to print and plot results.

def plot_confusion_matrix(y_test, y_pred):
    # Calculate the confusion matrix from the true and predicted target values
    cm = confusion_matrix(y_test, y_pred)
    # Display the confusion matrix values in the Streamlit app
    st.write("\nConfusion Matrix:\n", cm)

    # Create a new figure for the confusion matrix plot
    plt.figure(figsize=(8, 6))
    # Plot the confusion matrix using a heatmap for better visualization
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    # Set labels for the axes
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # Set a title for the plot
    plt.title('Confusion Matrix')
    # Display the plot in the Streamlit app
    st.pyplot(plt)

def plot_roc_curve(y_test, y_prob):
    # Calculate the false positive rate (fpr) and true positive rate (tpr) for different thresholds
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Create a new figure for the ROC curve plot
    plt.figure(figsize=(8, 6))
    # Plot the ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # Plot a diagonal line (random classifier) for comparison
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # Set limits for the axes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Set labels for the axes
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Set a title for the plot
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    # Add a legend to the plot
    plt.legend(loc="lower right")
    # Display the plot in the Streamlit app
    st.pyplot(plt)

def plot_churn_risk(df):
    # Plot a histogram showing the distribution of churn risk probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(df['Churn_Risk'], bins=20, color='blue', edgecolor='k', alpha=0.7)
    plt.title('Distribution of Churn Risk')
    plt.xlabel('Churn Risk')
    plt.ylabel('Number of Clients')
    # Display the histogram plot in the Streamlit app
    st.pyplot(plt)

    # Plot a scatter plot showing the churn risk of each client
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Client_ID'], df['Churn_Risk'], alpha=0.6, color='red')
    plt.title('Churn Risk of Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Churn Risk')
    # Display the scatter plot in the Streamlit app
    st.pyplot(plt)
