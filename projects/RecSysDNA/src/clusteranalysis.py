# clusteranalysis.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains the system that analyzes sales events using a hybrid
# machine learning approach. The first phase employs a Gradient Boosting +
# Random Forest to learn the relative importance of contextual variables in
# determining whether a sale takes place. The second phase uses a k-Means Clus-
# tering algorithm to classify the sales reported by our user. Finally, the
# last phase applies a Decision Tree to determine the "typical context" (i.e.,
# the values assumed by the contextual variables) that leads to a sale.

# In sec. 1, we declare functions concerning data manipulation. In sec. 2, we
# define the core routines of our machine learning system: 1. The hybrid
# neural network, consisting of a Gradient Boosting + Random Forest; 2. The
# Decision Tree classifier. Finally, in sec. 3, we write the functions to print
# and plot the results into the web interface.

# Data manipulation and analysis
import pandas as pd

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit for web app interface
import streamlit as st

# Section 1: Data manipulation.

def convert_timestamps(data):
    # Convert the 'timestamp' column to datetime and extract hour, month, and year.
    # Parameters:
    # data (pd.DataFrame): The input dataframe containing a 'timestamp' column.
    # Returns:
    # pd.DataFrame: The dataframe with additional 'hour', 'month', and 'year' columns.
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['month'] = data['timestamp'].dt.month
    data['year'] = data['timestamp'].dt.year
    return data

def one_hot_encode(data, categorical_features):
    # Perform one-hot encoding on specified categorical features.
    # Parameters:
    # data (pd.DataFrame): The input dataframe containing categorical features.
    # categorical_features (list): List of column names to be one-hot encoded.
    # Returns:
    # pd.DataFrame: The dataframe with one-hot encoded features.
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_cats = onehot_encoder.fit_transform(data[categorical_features])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=onehot_encoder.get_feature_names_out(categorical_features))
    return encoded_cats_df

def drop_columns(data, columns_to_drop):
    # Drop specified columns from the dataframe.
    # Parameters:
    # data (pd.DataFrame): The input dataframe.
    # columns_to_drop (list): List of column names to be dropped.
    # Returns:
    # pd.DataFrame: The dataframe with specified columns dropped.
    return data.drop(columns=columns_to_drop)

def merge_data(data, encoded_cats_df):
    # Merge original dataframe with one-hot encoded dataframe.
    # Parameters:
    # data (pd.DataFrame): The original dataframe.
    # encoded_cats_df (pd.DataFrame): The dataframe containing one-hot encoded features.
    # Returns:
    # pd.DataFrame: The merged dataframe.
    return pd.concat([data.reset_index(drop=True), encoded_cats_df.reset_index(drop=True)], axis=1)

def adjust_product_id(data):
    # Adjust 'product_id' by subtracting 1.
    # Parameters:
    # data (pd.DataFrame): The input dataframe containing a 'product_id' column.
    # Returns:
    # pd.DataFrame: The dataframe with adjusted 'product_id' column.
    data['product_id'] = data['product_id'] - 1
    return data

def clusterAnalysis_preprocess_data(data):
    # Main preprocessing function to handle the overall data preprocessing.
    # Parameters:
    # data (pd.DataFrame): The input dataframe.
    # Returns:
    # pd.DataFrame: The preprocessed dataframe.
    context = {
        'timestamp_column': 'timestamp',
        'categorical_features': [
            'weather_condition', 'operating_system', 'device_type', 'age_group', 'gender',
            'income_level', 'mood', 'activity', 'time_of_day', 'day_of_week', 'season'
        ]
    }
    data = convert_timestamps(data)
    encoded_cats_df = one_hot_encode(data, context['categorical_features'])
    columns_to_drop = context['categorical_features'] + [context['timestamp_column']]
    data = drop_columns(data, columns_to_drop)
    data = merge_data(data, encoded_cats_df)
    data = adjust_product_id(data)
    return data

# Section 2: Core

def clusterAnalysis_train_models(X_train, y_train):
    # Train Random Forest and XGBoost classifiers on the training data.
    # Parameters:
    # X_train (pd.DataFrame): The training feature set.
    # y_train (pd.Series): The training labels.
    # Returns:
    # tuple: Trained Random Forest and XGBoost models.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    gbm_model = XGBClassifier(n_estimators=100, random_state=42)
    gbm_model.fit(X_train, y_train)
    return rf_model, gbm_model

def clusterAnalysis_feature_importances(rf_model, gbm_model, X):
    # Calculate and combine feature importances from Random Forest and XGBoost models.
    # Parameters:
    # rf_model (RandomForestClassifier): Trained Random Forest model.
    # gbm_model (XGBClassifier): Trained XGBoost model.
    # X (pd.DataFrame): The feature set.
    # Returns:
    # pd.DataFrame: DataFrame containing features and their combined importances.
    rf_importances = rf_model.feature_importances_
    gbm_importances = gbm_model.feature_importances_
    combined_importances = (rf_importances + gbm_importances) / 2
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': combined_importances
    }).sort_values(by='importance', ascending=False)
    return feature_importance_df

def clusterAnalysis_kmeans_clustering(X_important):
    # Perform KMeans clustering on the important features.
    # Parameters:
    # X_important (pd.DataFrame): DataFrame of important features.
    # Returns:
    # np.ndarray: Array of cluster labels.
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_important)
    return clusters

def clusterAnalysis_sort_clusters(data, clusters):
    # Sort clusters based on the count of 'product_id' in each cluster.
    # Parameters:
    # data (pd.DataFrame): The original data with 'cluster' column.
    # clusters (np.ndarray): Array of cluster labels.
    # Returns:
    # pd.DataFrame: DataFrame with clusters sorted by sales count.
    data['cluster'] = clusters
    cluster_sales = data.groupby('cluster')['product_id'].count().sort_values(ascending=False).index
    data['cluster'] = data['cluster'].apply(lambda x: list(cluster_sales).index(x))
    return data

def clusterAnalysis_train_decision_tree(data):
    # Train a Decision Tree classifier on high sales events.
    # Parameters:
    # data (pd.DataFrame): The original data with cluster information.
    # Returns:
    # tuple: Trained Decision Tree model and feature set for high sales events.
    st.header("Decision Tree Model for High Sales Events")
    high_sales_data = data[data['cluster'] == 0]
    X_high_sales = high_sales_data.drop(columns=['product_id', 'cluster'])
    y_high_sales = high_sales_data['product_id']
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_high_sales, y_high_sales)
    return dt_model, X_high_sales

# Section 3: User interface. Print table of results and plot clusters.

def clusterAnalysis_print_cluster_tables(data):
    # Display descriptive statistics for each cluster.
    # Parameters:
    # data (pd.DataFrame): The input dataframe with cluster information.
    # Returns:
    # None
    st.header("Cluster Analysis Results")
    for cluster in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster]
        st.subheader(f"Cluster {cluster}:")
        st.write(cluster_data.describe())

def clusterAnalysis_plot_clusters(X_important, data):
    # Plot clusters of sales events.
    # Parameters:
    # X_important (pd.DataFrame): DataFrame of important features.
    # data (pd.DataFrame): The original data with cluster labels.
    # Returns:
    # None
    st.header("Clusters of Sales Events")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_important.iloc[:, 0], y=X_important.iloc[:, 1], hue=data['cluster'], palette='viridis', ax=ax)
    ax.set_title('Clusters of Sales Events')
    st.pyplot(fig)

def clusterAnalysis_plot_decision_tree(dt_model, X_high_sales):
    # Plot decision tree for high sales events.
    # Parameters:
    # dt_model (DecisionTreeClassifier): Trained Decision Tree model.
    # X_high_sales (pd.DataFrame): Feature set for high sales events.
    # Returns:
    # None
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_model, feature_names=list(X_high_sales.columns), filled=True, class_names=[str(i) for i in dt_model.classes_], ax=ax)
    ax.set_title("Decision Tree for High Sales Events")
    st.pyplot(fig)

def clusterAnalysis_print_feature_importances(dt_model, X_high_sales):
    # Display feature importances for high sales events.
    # Parameters:
    # dt_model (DecisionTreeClassifier): Trained Decision Tree model.
    # X_high_sales (pd.DataFrame): Feature set for high sales events.
    # Returns:
    # None
    dt_importances = pd.DataFrame({
        'feature': X_high_sales.columns,
        'importance': dt_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    st.header("Feature Importances for High Sales Events")
    st.write(dt_importances)
