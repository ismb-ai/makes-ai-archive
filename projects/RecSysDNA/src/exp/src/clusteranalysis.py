import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree

def clusterAnalysis_load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clusterAnalysis_preprocess_data(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['month'] = data['timestamp'].dt.month
    data['year'] = data['timestamp'].dt.year

    categorical_features = ['weather_condition', 'operating_system', 'device_type', 'age_group', 'gender', 'income_level', 'mood', 'activity', 'time_of_day', 'day_of_week', 'season']
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_cats = onehot_encoder.fit_transform(data[categorical_features])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=onehot_encoder.get_feature_names_out(categorical_features))

    data = data.drop(columns=categorical_features + ['timestamp'])
    data = pd.concat([data.reset_index(drop=True), encoded_cats_df.reset_index(drop=True)], axis=1)

    data['product_id'] = data['product_id'] - 1
    return data

def clusterAnalysis_train_models(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    gbm_model = XGBClassifier(n_estimators=100, random_state=42)
    gbm_model.fit(X_train, y_train)

    return rf_model, gbm_model

def clusterAnalysis_feature_importances(rf_model, gbm_model, X):
    rf_importances = rf_model.feature_importances_
    gbm_importances = gbm_model.feature_importances_
    combined_importances = (rf_importances + gbm_importances) / 2

    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': combined_importances
    }).sort_values(by='importance', ascending=False)

    return feature_importance_df

def clusterAnalysis_kmeans_clustering(X_important):
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_important)
    return clusters

def clusterAnalysis_sort_clusters(data, clusters):
    cluster_sales = data.groupby('cluster')['product_id'].count().sort_values(ascending=False).index
    data['cluster'] = data['cluster'].apply(lambda x: list(cluster_sales).index(x))
    return data

def clusterAnalysis_print_cluster_tables(data):
    for cluster in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster]
        print(f"Cluster {cluster}:")
        print(cluster_data.describe(include='all'))
        print("\n")

def clusterAnalysis_plot_clusters(X_important, data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_important.iloc[:, 0], y=X_important.iloc[:, 1], hue=data['cluster'], palette='viridis')
    plt.title('Clusters of Sales Events')
    plt.show()

def clusterAnalysis_train_decision_tree(data):
    high_sales_data = data[data['cluster'] == 0]
    X_high_sales = high_sales_data.drop(columns=['product_id', 'cluster'])
    y_high_sales = high_sales_data['product_id']

    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_high_sales, y_high_sales)

    return dt_model, X_high_sales

def clusterAnalysis_plot_decision_tree(dt_model, X_high_sales):
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=list(X_high_sales.columns), filled=True, class_names=[str(i) for i in dt_model.classes_])
    plt.show()

def clusterAnalysis_print_feature_importances(dt_model, X_high_sales):
    dt_importances = pd.DataFrame({
        'feature': X_high_sales.columns,
        'importance': dt_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("Feature importances for high sales events:")
    print(dt_importances)

def main():
    data = clusterAnalysis_load_data('/Users/igormol/Desktop/RecSysDNA/sample/large_contextual_data.csv')
    data = clusterAnalysis_preprocess_data(data)
    X = data.drop(columns=['product_id'])
    y = data['product_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model, gbm_model = clusterAnalysis_train_models(X_train, y_train)
    feature_importance_df = clusterAnalysis_feature_importances(rf_model, gbm_model, X)

    important_features = feature_importance_df[feature_importance_df['importance'] > 0.01]['feature']
    X_important = X[important_features]

    clusters = clusterAnalysis_kmeans_clustering(X_important)
    data['cluster'] = clusters
    data = clusterAnalysis_sort_clusters(data, clusters)

    clusterAnalysis_print_cluster_tables(data)
    clusterAnalysis_plot_clusters(X_important, data)

    dt_model, X_high_sales = clusterAnalysis_train_decision_tree(data)
    clusterAnalysis_plot_decision_tree(dt_model, X_high_sales)
    clusterAnalysis_print_feature_importances(dt_model, X_high_sales)

if __name__ == "__main__":
    main()
