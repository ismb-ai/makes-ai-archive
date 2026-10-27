import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(columns=['Unnamed: 7'])

    # Encode categorical variables
    label_encoders = {}
    for column in ['Client_ID', 'Product_ID', 'Product_Name', 'Brand_Name']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Convert Sales_Date to datetime and extract relevant features
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'])
    data['Year'] = data['Sales_Date'].dt.year
    data['Month'] = data['Sales_Date'].dt.month
    data['Day'] = data['Sales_Date'].dt.day
    data = data.drop(columns=['Sales_Date'])

    return data, label_encoders

def split_and_scale_data(data):
    X = data.drop(columns=['Monetary_Value'])
    y = data['Monetary_Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train_scaled, y_train):
    # Gradient Boosting
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train_scaled, y_train)
    gb_importance = gb_model.feature_importances_

    # Random Forest
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train_scaled, y_train)
    rf_importance = rf_model.feature_importances_

    # Combine importances
    feature_importance = (gb_importance + rf_importance) / 2

    return gb_model, rf_model, feature_importance

def train_decision_tree(X_train_scaled, y_train):
    dt_model = DecisionTreeRegressor(max_depth=5)
    dt_model.fit(X_train_scaled, y_train)
    return dt_model

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def save_results(feature_importance, dt_model, X_test_scaled, y_test, pdf_path):
    feature_names = X_test_scaled.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    mse, r2 = evaluate_model(dt_model, X_test_scaled, y_test)

    with PdfPages(pdf_path) as pdf:
        # Feature Importance Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Decision Tree Plot
        plt.figure(figsize=(20, 10))
        plot_tree(dt_model, feature_names=feature_names, filled=True, rounded=True)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Model Evaluation Results
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.1, 0.8, f'Mean Squared Error: {mse}', fontsize=12)
        ax.text(0.1, 0.6, f'R-squared: {r2}', fontsize=12)
        ax.axis('off')
        pdf.savefig()
        plt.close()

        # Feature Importances from Decision Tree
        importance_df_dt = pd.DataFrame({'Feature': feature_names, 'Importance': dt_model.feature_importances_})
        importance_df_dt = importance_df_dt.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df_dt, ax=ax)
        ax.set_title('Feature Importance from Decision Tree')
        pdf.savefig()
        plt.close()

    print(f'Results saved to {pdf_path}')

def main():
    file_path = '/Users/igormol/Desktop/RecSysDNA/sample/Vaz.csv'
    pdf_path = '/Users/igormol/Desktop/RecSysDNA/sample/analysis_results.pdf'

    data = load_data(file_path)
    data, label_encoders = preprocess_data(data)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(data)
    gb_model, rf_model, feature_importance = train_models(X_train_scaled, y_train)
    dt_model = train_decision_tree(X_train_scaled, y_train)
    save_results(feature_importance, dt_model, X_test_scaled, y_test, pdf_path)

if __name__ == "__main__":
    main()
