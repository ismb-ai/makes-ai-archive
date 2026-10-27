# webinterface.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the routines responsible for building our web
# interface. In sec. 1, we declare functions to load and processes user's data.
# In sec. 2, we construct routines that defines the user's interface, such as
# page's layout, text boxes, buttons and tabs. Sec. 3, contains Executive
# functions that performs the business analysis for each tab.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
import seaborn as sns
from recsys import *
from rfm import *
from trends import *
from roi import *
from churnanalysis import *
from anomaly import *

# Section 1: Functions to load and process the client's data

# Load and preprocess the user's CSV file.
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')
    excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
    excluded_brands = ['DEERE']
    df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
    df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])  # Convert Sales_Date to datetime
    return df

def upload_csv_file():
    # Define a string to be used as the label for the file uploader
    upload_key = "Upload CSV File"

    # Create a file uploader widget in the sidebar of a Streamlit app
    # - The label of the uploader is set to the value of 'upload_key'
    # - The uploader is assigned a unique key 'file_uploader'
    # - The uploader only accepts files of type 'csv'
    file = st.sidebar.file_uploader(upload_key, key="file_uploader", type=["csv"])

    # Return the uploaded file object
    return file

def process_data(file):
    # Read the sales data from the provided file using the 'sales_read_data' function
    df = sales_read_data(file)

    # Convert date-related columns in the DataFrame to timestamps using the 'sales_convert_to_timestamps' function
    df = sales_convert_to_timestamps(df)

    # Calculate RFM (Recency, Frequency, Monetary) scores for the data using the 'sales_calculate_rfm_scores' function
    rfm_scores = sales_calculate_rfm_scores(df)

    # Calculate the geometric mean of the RFM scores and store it in a new column 'Geometric_Mean' in the 'rfm_scores' DataFrame
    rfm_scores['Geometric_Mean'] = sales_calculate_geometric_mean(rfm_scores[['Recency', 'Frequency', 'Monetary']])

    # Return the processed sales data DataFrame and the RFM scores DataFrame
    return df, rfm_scores

def create_product_features(df, rfm_scores):
    # Group the input DataFrame 'df' by the 'Product_ID' column
    grouped_data = df.groupby('Product_ID')

    # Initialize an empty DataFrame 'product_features' with specified columns
    product_features = pd.DataFrame(columns=['Angular_Coefficient', 'Mean_RFM_Score', 'Cluster'])

    # Iterate over each product group in the grouped data
    for product_id, group in grouped_data:
        # Extract the 'Monetary_Value' column from the group and reshape it for linear regression
        X = group['Monetary_Value'].values.reshape(-1, 1)

        # Convert the 'Sales_Date' column to datetime and then to integer (number of nanoseconds since epoch)
        y = pd.to_datetime(group['Sales_Date']).astype(int)

        # Create a linear regression model
        model = LinearRegression()

        # Fit the linear regression model using 'Monetary_Value' as the predictor and 'Sales_Date' as the response
        model.fit(X, y)

        # Retrieve the angular coefficient (slope) from the linear regression model
        angular_coefficient = model.coef_[0]

        # Get the geometric mean RFM score for the current product from the 'rfm_scores' DataFrame
        mean_rfm_score = rfm_scores.loc[product_id, 'Geometric_Mean']

        # Add a new row to 'product_features' DataFrame with the computed features for the current product
        product_features.loc[product_id] = [angular_coefficient, mean_rfm_score, np.nan]

    # Perform clustering on the 'product_features' DataFrame and update it with cluster labels
    product_features = sales_perform_clustering(product_features)

    # Return the 'product_features' DataFrame with computed features and cluster labels
    return product_features

# Section 2: Routines concerning the web-app interface

def display_intro():
    st.title("Makes Business Analytics X")
    intro_text = """
    Welcome to Makes Business Analytics X!

    Makes Business Analytics X is an advanced recommendation system web application designed to optimize customer engagement and enhance business profitability. By leveraging cutting-edge machine learning algorithms, Makes RecSys Pro offers a comprehensive suite of analysis tools, including:

    1. Personalized Product Recommendations: Tailors product recommendations to individual customers based on their historical purchasing behavior, fostering customer satisfaction and loyalty.
    2. Dynamic RFM Analysis: Utilizes RFM (Recency, Frequency, Monetary) analysis to segment customers and identify high-value customer segments, enabling targeted marketing strategies for maximum revenue generation.
    3. Client Similarity Analysis: Identifies groups of clients with similar purchasing behavior, allowing businesses to create tailored marketing campaigns and personalized recommendations to improve customer satisfaction.
    4. Product Performance Analysis: Evaluates the performance of products based on their recency, frequency, and monetary value, empowering businesses to optimize inventory management and drive sales growth.

    Experience the power of data-driven decision-making with Makes RecSys Pro! Simply upload your CSV file to get started.
    """
    st.markdown(intro_text)

def select_analysis_tab():
    tabs = ["Recommendations: Clients to Products", "Recommendations: Products to Clients", "RFM Analysis",
            "Client Similarity Analysis", "Product Analysis", "RFM Trends Analysis", "ABC Analysis",
            "Sales Performance", "ROI/ROE Analysis", "Client Retention Analysis", "Sales Anomaly Detection"]
    selected_tab = st.sidebar.radio("Select Analysis", tabs)
    return selected_tab

# Section 3: Executive functions; routines that performs buiness analysis.

def run_recommendations_clients_to_products(df, interaction_matrix):
    st.subheader("Recommendations for Clients to Products")
    st.markdown("""
    This analysis provides personalized product recommendations for clients based on their historical purchasing behavior. By leveraging machine learning algorithms, it identifies products that are likely to be of interest to individual clients, enabling businesses to offer tailored recommendations and enhance customer satisfaction.
    """)

    # Create interaction matrix if not provided
    if interaction_matrix is None:
        interaction_matrix = create_interaction_matrix(df)

    # Split interaction matrix into training and testing data
    train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)

    # Build and train the autoencoder model
    autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
    train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)

    # Extract embeddings from the encoder model
    encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)

    # Precompute recommendations for all clients and store in a dictionary
    all_recommendations = {}
    for client_id in interaction_matrix.index:
        all_recommendations[client_id] = recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=15)

    # Ensure 'Product_ID' is unique by dropping duplicates or aggregating
    df_unique_products = df.drop_duplicates(subset='Product_ID')

    # Convert product information to a dictionary for quick lookup
    product_info = df_unique_products.set_index('Product_ID')[['Product_Name', 'Brand_Name']].to_dict('index')

    # Process and display recommendations for each client
    for client_id, recommended_products in all_recommendations.items():
        with st.expander(f"Client {client_id} Recommendations"):
            actual_products = interaction_matrix.loc[client_id]
            recall_at_k = len(set(recommended_products).intersection(actual_products[actual_products > 0].index)) / len(actual_products[actual_products > 0])

            table_data = [
                [product_id, product_info.get(product_id, {}).get('Product_Name', 'Unknown'), product_info.get(product_id, {}).get('Brand_Name', 'Unknown')]
                for product_id in recommended_products
            ]

            st.write(f"Recall at K = 5: {recall_at_k:.2f}")
            st.table(pd.DataFrame(table_data, columns=["Product ID", "Product Name", "Brand"]))

def run_recommendations_products_to_clients(df, interaction_matrix):
    # Display subheader and markdown with explanation
    st.subheader("Recommendations for Products to Clients")
    st.markdown("""
    This analysis provides recommendations for products that are likely to be purchased by clients based on their preferences and historical buying patterns. By understanding the interests of individual clients, businesses can optimize their product offerings and enhance customer engagement.
    """)

    # Create interaction matrix if not provided
    if interaction_matrix is None:
        interaction_matrix = create_interaction_matrix(df)

    # Split interaction matrix into training and testing data
    train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)

    # Build and train the autoencoder model
    autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
    train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)

    # Extract embeddings from the encoder model
    encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)

    # Precompute recommendations for all products and store in a dictionary
    all_recommendations = {}
    for product_id in interaction_matrix.columns:
        all_recommendations[product_id] = recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)

    # Process and display recommendations for each product
    for product_id, recommended_clients in all_recommendations.items():
        with st.expander(f"Product {product_id} Recommendations"):
            if not recommended_clients.empty:  # Check if recommended_clients is not empty
                recall_at_k = len(set(recommended_clients).intersection(interaction_matrix.index)) / len(interaction_matrix.index)
                recommendation_data = [[client_id] for client_id in recommended_clients]
                st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                st.table(pd.DataFrame(recommendation_data, columns=["Client ID"]))
            else:
                st.write("No recommendations for this product.")

def run_rfm_analysis(df):
    # Display a subheader for the RFM Analysis section in a Streamlit app
    st.subheader("RFM Analysis")

    # Provide a markdown description explaining what RFM analysis is and its benefits
    st.markdown("""
    RFM (Recency, Frequency, Monetary) analysis segments customers based on their purchasing behavior. It helps businesses identify high-value customers, understand their preferences, and tailor marketing strategies to enhance customer loyalty and maximize revenue.
    """)

    # Compute RFM scores for the customers using the given DataFrame 'df'
    rfm_data = compute_rfm(df)

    # Perform clustering on the RFM data to segment customers and calculate the silhouette score
    rfm_data, silhouette_avg = perform_clustering(rfm_data)

    # Plot the RFM clusters to visualize the customer segments
    plot_rfm_clusters(rfm_data)

    # Print a results table displaying the RFM data and the silhouette average score
    print_rfm_results_table(rfm_data, silhouette_avg)

def run_client_similarity_analysis(df, interaction_matrix):
    # Display a subheader for the Client Similarity Analysis section in a Streamlit app
    st.subheader("Client Similarity Analysis")

    # Provide a markdown description explaining what Client Similarity Analysis is and its benefits
    st.markdown("""
    Client similarity analysis identifies groups of clients with similar purchasing behavior. By segmenting clients based on their preferences and habits, businesses can create targeted marketing campaigns, personalized recommendations, and improve customer satisfaction.
    """)

    # Create interaction matrix if not provided
    if interaction_matrix is None:
        interaction_matrix = create_interaction_matrix(df)

    # Compute client-to-client recommendations based on the interaction matrix
    result_table = client_to_client_recommender(interaction_matrix)

    # Organize the results table for display
    organized_table = []
    for client_id, recommended_clients, recall_at_k in result_table:
        organized_table.append([client_id, ", ".join(map(str, recommended_clients)), recall_at_k])

    # Define column headers for the table
    headers = ["Client ID", "Similar Clients", "Similarity Score"]

    # Display the organized table in a Streamlit app
    st.table(pd.DataFrame(organized_table, columns=headers))

def run_rfm_trends_analysis(df):
    # Display a subheader for the RFM Trends Analysis section in a Streamlit app
    st.subheader("RFM Trends Analysis")

    # Provide a markdown description explaining what RFM trends analysis is and its benefits
    st.markdown("""
    This analysis examines the RFM trends of clients over time and clusters them based on their RFM trend patterns. It provides insights into customer behavior and identifies segments of clients with similar RFM trend trajectories.
    """)

    # Preprocess the RFM data from the provided DataFrame 'df'
    rfm = preprocess_rfm_data(df)

    # Cluster clients based on their RFM trends and calculate angular coefficients
    angular_coefficients_df = cluster_clients(rfm)

    # Plot the results of the clustering to visualize the different clusters
    plot_cluster_results(angular_coefficients_df)

    # Print a table displaying the clustering results, including the angular coefficients
    print_clustering_results_table(angular_coefficients_df)

    # Forecast future RFM trends for the clients based on their clustering results
    forecast_rfm_trends(rfm, angular_coefficients_df)

def run_product_analysis(df):
    # Display a subheader for the Product Analysis section in a Streamlit app
    st.subheader("Product Analysis")

    # Compute RFM scores for products using the given DataFrame 'df'
    product_rfm_data = compute_product_rfm(df)

    # Perform clustering on the product RFM data to segment products and calculate the silhouette score
    product_rfm_data, silhouette_avg = perform_product_clustering(product_rfm_data)

    # Plot the product clusters to visualize the product segments
    plot_product_clusters(product_rfm_data)

    # Print a results table displaying the product RFM data and the silhouette average score
    print_product_results_table(product_rfm_data, silhouette_avg)

def run_abc_analysis(df):
    # Display a subheader for the ABC Analysis section in a Streamlit app
    st.subheader("ABC Analysis")

    # Provide a markdown description explaining what ABC analysis is and its benefits
    st.markdown("""
    ABC analysis categorizes products or clients into three groups (A, B, and C) based on their importance or contribution to the business. It helps in prioritizing resources and optimizing inventory management.
    """)

    # Calculate sales for each product using the given DataFrame 'df'
    product_sales = calculate_product_sales(df)

    # Classify each product into ABC categories based on their sales
    product_sales['ABC_Class'] = product_sales.apply(classify_product, axis=1)

    # Plot the clustering of products based on their ABC classification
    plot_product_clustering(product_sales)

    # Add reorder points and sales frequency information to the product sales data
    product_sales = add_reorder_and_frequency(product_sales)

    # Print the final results table displaying the product sales data with ABC classification, reorder points, and sales frequency
    print_results(product_sales)

def run_sales_performance_analysis(df, product_features):
    # Display a subheader for the Sales Performance Analysis section in a Streamlit app
    st.subheader("Sales Performance Analysis")

    # Provide a markdown description explaining what Sales Performance Analysis is and its benefits
    st.markdown("""
    This analysis focuses on evaluating the performance of products based on their sales data. It includes RFM analysis, clustering of products, and forecast of sales trends using linear regression.
    """)

    # Plot clusters of products based on their features
    sales_plot_clusters(product_features)

    # Print the product features data in a tabular format
    sales_print_product_features(product_features)

    # Group the sales data by 'Product_ID'
    grouped_data = df.groupby('Product_ID')

    # Calculate RFM scores for the products using the given DataFrame 'df'
    rfm_scores = sales_calculate_rfm_scores(df)

    # Plot the sales forecast for the top 'num_products' products using linear regression
    sales_plot_forecast(grouped_data, rfm_scores, num_products=10)

def run_roi_roe_analysis(df):
    # Display a subheader for the ROI/ROE Analysis section in a Streamlit app
    st.subheader("ROI/ROE Analysis")

    # Provide a markdown description explaining what ROI/ROE analysis is and its benefits
    st.markdown("""
    This analysis focuses on evaluating the monthly metrics such as ROI (Return on Investment) and ROE (Return on Equity) for each product based on their sales data.
    """)

    # Calculate monthly ROI and ROE metrics for each product using the given DataFrame 'df'
    monthly_data = roi_calculate_monthly_metrics(df)

    # Display a subheader for the Monthly Metrics Table section in a Streamlit app
    st.subheader("Monthly Metrics Table")

    # Print the calculated monthly ROI and ROE metrics in a tabular format
    roi_print_table(monthly_data)

    # Display a subheader for the Monthly Metrics Plots section in a Streamlit app
    st.subheader("Monthly Metrics Plots")

    # Plot the monthly ROI and ROE metrics to visualize the performance trends over time
    roi_plot_metrics(monthly_data)

def run_churn_analysis(df):
    # Display a subheader for the client retention analysis section in the Streamlit app
    st.subheader("Client Retention Analysis")

    # Display markdown text explaining what client retention analysis is and how churn risk is quantified
    st.markdown("""
    Client retention analysis identifies customers who are likely to stop doing business with the company.
    The metric "Churn Risk" quantifies the likelihood that a customer will discontinue their relationship with a company within a specific time period. This metric is expressed as a value between 0 and 1:

    1. A churn risk of 0 means that the customer is highly unlikely to leave the company.
    This typically suggests strong customer satisfaction, loyalty, and engagement.

    2. A churn risk of 1 means that there is a considerable probability that the customer may leave the company.
    This could be due to dissatisfaction, poor service, better alternatives, or other factors driving them away.

    By predicting churn, businesses can take proactive measures to retain customers and reduce churn rates.
    """)

    # Load and preprocess the churn data using a custom function, which returns a cleaned DataFrame
    df_churn = load_and_preprocess_churn_data(df)

    # Define the features to be used for the churn analysis
    features = ['Total_Monetary_Value', 'Total_Discount', 'Unique_Products', 'Days_Since_Last_Purchase']
    # Define the target variable for the churn analysis
    target = 'Churn'

    # Train a Random Forest model using the preprocessed data and specified features and target
    rf = train_rf_model(df_churn, features, target)

    # Use the trained model to predict churn risk for the customers in the DataFrame
    predict_churn_risk(rf, df_churn, features)

    # Plot the churn risk for the customers to visualize the results of the analysis
    plot_churn_risk(df_churn)

def run_sales_anomaly_detection(df):
    # Display a subheader for the client retention analysis section in the Streamlit app
    st.subheader("Sales Anomalies")

    # Display markdown text explaining what client retention analysis is and how churn risk is quantified
    st.markdown("""
    We utilize a machine learning methodology leveraging Generative Adversarial Networks (GANs) to identify anomalies within sales data. The insights provided by our algorithm regarding anomalous sales may hold significance in fraud detection investigations. It's important to note that while our system flags certain sales as anomalous, not all anomalies indicate fraudulent activity. However, in cases where suspicion arises, the data generated by our system could prove instrumental in understanding unusual behaviors.
    """)

    # Ensure that 'Product_ID' and 'Client_ID' columns exist in the DataFrame
    if 'Product_ID' not in df.columns or 'Client_ID' not in df.columns:
        st.error("DataFrame does not contain 'Product_ID' or 'Client_ID' columns.")
        return

    # Standardize the features 'Product_ID' and 'Client_ID'
    scaler = StandardScaler()
    df[['Feature1', 'Feature2']] = scaler.fit_transform(df[['Product_ID', 'Client_ID']])

    # Initialize and train the GAN model
    generator = build_generator()
    discriminator = build_discriminator()
    compile_discriminator(discriminator)
    combined = create_combined_model(generator, discriminator)
    train_gan(combined, discriminator, generator, df, epochs=10, batch_size=64)

    # Detect anomalies
    anomalies_df = detect_anomalies(generator, df)

    # Print anomalies
    print_anomalies(anomalies_df)

    # Plot anomalies
    plot_anomalies(anomalies_df)
