# webinterface.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the routines responsible for building our web
# interface. In sec. 1, we declare functions to load and processes user's data.
# In sec. 2, we construct routines that defines the user's interface, such as
# page's layout, text boxes, buttons and tabs. Sec. 3, contains Executive
# functions that performs the business analysis for each tab.

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from tensorflow.keras.models import Model
from recsys import *
from rfm import *
from trends import *
from roi import *
from churnanalysis import *
from anomaly import *
from pxm import *
from bayes import *
from deepRecSys import *
from dynamicalRecSys import *

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
    st.title("Makes Business Analytics")
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
    tabs = ["Recomendações: Produtos para Clientes", "Recomendações: Clientes para Produtos", "Deep RecSys", "Dynamical RecSys", "Análise RFM",
            "Clientes Similares", "Produtos Similares", "Tendências de RFM", "Análise ABC",
            "Performance de Vendas", "Análise de Retorno de Investimento", "Análise de Retenção de Clientes", "Detecção de Anomalias em Vendas", "Estratégia de Vendas para Clientes em Risco"]
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

    # Ensure interaction matrix is not empty
    if interaction_matrix is None:
        interaction_matrix = create_interaction_matrix(df)
    else:
        # Print interaction matrix for debugging
        st.write("Interaction Matrix:", interaction_matrix)

    # Align product IDs in the interaction matrix
    interaction_matrix.columns = [f'product_{i}' for i in range(len(interaction_matrix.columns))]

    # Store the columns of the interaction matrix
    interaction_matrix_columns = interaction_matrix.columns.tolist()

    # Split interaction matrix into training and testing data
    train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)

    # Build and train the autoencoder model
    autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
    train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)

    # Debugging: Print message to confirm model training
    st.write("Autoencoder Model training completed.")

    # Extract embeddings from the encoder model
    encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)

    # Debugging: Print message to confirm embeddings generation
    st.write("Embeddings extracted successfully.")

    # Debugging: Print interaction matrix columns and embeddings shape
    st.write("Interaction Matrix Columns:", interaction_matrix_columns)
    st.write("Embeddings Shape:", embeddings.shape)

    # Find common products between interaction_matrix and embeddings
    embedding_product_ids = [f'product_{i}' for i in range(embeddings.shape[1])]
    common_products = list(set(interaction_matrix.columns) & set(embedding_product_ids))

    if not common_products:
        raise ValueError("No common products found between the interaction matrix and embeddings.")

    # Filter interaction_matrix to keep only the common products
    filtered_interaction_matrix = interaction_matrix[common_products]

    # Adjust embeddings to include only columns corresponding to common products
    common_product_indices = [embedding_product_ids.index(prod_id) for prod_id in common_products]
    filtered_embeddings = embeddings[:, common_product_indices]

    # Precompute recommendations for common products and store in a dictionary
    all_recommendations = {}
    for product_id in common_products:
        try:
            all_recommendations[product_id] = recommend_clients_to_products(filtered_embeddings, product_id, filtered_interaction_matrix, k=5)
        except ValueError as e:
            st.error(str(e))

    # Process and display recommendations for each product in common products
    for product_id, recommended_clients in all_recommendations.items():
        with st.expander(f"Product {product_id} Recommendations"):
            if not recommended_clients.empty:  # Check if recommended_clients is not empty
                recall_at_k = len(set(recommended_clients).intersection(filtered_interaction_matrix.index)) / len(filtered_interaction_matrix.index)
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

def run_anomalous_products_and_recommendations(df):
    # Detect anomalous products in the given DataFrame
    anomalous_products = pXm_detect_anomalous_products(df)

    # Recommend clients for anomalous products
    # The function pXm_recommend_clients_for_anomalous_products takes three parameters:
    # 1. The DataFrame containing product data
    # 2. The list of anomalous products detected
    # 3. None as there's no specific filter provided for recommending clients
    pXm_recommend_clients_for_anomalous_products(df, anomalous_products, None)

def run_client_risk_bayesian_analysis(df):
    st.subheader("Client Risk Bayesian Analysis")
    st.markdown("""
    The Client Risk Bayesian Analysis system integrates RFM scores and Churn Scores (calculated through Customer Retention Analysis) using a Bayesian Decision Tree approach to identify clients needing attention. An Autoencoder neural network is then applied to recommend products to these clients, enhancing their satisfaction and loyalty.""")

    churn_df = load_and_preprocess_churn_data(df)
    features = ['Total_Monetary_Value', 'Total_Discount', 'Unique_Products', 'Days_Since_Last_Purchase']
    target = 'Churn'
    rf_model = train_rf_model(churn_df, features, target)
    rfm_data = compute_rfm(df)
    churn_df['Churn_Risk'] = rf_model.predict_proba(churn_df[features])[:, 1]
    combined_df = pd.merge(rfm_data, churn_df[['Client_ID', 'Churn_Risk']], on='Client_ID')
    combined_df['RFM_Category'] = pd.qcut(combined_df['RFM_Score_Normalized'], q=5, labels=False)
    bayesian_tree = bayes_train_bayesian_decision_tree(combined_df, ['RFM_Score_Normalized', 'Churn_Risk'], 'RFM_Category')
    combined_df['Attention_Needed'] = bayesian_tree.predict(combined_df[['RFM_Score_Normalized', 'Churn_Risk']])
    bayes_display_results_table(combined_df)
    interaction_matrix = create_interaction_matrix(df)
    interaction_matrix.columns = [f'product_{i}' for i in range(len(interaction_matrix.columns))]
    interaction_matrix_columns = interaction_matrix.columns.tolist()
    X_train, X_test = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)
    autoencoder_model = build_autoencoder_model(X_train.shape[1])
    train_autoencoder(autoencoder_model, X_train, X_test, epochs=10, batch_size=32)
    encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
    embeddings = get_embeddings(encoder_model, interaction_matrix.values)
    st.write("Interaction Matrix Columns:", interaction_matrix_columns)
    st.write("Embeddings Shape:", embeddings.shape)
    embedding_product_ids = [f'product_{i}' for i in range(embeddings.shape[1])]
    common_products = list(set(interaction_matrix.columns) & set(embedding_product_ids))
    st.write("Common Products:", common_products)
    if not common_products:
        st.error("No common products found between the interaction matrix and embeddings. Please check your data preprocessing steps.")
        raise ValueError("No common products found between the interaction matrix and embeddings.")
    filtered_interaction_matrix = interaction_matrix[common_products]
    common_product_indices = [embedding_product_ids.index(prod_id) for prod_id in common_products]
    filtered_embeddings = embeddings[:, common_product_indices]
    clients_attention = combined_df[combined_df['Attention_Needed'] == 1]['Client_ID']
    recommendations = []
    for client_id in clients_attention:
        try:
            recommended_products = recommend_products_autoencoder(filtered_embeddings, client_id, filtered_interaction_matrix)
            recommendations.append({
                'Client ID': client_id,
                'Recommended Products': ', '.join(map(str, recommended_products))
            })
        except ValueError as e:
            st.error(str(e))
    recommendations_df = pd.DataFrame(recommendations)
    st.table(recommendations_df)

def run_deeprecsys(df):
    # Display a subheader and description for the Deep RecSys module
    st.subheader("Deep RecSys: Client-Product Recommendations")
    st.markdown("""
    This module leverages deep learning models and classical machine learning to provide client-product recommendations based on RFM scores and churn risk analysis.
    """)

    # Compute RFM scores and perform clustering
    rfm_data = compute_rfm(df)
    rfm_data, silhouette_avg = perform_clustering(rfm_data)
    # Plot RFM clusters and print clustering results
    plot_rfm_clusters(rfm_data)
    print_rfm_results_table(rfm_data, silhouette_avg)

    # Load and preprocess churn data, merge with RFM data
    churn_data = load_and_preprocess_churn_data(df)
    combined_data = pd.merge(rfm_data, churn_data, on='Client_ID')

    # Define features and target for RF model, train RF model
    features = ['Total_Monetary_Value', 'Total_Discount', 'Unique_Products', 'Days_Since_Last_Purchase']
    target = 'Churn'
    rf_model = train_rf_model(churn_data, features, target)
    # Predict churn risk using RF model
    combined_data = deeprecsys_predict_churn_risk(rf_model, combined_data, features)

    # Create interaction matrix and split into train/test sets
    interaction_matrix = create_interaction_matrix(df)
    X_train, X_test = train_test_split(interaction_matrix, test_size=0.2, random_state=42)
    # Build and train autoencoder model
    autoencoder_model = build_autoencoder_model(X_train.shape[1])
    train_autoencoder(autoencoder_model, X_train, X_test)
    # Get embeddings from autoencoder
    embeddings = get_embeddings(autoencoder_model, interaction_matrix.values)

    # Sidebar configuration for RFM and churn thresholds
    st.sidebar.header("RFM and Churn Configuration")
    rfm_threshold = st.sidebar.slider("Select RFM Score Threshold (0 to 100):", 0, 100, 50)
    churn_threshold = st.sidebar.slider("Select Churn Risk Threshold (0.0 to 1.0):", 0.0, 1.0, 0.5)
    selected_abc_class = select_abc_class()

    # Sidebar selection for client cluster
    cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
    selected_cluster = st.sidebar.selectbox("Select Client Cluster:", options=list(cluster_labels.values()), index=2)
    selected_cluster_index = {v: k for k, v in cluster_labels.items()}.get(selected_cluster, 2)

    # Filter clients based on thresholds and selected cluster
    filtered_clients = combined_data[(combined_data['RFM_Score_Normalized'] <= rfm_threshold) &
                                     (combined_data['Churn_Risk'] <= churn_threshold) &
                                     (combined_data['Cluster'] == selected_cluster_index)]

    # Display filtered client information
    st.write(f"Filtered Clients (RFM Score <= {rfm_threshold}, Churn Risk <= {churn_threshold}, Cluster = {selected_cluster}):")
    st.write(filtered_clients[['Client_ID', 'RFM_Score_Normalized', 'Churn_Risk', 'Cluster']])

    # Calculate ABC classification for products
    product_sales = calculate_product_sales(df)
    product_sales = add_abc_classification(product_sales)

    # Generate product recommendations for filtered clients
    recommendations = []
    for client_id in filtered_clients['Client_ID']:
        recommended_products = recommend_products_autoencoder(embeddings, client_id, interaction_matrix)
        filtered_recommended_products = product_sales[product_sales['Product_ID'].isin(recommended_products)]
        filtered_recommended_products = filtered_recommended_products[filtered_recommended_products['ABC_Class'] == selected_abc_class]
        recommendations.append({"Client_ID": client_id, "Recommended_Products": filtered_recommended_products['Product_ID'].tolist()})

    # Display product recommendations
    st.write("Product Recommendations for Filtered Clients:")
    recommendations_df = pd.DataFrame(recommendations)
    st.dataframe(recommendations_df)

def run_cluster_based_recommender(df):
    # Step 1: Read and preprocess the sales data
    # Display a subheader for client RFM trends analysis
    st.subheader("Client RFM Trends Analysis")

    # Preprocess the sales data to calculate RFM scores for clients
    rfm_client = preprocess_rfm_data(df)

    # Cluster the clients based on their RFM scores
    client_clusters_df = cluster_clients(rfm_client)

    # Plot the clustering results for clients
    dynamicalRecSys_plot_cluster_results(client_clusters_df, "Client_ID", "Client RFM Trends Clustering")

    # Print the clustering results table for clients
    print_clustering_results_table(client_clusters_df)

    # Step 2: Calculate and cluster the RFM trends for products
    # Display a subheader for product RFM trends analysis
    st.subheader("Product RFM Trends Analysis")

    # Calculate the RFM scores for products
    rfm_product = dynamicalRecSys_calculate_product_rfm_scores(df)

    # Cluster the products based on their RFM scores
    product_clusters_df = dynamicalRecSys_cluster_products(rfm_product)

    # Plot the clustering results for products
    dynamicalRecSys_plot_cluster_results(product_clusters_df, "Product_ID", "Product RFM Trends Clustering")

    # Print the clustering results table for products
    dynamicalRecSys_print_clustering_results_table(product_clusters_df)

    # Step 3: Allow user selection and recommend products to clients
    # Display a subheader for recommendations
    st.subheader("Recommendations")

    # Move selection boxes to the sidebar
    # Allow the user to select a client cluster from a dropdown menu
    client_cluster_choice = st.sidebar.selectbox("Choose a client cluster", client_clusters_df['Cluster_Label'].unique())

    # Allow the user to select a product cluster from a dropdown menu
    product_cluster_choice = st.sidebar.selectbox("Choose a product cluster", product_clusters_df['Cluster_Label'].unique())

    # Generate and display product recommendations for the selected client and product clusters
    dynamicalRecSys_recommend_products_to_clients(df, client_clusters_df, product_clusters_df, client_cluster_choice, product_cluster_choice)
