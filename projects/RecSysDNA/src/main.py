# main.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# Web Interface:
import streamlit as st

# Data Analysis and Machine Learning:
from sklearn.model_selection import train_test_split
import pandas as pd

# Internal Modules:
from recsys import *
from ranksys import *
from profiler import *
from clusteranalysis import *

def main():
    """
    Main function to run the Product Recommendation, Profiling, and Ranking System application.
    This function sets up the Streamlit interface, handles file uploads, and navigates between different app modes.
    """
    # Set the title of the Streamlit application
    st.title("Product Recommendation, Profiling, and Ranking System")

    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data from the uploaded CSV file
        data = load_data(uploaded_file)

        # Define contextual columns for various analyses
        contextual_columns = [
            'weather_condition', 'operating_system', 'device_type', 'age_group', 'gender',
            'income_level', 'mood', 'activity', 'time_of_day', 'day_of_week', 'season'
        ]

        # Sidebar navigation for different app modes
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio("Choose the app mode", ["Recommendations", "Profiler", "Ranking System", "Cluster Analysis"])

        if app_mode == "Recommendations":
            # Recommendations mode
            if 'user_id' not in data.columns or 'product_id' not in data.columns:
                st.error("The uploaded file must contain 'user_id' and 'product_id' columns.")
                return

            # Construct interaction matrix for recommendations
            interaction_matrix = construct_interaction_matrix(data)
            contextual_features, label_encoders, scaler = prepare_contextual_features(data)

            # Flatten the interaction matrix
            interaction_flattened = interaction_matrix.stack().reset_index()
            interaction_flattened.columns = ['user_id', 'product_id', 'interaction_count']
            interaction_flattened['interaction_count'] -= interaction_flattened['interaction_count'].min()

            # Merge features and prepare training and testing sets
            X, y = merge_features(interaction_flattened, contextual_features)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Get unique values for contextual columns
            unique_values = get_unique_values(data, contextual_columns)

            # User filters for recommendations
            user_filters = {}
            for column in contextual_columns:
                user_filters[column] = st.selectbox(f"Select {column}", options=['any'] + list(unique_values[column]))

            # Sidebar sliders for contextual feature weights
            st.sidebar.title("Contextual Feature Weights")
            weights = {}
            for column in contextual_columns:
                weights[column] = st.sidebar.slider(f"Weight for {column}", min_value=0.0, max_value=1.0, value=1.0)

            # Filter data based on user selections
            filtered_data = filter_data(data, user_filters)

            # Train the recommendation model
            model = train_model(X_train, y_train, weights)

            # Generate recommendations
            recommendations = generate_recommendations(filtered_data, model, contextual_features, scaler, X_train.columns)

            # Calculate and display recall at 10
            true_interactions = interaction_flattened[interaction_flattened['interaction_count'] > 0]
            recall = recall_at_k(recommendations, true_interactions, k=10)
            st.write(f"Recall at 10: {recall}")

            # Display recommendations for each user
            for user_id, products in recommendations.items():
                with st.expander(f"Recommendations for User {user_id}", expanded=True):
                    st.markdown(f"<div style='display: flex; flex-wrap: wrap;'>", unsafe_allow_html=True)
                    for product in products:
                        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px;'>{product}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        elif app_mode == "Profiler":
            # Profiler mode
            df, label_encoders = profiler_encode_categorical(data)
            X, y = profiler_prepare_features_and_target(df)
            X = profiler_standardize_numerical(X)
            X_train, X_test, y_train, y_test = profiler_split_data(X, y)
            y_train, y_test = profiler_convert_target_to_categorical(y_train, y_test)
            model, input_layers = profiler_build_model(X, label_encoders, y_train)
            input_data_train, input_data_test = profiler_prepare_input_data(X, X_train, X_test)
            profiler_train_model(model, input_data_train, y_train)
            profiler_evaluate_model(model, input_data_test, y_test)
            profiler_determine_typical_profiles(df, label_encoders)

        elif app_mode == "Ranking System":
            # Ranking System mode
            st.write("Using a genetic algorithm to determine the importance of each feature in predicting product sales.")

            # Load and encode data
            data_encoded = ranksys_one_hot_encode(data, contextual_columns)
            toolbox = ranksys_initialize_deap(data_encoded)

            product_ids = data['product_id'].unique()
            selected_product_id = st.selectbox("Select Product ID", product_ids)

            if st.button("Run Genetic Algorithm"):
                data_encoded['sold'] = data['product_id'].apply(lambda x: 1 if x == selected_product_id else 0)
                y = data_encoded['sold']
                X = data_encoded.drop(columns=['user_id', 'product_id', 'timestamp', 'sold'])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                halloffame = ranksys_run_genetic_algorithm(toolbox, X_train, y_train, X_test, y_test)
                best_individual = halloffame[0]
                selected_features = [feature for feature, included in zip(X_train.columns, best_individual) if included]

                model = ranksys_train_model(X_train, y_train, selected_features)
                feature_importance_df = ranksys_get_feature_importances(model, selected_features)

                ranksys_create_feature_importance_table(selected_product_id, feature_importance_df)

        elif app_mode == "Cluster Analysis":
            # Cluster Analysis mode
            st.write("Performing Cluster Analysis...")

            # Preprocess data for cluster analysis
            data = clusterAnalysis_preprocess_data(data)
            X = data.drop(columns=['product_id'])
            y = data['product_id']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train models for cluster analysis
            rf_model, gbm_model = clusterAnalysis_train_models(X_train, y_train)
            feature_importance_df = clusterAnalysis_feature_importances(rf_model, gbm_model, X)

            # Select important features and perform clustering
            important_features = feature_importance_df[feature_importance_df['importance'] > 0.01]['feature']
            X_important = X[important_features]

            clusters = clusterAnalysis_kmeans_clustering(X_important)
            data['cluster'] = clusters
            data = clusterAnalysis_sort_clusters(data, clusters)

            # Display cluster analysis results
            st.write("Cluster Analysis Results:")
            clusterAnalysis_print_cluster_tables(data)
            clusterAnalysis_plot_clusters(X_important, data)

            # Train and display decision tree model for high sales events
            dt_model, X_high_sales = clusterAnalysis_train_decision_tree(data)
            clusterAnalysis_plot_decision_tree(dt_model, X_high_sales)
            clusterAnalysis_print_feature_importances(dt_model, X_high_sales)

if __name__ == "__main__":
    main()
