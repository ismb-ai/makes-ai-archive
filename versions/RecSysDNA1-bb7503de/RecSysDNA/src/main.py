import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import multiprocessing
import numpy as np
from recsys import *
from ranksys import *
from profiler import *

def main():
    st.title("Product Recommendation, Profiling, and Ranking System")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        contextual_columns = ['weather_condition', 'operating_system', 'device_type', 'age_group', 'gender', 'income_level', 'mood', 'activity', 'time_of_day', 'day_of_week', 'season']

        # Sidebar for navigation
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio("Choose the app mode", ["Recommendations", "Profiler", "Ranking System"])

        if app_mode == "Recommendations":
            # Recommendations mode
            if 'user_id' not in data.columns or 'product_id' not in data.columns:
                st.error("The uploaded file must contain 'user_id' and 'product_id' columns.")
                return

            interaction_matrix = construct_interaction_matrix(data)
            contextual_features, label_encoders, scaler = prepare_contextual_features(data)

            interaction_flattened = interaction_matrix.stack().reset_index()
            interaction_flattened.columns = ['user_id', 'product_id', 'interaction_count']
            interaction_flattened['interaction_count'] -= interaction_flattened['interaction_count'].min()

            X, y = merge_features(interaction_flattened, contextual_features)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            unique_values = get_unique_values(data, contextual_columns)

            user_filters = {}
            for column in contextual_columns:
                user_filters[column] = st.selectbox(f"Select {column}", options=['any'] + list(unique_values[column]))

            # Adding sliders for contextual feature weights
            st.sidebar.title("Contextual Feature Weights")
            weights = {}
            for column in contextual_columns:
                weights[column] = st.sidebar.slider(f"Weight for {column}", min_value=0.0, max_value=1.0, value=1.0)

            filtered_data = filter_data(data, user_filters)

            model = train_model(X_train, y_train, weights)

            recommendations = generate_recommendations(filtered_data, model, contextual_features, scaler, X_train.columns)

            true_interactions = interaction_flattened[interaction_flattened['interaction_count'] > 0]
            recall = recall_at_k(recommendations, true_interactions, k=10)
            st.write(f"Recall at 10: {recall}")

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

if __name__ == "__main__":
    main()
