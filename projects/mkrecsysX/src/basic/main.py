# main.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This code contains the main routine of our recommender and RFM systems.
# The web interface is implemented within the framework of streamlit,
# wherein the user is invited to upload a CSV file from which our analysis is
# performed upon, the results of which are shown enclosed within tabs.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.cluster import KMeans
import numpy as np
from tabulate import tabulate
from client_product_recsys import *
from product_client_recsys import *
from rfm_analysis import *
from frequently_bought_items import *

def main():
    # Set the title for the Streamlit app
    st.title("Makes RecSys X Advanced")

    # Display the description of the application
    st.write("""
    Welcome to Makes RecSys X Advanced!
    This sophisticated web application offers advanced recommendation systems and RFM analysis tailored for our esteemed clientele.
    Upload your CSV file to embark on a journey of exploring cutting-edge recommendation techniques and gaining insightful business intelligence.
    """)

    # Generate unique keys for file uploaders
    upload_key = "Upload CSV File"

    # Upload CSV file
    file = st.sidebar.file_uploader(upload_key, key="file_uploader", type=["csv"])

    # If file is uploaded
    if file is not None:
        # Load data from the uploaded file
        df = load_data(file)
        # Create interaction matrix from the data
        interaction_matrix = create_interaction_matrix(df)

        # Train autoencoder
        train_data, test_data = train_test_split(interaction_matrix.values, test_size=0.2, random_state=42)
        autoencoder_model = build_autoencoder_model(input_size=len(interaction_matrix.columns))
        train_autoencoder(autoencoder_model, train_data, test_data, epochs=10, batch_size=32)
        encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=1).output)
        embeddings = get_embeddings(encoder_model, interaction_matrix.values)

        # Create tabs for different functionalities
        tabs = ["Recommendations: Clients to Products",
                "Recommendations: Products to Clients",
                "RFM Analysis",
                "Frequently Bought Items"]
        selected_tab = st.sidebar.radio("Navigation", tabs)

        # Recommendations: Clients to Products
        if selected_tab == "Recommendations: Clients to Products":
            # Display client-centric product recommendations
            st.subheader("Client-Centric Product Recommendations")
            st.write("""
            This section provides discerning clients with personalized recommendations, meticulously crafted based on their historical product interactions.
            Leveraging sophisticated autoencoder techniques, our system discerns latent features to furnish tailored product suggestions of utmost relevance.
            """)
            # Iterate over each client
            for client_id in interaction_matrix.index:
                # Expandable section for each client's recommendations
                with st.expander(f"Client {client_id} Recommendations"):
                    # Get recommendations for the client using autoencoder
                    recommended_products = recommend_products_autoencoder(embeddings, client_id, interaction_matrix, k=15)
                    actual_products = interaction_matrix.loc[client_id]
                    recall_at_k = len(set(recommended_products).intersection(actual_products[actual_products > 0].index)) / len(actual_products[actual_products > 0])
                    table_data = []
                    for product_id in recommended_products:
                        product_name = df.loc[df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                        brand = df.loc[df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                        table_data.append([product_id, product_name, brand])
                    st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                    st.dataframe(pd.DataFrame(table_data, columns=["Product ID", "Product Name", "Brand"]))

        # Recommendations: Products to Clients
        elif selected_tab == "Recommendations: Products to Clients":
            # Display product-focused client recommendations
            st.subheader("Product-Focused Client Recommendations")
            st.write("""
            This segment offers refined recommendations tailored for each product, pinpointing prospective clients likely to exhibit interest.
            By identifying target clientele, it empowers businesses with precise targeting strategies, optimizing marketing endeavors.
            """)
            # Iterate over each product
            for product_id in interaction_matrix.columns:
                # Expandable section for each product's recommendations
                with st.expander(f"Product {product_id} Recommendations"):
                    # Get recommendations for the product
                    recommended_clients = recommend_clients_to_products(embeddings, product_id, interaction_matrix, k=5)
                    if len(recommended_clients) > 0:
                        recall_at_k = len(set(recommended_clients).intersection(interaction_matrix.index)) / len(interaction_matrix.index)
                        table_data = []
                        for client_id in recommended_clients:
                            table_data.append([client_id])
                        st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                        st.dataframe(pd.DataFrame(table_data, columns=["Client ID"]))
                    else:
                        st.write("No recommendations for this product.")

        # RFM Analysis
        elif selected_tab == "RFM Analysis":
            # Display RFM analysis
            st.subheader("RFM Analysis: Unveiling Client Segmentation")
            st.write("""
            This section conducts meticulous RFM (Recency, Frequency, Monetary) analysis to unveil nuanced client segments.
            By clustering clients based on their behavior, it provides actionable insights, enabling tailored marketing strategies for diverse clientele.
            """)
            # Compute RFM scores and perform clustering
            rfm_data = compute_rfm(df)
            rfm_data, silhouette_avg = perform_clustering(rfm_data)
            plot_rfm_clusters(rfm_data)
            print_rfm_results_table(rfm_data, silhouette_avg)

        # Frequently Bought Items
        elif selected_tab == "Frequently Bought Items":
            # Display frequently bought items recommender system
            st.subheader("Frequently Bought Items Recommender System")
            st.write("""
            This cutting-edge recommender system suggests frequently bought items analogous to each product, amplifying cross-selling and upselling initiatives.
            By identifying correlated product pairs, it augments overall sales performance, fostering sustainable business growth.
            """)
            # Instantiate similarity matrix
            similarity_matrix = instantiate_similarity_matrix(interaction_matrix)

            # Display recommendations for each product
            for product_id in interaction_matrix.columns:
                similar_products, recall_at_k = recommend_products_for_product(similarity_matrix, interaction_matrix, product_id, k=5)
                st.subheader(f"Recommendations for Product ID: {product_id} (Recall at K: {recall_at_k:.2f})")
                recommended_product_info = df[df['Product_ID'].isin(similar_products)][['Product_ID', 'Product_Name', 'Brand_Name']].drop_duplicates()
                st.dataframe(recommended_product_info)

if __name__ == "__main__":
    main()
