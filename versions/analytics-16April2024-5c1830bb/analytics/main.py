# main.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This file contains the main routine for our business analytics tool, wherein
# one finds: (i) a client and product-centric recommender system, (ii) an RFM
# and ABC analysis, (iii) a forecast and clustering tool for the time-evolution
# of product's and client's RFM, and (iv) a ROI/ROE analysis. The tool is imple-
# mented in the framework of streamlit's library in the format of a web app.

# The following document is organized into four sections. In sec. 1, we define
# functions responsible to load and process user's data. In sec. 2, we declare
# the routines that builds the user's interface. In sec. 3, one finds the
# executive functions that calls the core computations of our analytic engine.
# Finally, sec. 4 contains the main routine.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
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
from webinterface import *

def main():
    # Display the introductory section in the Streamlit app
    display_intro()

    # Allow the user to upload a CSV file through a file uploader in the Streamlit sidebar
    file = upload_csv_file()

    # If a file is uploaded, proceed with the data processing and analysis
    if file is not None:
        # Process the uploaded CSV file to obtain the DataFrame 'df' and RFM scores 'rfm_scores'
        df, rfm_scores = process_data(file)

        # Create product features from the processed DataFrame 'df' and RFM scores 'rfm_scores'
        product_features = create_product_features(df, rfm_scores)

        # Initialize the interaction matrix as None
        interaction_matrix = None

        # Allow the user to select an analysis tab from the available options
        selected_tab = select_analysis_tab()

        # Perform the corresponding analysis based on the selected tab
        if selected_tab == "Recommendations: Clients to Products":
            # Run recommendations analysis for clients to products
            run_recommendations_clients_to_products(df, interaction_matrix)
        elif selected_tab == "Recommendations: Products to Clients":
            # Run recommendations analysis for products to clients
            run_recommendations_products_to_clients(df, interaction_matrix)
        elif selected_tab == "RFM Analysis":
            # Run RFM (Recency, Frequency, Monetary) analysis
            run_rfm_analysis(df)
        elif selected_tab == "Client Similarity Analysis":
            # Run client similarity analysis
            run_client_similarity_analysis(df, interaction_matrix)
        elif selected_tab == "RFM Trends Analysis":
            # Run RFM trends analysis over time
            run_rfm_trends_analysis(df)
        elif selected_tab == "Product Analysis":
            # Run product analysis
            run_product_analysis(df)
        elif selected_tab == "ABC Analysis":
            # Run ABC (Activity-Based Costing) analysis
            run_abc_analysis(df)
        elif selected_tab == "Sales Performance":
            # Run sales performance analysis
            run_sales_performance_analysis(df, product_features)
        elif selected_tab == "ROI/ROE Analysis":
            # Run ROI (Return on Investment) and ROE (Return on Equity) analysis
            run_roi_roe_analysis(df)

if __name__ == "__main__":
    main()
