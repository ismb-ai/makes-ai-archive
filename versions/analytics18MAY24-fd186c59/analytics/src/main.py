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
from sklearn.ensemble import IsolationForest
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
    # Display the introduction to the program.
    display_intro()

    # Upload a CSV file and store the returned file object.
    file = upload_csv_file()

    # If a file is uploaded successfully:
    if file is not None:
        # Process the data from the uploaded file, including computing RFM scores.
        df, rfm_scores = process_data(file)

        # Create product features based on the processed data and RFM scores.
        product_features = create_product_features(df, rfm_scores)

        # Initialize the interaction matrix as None.
        interaction_matrix = None

        # Select the analysis tab to determine the type of analysis to perform.
        selected_tab = select_analysis_tab()

        # Based on the selected tab, run the corresponding analysis function.
        if selected_tab == "Recommendations: Clients to Products":
            run_recommendations_clients_to_products(df, interaction_matrix)
        elif selected_tab == "Recommendations: Products to Clients":
            run_recommendations_products_to_clients(df, interaction_matrix)
        elif selected_tab == "RFM Analysis":
            run_rfm_analysis(df)
        elif selected_tab == "Client Similarity Analysis":
            run_client_similarity_analysis(df, interaction_matrix)
        elif selected_tab == "RFM Trends Analysis":
            run_rfm_trends_analysis(df)
        elif selected_tab == "Product Analysis":
            run_product_analysis(df)
        elif selected_tab == "ABC Analysis":
            run_abc_analysis(df)
        elif selected_tab == "Sales Performance":
            run_sales_performance_analysis(df, product_features)
        elif selected_tab == "ROI/ROE Analysis":
            run_roi_roe_analysis(df)
        elif selected_tab == "Client Retention Analysis":
            run_churn_analysis(df)
        elif selected_tab == "Sales Anomaly Detection":
            run_sales_anomaly_detection(df)
        elif selected_tab == "Product Experience Analysis":
            run_anomalous_products_and_recommendations(df)

if __name__ == "__main__":
    main()
