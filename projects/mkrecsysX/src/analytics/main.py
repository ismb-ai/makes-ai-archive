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
    # Display the introduction section
    display_intro()

    # Upload a CSV file and process the data if a file is uploaded
    file = upload_csv_file()

    if file is not None:
        # Process the uploaded data to obtain the DataFrame and RFM scores
        df, rfm_scores = process_data(file)

        # Create product features based on the processed DataFrame and RFM scores
        product_features = create_product_features(df, rfm_scores)

        # Initialize interaction matrix as None
        interaction_matrix = None

        # Select the analysis tab to determine which analysis to perform
        selected_tab = select_analysis_tab()

        # Perform the selected analysis based on the chosen tab
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

if __name__ == "__main__":
    main()
