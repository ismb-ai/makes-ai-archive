# bayes.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the routines for the system that employs a
# Bayesian Decision Tree, based on the RFM scores and "Churn Scores" of clients,
# in order to determine which customers requires attention. Then, we use an
# Autoencoder to recommend products to these customers.

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
import streamlit as st
from churnanalysis import *

def bayes_train_bayesian_decision_tree(df, features, target):
    # Extract features and target variable from the DataFrame
    X, y = df[features], df[target]
    # Create a pipeline with StandardScaler and Gaussian Naive Bayes
    model = make_pipeline(StandardScaler(), GaussianNB())
    # Fit the model to the data
    model.fit(X, y)
    # Return the trained model
    return model

def bayes_display_results_table(combined_df):
    # Display results table in Streamlit using Markdown format
    st.write(combined_df[['Client_ID', 'RFM_Score_Normalized', 'Churn_Risk', 'Attention_Needed']].to_markdown())
