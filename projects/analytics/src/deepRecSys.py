# deepRecSys.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# Data manipulation and analysis
import pandas as pd

# Streamlit for web interface
import streamlit as st

# RFM computations
from rfm import *

def deeprecsys_predict_churn_risk(rf, df, features):
    # Predict churn risk using the trained RF model and features
    df['Churn_Risk'] = rf.predict_proba(df[features])[:, 1]
    return df

def select_abc_class():
    # Sidebar selection for ABC class of products
    abc_classes = ["A", "B", "C"]
    selected_abc_class = st.sidebar.selectbox("Select ABC Class for Products:", abc_classes)
    return selected_abc_class

def add_abc_classification(product_sales):
    # Add ABC classification to product sales data
    product_sales['ABC_Class'] = product_sales.apply(classify_product, axis=1)
    return product_sales
