import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import squarify
import base64
import io

# Load and preprocess data
def load_and_preprocess_data(data):
    data['Brand_Name'].fillna('Unknown', inplace=True)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')

    # Drop rows with invalid Sales_Date
    data = data.dropna(subset=['Sales_Date'])

    today = datetime.now()
    data['Recency'] = (today - data['Sales_Date']).dt.days
    data['Monetary_Value'] = pd.to_numeric(data['Monetary_Value'], errors='coerce')

    # Drop rows with invalid Monetary_Value
    data = data.dropna(subset=['Monetary_Value'])

    rfm = data.groupby('Product_ID').agg({
        'Recency': 'min',
        'Sales_Date': 'count',  # Frequency
        'Monetary_Value': 'sum'  # Monetary
    }).rename(columns={'Sales_Date': 'Frequency'}).reset_index()
    return rfm

# Assign RFM scores
def assign_rfm_scores(rfm):
    # Ensure all necessary columns are numeric
    rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce').fillna(0)
    rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce').fillna(0)
    rfm['Monetary_Value'] = pd.to_numeric(rfm['Monetary_Value'], errors='coerce').fillna(0)

    # Calculate quantiles
    quantiles = rfm[['Recency', 'Frequency', 'Monetary_Value']].quantile(q=[0.2, 0.4, 0.6, 0.8]).to_dict()

    def r_score(x):
        if x <= quantiles['Recency'][0.2]:
            return 5
        elif x <= quantiles['Recency'][0.4]:
            return 4
        elif x <= quantiles['Recency'][0.6]:
            return 3
        elif x <= quantiles['Recency'][0.8]:
            return 2
        else:
            return 1

    def f_score(x):
        if x <= quantiles['Frequency'][0.2]:
            return 1
        elif x <= quantiles['Frequency'][0.4]:
            return 2
        elif x <= quantiles['Frequency'][0.6]:
            return 3
        elif x <= quantiles['Frequency'][0.8]:
            return 4
        else:
            return 5

    def m_score(x):
        if x <= quantiles['Monetary_Value'][0.2]:
            return 1
        elif x <= quantiles['Monetary_Value'][0.4]:
            return 2
        elif x <= quantiles['Monetary_Value'][0.6]:
            return 3
        elif x <= quantiles['Monetary_Value'][0.8]:
            return 4
        else:
            return 5

    rfm['R_Score'] = rfm['Recency'].apply(r_score)
    rfm['F_Score'] = rfm['Frequency'].apply(f_score)
    rfm['M_Score'] = rfm['Monetary_Value'].apply(m_score)
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Sort by RFM_Score in decreasing order
    rfm = rfm.sort_values(by='RFM_Score', ascending=False).reset_index(drop=True)
    return rfm

# Perform hierarchical clustering
def perform_hierarchical_clustering(rfm, n_clusters=4):
    features = rfm[['R_Score', 'F_Score', 'M_Score']].values

    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    rfm['Cluster'] = cluster.fit_predict(features)
    return rfm, cluster

# Assign cluster names
def assign_cluster_names(rfm):
    cluster_labels = {
        0: "Produtos Populares",
        1: "Produtos Recentes",
        2: "Produtos em Risco",
        3: "Produtos Descontinuados"
    }

    # Sort clusters by RFM score mean
    cluster_means = rfm.groupby('Cluster')[['R_Score', 'F_Score', 'M_Score']].mean().mean(axis=1).sort_values(ascending=False).index.tolist()
    cluster_labels_sorted = {cluster: cluster_labels[i] for i, cluster in enumerate(cluster_means)}

    rfm['Cluster'] = rfm['Cluster'].map(cluster_labels_sorted)
    return rfm

# Plot a treemap of the clusters
def plot_treemap(rfm):
    cluster_summary = rfm.groupby('Cluster').agg({
        'Product_ID': 'count',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean'
    }).rename(columns={'Product_ID': 'Count'}).reset_index()

    cluster_summary['Percentage'] = cluster_summary['Count'] / cluster_summary['Count'].sum() * 100
    labels = [
        f"{row['Cluster']}\n{row['Percentage']:.2f}%\nR: {row['R_Score']:.2f}\nF: {row['F_Score']:.2f}\nM: {row['M_Score']:.2f}"
        for index, row in cluster_summary.iterrows()
    ]

    sizes = cluster_summary['Count']
    color_palette = plt.cm.get_cmap('tab10')  # Use a distinct color palette
    colors = [color_palette(i) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax)
    ax.axis('off')
    return fig

# Calculate the silhouette score for each product
def calculate_silhouette_scores(rfm):
    features = rfm[['R_Score', 'F_Score', 'M_Score']].values
    silhouette_avg = silhouette_score(features, rfm['Cluster'])
    rfm['Silhouette_Score'] = silhouette_samples(features, rfm['Cluster'])
    return rfm, silhouette_avg

# Function to download dataframe as CSV
def get_csv_download_link(dataframe, file_name='rfm_table.csv'):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV file</a>'
    return href

# Function to download the treemap as an image
def get_image_download_link(fig, file_name='treemap.png'):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{file_name}">Download Treemap</a>'
    return href

# Main function
def main():
    st.title("Análise de RFM para Produtos")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        rfm = load_and_preprocess_data(data)
        rfm = assign_rfm_scores(rfm)
        rfm, cluster = perform_hierarchical_clustering(rfm, n_clusters=4)
        rfm = assign_cluster_names(rfm)
        rfm, mean_silhouette_score = calculate_silhouette_scores(rfm)

        # Creating tabs
        tabs = st.tabs(["Treemap", "Tabela RFM"])

        with tabs[0]:
            st.header("Treemap dos Clusters de Produtos")
            fig = plot_treemap(rfm)
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig), unsafe_allow_html=True)

        with tabs[1]:
            st.header("Tabela RFM com Clusters")
            st.write(rfm)
            st.markdown(get_csv_download_link(rfm), unsafe_allow_html=True)

        st.subheader(f"Mean Silhouette Score: {mean_silhouette_score:.4f}")

if __name__ == "__main__":
    main()
