import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import squarify
import base64
import io
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.mixture import GaussianMixture

# Load and preprocess data
def load_and_preprocess_data(data):
    data['Client_ID'].fillna('Unknown', inplace=True)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')

    # Drop rows with invalid Sales_Date
    data = data.dropna(subset=['Sales_Date'])

    today = datetime.now()
    data['Recency'] = (today - data['Sales_Date']).dt.days
    data['Monetary_Value'] = pd.to_numeric(data['Monetary_Value'], errors='coerce')

    # Drop rows with invalid Monetary_Value
    data = data.dropna(subset=['Monetary_Value'])

    rfm = data.groupby('Client_ID').agg({
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

    # Check for any remaining non-numeric values
    if rfm[['Recency', 'Frequency', 'Monetary_Value']].isnull().values.any():
        raise ValueError("Non-numeric values found in RFM columns after conversion.")

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

# Build the VAE model
def build_vae(input_dim, latent_dim):
    inputs = Input(shape=(input_dim,))
    h = Dense(64, activation='relu')(inputs)
    h = Dense(32, activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_h = Dense(32, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(inputs, x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss(inputs, x_decoded_mean))
    vae.compile(optimizer='rmsprop')
    encoder = Model(inputs, z_mean)
    return vae, encoder

# Perform DEC clustering
def perform_dec_clustering(rfm, n_clusters=4, latent_dim=2, epochs=50, batch_size=256):
    features = rfm[['R_Score', 'F_Score', 'M_Score']].values

    vae, encoder = build_vae(features.shape[1], latent_dim)
    vae.fit(features, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    encoded_features = encoder.predict(features)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag')
    rfm['Cluster'] = gmm.fit_predict(encoded_features)
    return rfm, gmm

# Assign cluster names
def assign_cluster_names(rfm):
    cluster_labels = {
        0: "Clientes Valiosos",
        1: "Clientes Recentes",
        2: "Clientes Frequentes",
        3: "Clientes em Risco"
    }

    # Sort clusters by RFM score mean
    cluster_means = rfm.groupby('Cluster')[['R_Score', 'F_Score', 'M_Score']].mean().mean(axis=1).sort_values(ascending=False).index.tolist()
    cluster_labels_sorted = {cluster: cluster_labels[i] for i, cluster in enumerate(cluster_means)}

    rfm['Cluster'] = rfm['Cluster'].map(cluster_labels_sorted)
    return rfm

# Plot a treemap of the clusters
def plot_treemap(rfm):
    cluster_summary = rfm.groupby('Cluster').agg({
        'Client_ID': 'count',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean'
    }).rename(columns={'Client_ID': 'Count'}).reset_index()

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

# Calculate the silhouette score for each client
def calculate_silhouette_scores(rfm):
    features = rfm[['R_Score', 'F_Score', 'M_Score']].values
    cluster_labels = rfm['Cluster'].astype('category').cat.codes
    mean_silhouette_score = silhouette_score(features, cluster_labels)
    return rfm, mean_silhouette_score

# Function to download the RFM table as a CSV
def get_csv_download_link(dataframe, file_name='rfm_clusters.csv'):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
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
    st.title("Análise de RFM para Clientes")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        rfm = load_and_preprocess_data(data)
        rfm = assign_rfm_scores(rfm)
        rfm, gmm = perform_dec_clustering(rfm, n_clusters=4, latent_dim=2, epochs=50, batch_size=256)
        rfm = assign_cluster_names(rfm)
        rfm, mean_silhouette_score = calculate_silhouette_scores(rfm)

        # Creating tabs
        tab1, tab2 = st.tabs(["Treemap", "Tabela RFM"])

        with tab1:
            st.header("Treemap dos Clusters de Clientes")
            fig = plot_treemap(rfm)
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig), unsafe_allow_html=True)

        with tab2:
            st.header("Tabela RFM com Clusters")
            st.write(rfm)
            st.markdown(get_csv_download_link(rfm), unsafe_allow_html=True)

        st.subheader(f"Mean Silhouette Score: {mean_silhouette_score:.4f}")

if __name__ == "__main__":
    main()
