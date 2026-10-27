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

class ClientProductRecSys:
    def __init__(self, file_path):
        self.df = self.load_data(file_path)
        self.interaction_matrix = self.create_interaction_matrix(self.df)
        self.train_data, self.test_data = train_test_split(self.interaction_matrix.values, test_size=0.2, random_state=42)
        self.autoencoder_model = self.build_autoencoder_model(input_size=len(self.interaction_matrix.columns))
        self.encoder_model = Model(inputs=self.autoencoder_model.input, outputs=self.autoencoder_model.get_layer(index=1).output)
        self.embeddings = self.get_embeddings(self.train_data)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['Product_ID'] = df.groupby('Product_Name')['Product_ID'].transform('first')
        excluded_substrings = ['DEERE', 'oleo', 'KIT', 'PAINEL', 'INSTALACAO', 'SERV', 'CAPAS', 'SERVICOS', 'ICMS', 'CAPA', 'DIVERSOS']
        excluded_brands = ['DEERE']
        df = df[~df['Product_Name'].str.contains('|'.join(excluded_substrings), case=False)]
        df = df[~df['Brand_Name'].str.contains('|'.join(excluded_brands), case=False)]
        return df

    def create_interaction_matrix(self, data_frame):
        return data_frame.groupby(['Client_ID', 'Product_ID']).size().unstack(fill_value=0)

    def build_autoencoder_model(self, input_size):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(input_size,)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(input_size, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_autoencoder(self, epochs=10, batch_size=32):
        self.autoencoder_model.fit(self.train_data, self.train_data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(self.test_data, self.test_data))

    def get_embeddings(self, X):
        return self.encoder_model.predict(X)

    def recommend_products_autoencoder(self, client_id, k=5):
        client_index = self.interaction_matrix.index.get_loc(client_id)
        client_embedding = self.embeddings[client_index]
        similarities = np.dot(self.embeddings, client_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(client_embedding))
        valid_indices = np.where(similarities.argsort()[::-1] < len(self.interaction_matrix.columns))[0]
        product_indices = similarities.argsort()[::-1][valid_indices][:k]
        client_purchases = self.interaction_matrix.loc[client_id]
        purchased_product_ids = client_purchases[client_purchases > 0].index.values
        recommended_products = [product_id for product_id in self.interaction_matrix.columns[product_indices].values if product_id not in purchased_product_ids][:k]
        return recommended_products

class ProductClientRecSys:
    def __init__(self, interaction_matrix, embeddings):
        self.interaction_matrix = interaction_matrix
        self.embeddings = embeddings

    def recommend_clients_to_products(self, product_id, k=5):
        product_index = self.interaction_matrix.columns.get_loc(product_id)
        product_embedding = self.embeddings[:, product_index]  # Extracting the product embedding
        transposed_embeddings = self.embeddings.T
        recommendations = []
        for client_embedding in transposed_embeddings:
            similarity = np.dot(client_embedding, product_embedding)
            recommendations.append(similarity)
        recommendations = np.array(recommendations)
        valid_indices = np.where(recommendations.argsort()[::-1] < len(self.interaction_matrix.columns))[0]
        client_indices = recommendations.argsort()[::-1][valid_indices][:k]
        return self.interaction_matrix.index[client_indices]

class RFMAnalysis:
    def __init__(self, file_path):
        self.df = self.load_data(file_path)
        if self.df is not None:
            self.rfm_data = self.compute_rfm()
            self.silhouette_avg = None

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                st.warning("The uploaded file is empty.")
                return None
            df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])  # Convert Sales_Date to datetime
            return df
        except pd.errors.EmptyDataError:
            st.warning("The uploaded file is empty.")
            return None
        except pd.errors.ParserError:
            st.error("Failed to parse the file. Please ensure it is in CSV format.")
            return None

    def compute_rfm(self):
        if self.df is not None:
            current_date = self.df['Sales_Date'].max()
            rfm_data = self.df.groupby('Client_ID').agg({
                'Sales_Date': lambda x: (current_date - x.max()).days,  # Recency
                'Product_ID': 'count',  # Frequency
                'Monetary_Value': 'sum'  # Monetary
            }).reset_index()
            rfm_data['Recency'] = pd.qcut(rfm_data['Sales_Date'], q=10, labels=False, duplicates='drop') + 1
            rfm_data['Frequency'] = pd.qcut(rfm_data['Product_ID'], q=10, labels=False, duplicates='drop') + 1
            rfm_data['Monetary'] = pd.qcut(rfm_data['Monetary_Value'], q=10, labels=False, duplicates='drop') + 1
            rfm_data['RFM_Score'] = rfm_data['Recency'] + rfm_data['Frequency'] + rfm_data['Monetary']
            rfm_data['RFM_Score_Normalized'] = (rfm_data['RFM_Score'] - rfm_data['RFM_Score'].min()) / (rfm_data['RFM_Score'].max() - rfm_data['RFM_Score'].min()) * 100
            return rfm_data
        else:
            return None

    def perform_clustering(self):
        if self.rfm_data is not None:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.rfm_data['Cluster'] = kmeans.fit_predict(self.rfm_data[['RFM_Score_Normalized']])
            self.silhouette_avg = silhouette_score(self.rfm_data[['RFM_Score_Normalized']], self.rfm_data['Cluster'])
            cluster_silhouette_scores = silhouette_samples(self.rfm_data[['RFM_Score_Normalized']], self.rfm_data['Cluster'])
            self.rfm_data['Silhouette'] = cluster_silhouette_scores
        else:
            st.error("RFM data is not available for clustering.")

    def plot_rfm_clusters(self):
        if self.rfm_data is not None:
            plt.figure(figsize=(10, 6))
            colors = {0: '#4C72B0', 1: '#55A868', 2: '#C44E52'}
            for cluster, color in colors.items():
                cluster_data = self.rfm_data[self.rfm_data['Cluster'] == cluster]
                plt.scatter(cluster_data['RFM_Score_Normalized'], cluster_data['Client_ID'], color=color, label=f'Cluster {cluster}')
                cluster_mean_silhouette = cluster_data['Silhouette'].mean()
                plt.errorbar(cluster_mean_silhouette, max(cluster_data['Client_ID']), xerr=cluster_data['Silhouette'].std(), fmt='o', color='black')
            plt.xlabel('RFM Score (Normalized)')
            plt.ylabel('Client ID')
            plt.title('Clustering of Clients based on RFM Scores')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.error("RFM data is not available for plotting.")

    def print_rfm_results_table(self):
        if self.rfm_data is not None:
            cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
            self.rfm_data['RFM_Label'] = self.rfm_data['Cluster'].map(cluster_labels)
            table_data = self.rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Silhouette']]
            table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
            st.write("RFM Clustering Results:")
            st.dataframe(table_data)
            if self.silhouette_avg is not None:
                st.write(f"Average Silhouette Score: {self.silhouette_avg}")
        else:
            st.error("RFM data is not available to print results.")

    def print_rfm_results_table(self):
        if self.rfm_data is not None:
            cluster_labels = {0: "Diamond", 1: "Gold", 2: "Silver"}
            self.rfm_data['RFM_Label'] = self.rfm_data['Cluster'].map(cluster_labels)
            table_data = self.rfm_data[['Client_ID', 'RFM_Score_Normalized', 'RFM_Label', 'Silhouette']]
            table_data = table_data.sort_values(by='RFM_Score_Normalized', ascending=False)
            st.write("RFM Clustering Results:")
            st.dataframe(table_data)
            if self.silhouette_avg is not None:
                st.write(f"Average Silhouette Score: {self.silhouette_avg}")
        else:
            st.error("RFM data is not available to print results.")

class FrequentlyBoughtItemsRecSys:
    def __init__(self, interaction_matrix):
        self.interaction_matrix = interaction_matrix

    def instantiate_similarity_matrix(self):
        self.similarity_matrix = cosine_similarity(self.interaction_matrix.T)

    def get_similar_products(self, product_id, k=5):
        product_index = self.interaction_matrix.columns.get_loc(product_id)
        similar_indices = self.similarity_matrix[product_index].argsort()[:-k-1:-1]
        similar_products = self.interaction_matrix.columns[similar_indices].tolist()[1:]
        return similar_products

    def recommend_products_for_product(self, product_id, k=5):
        similar_products = self.get_similar_products(product_id, k)
        relevant_items = self.interaction_matrix[self.interaction_matrix[product_id] > 0].index.tolist()
        recommended_relevant = sum(1 for item in similar_products if item in relevant_items)
        recall_at_k = len(relevant_items)
        return similar_products, recall_at_k

def main():
    st.title("Makes RecSys X Advanced")

    # Home page description
    st.write("""
    Welcome to Makes RecSys X Advanced!

    This sophisticated web application offers advanced recommendation systems and RFM analysis tailored for our esteemed clientele.
    Upload your CSV file to embark on a journey of exploring cutting-edge recommendation techniques and gaining insightful business intelligence.
    """)

    # Generate unique keys for file uploaders
    upload_key = "Upload CSV File"

    # Upload CSV file
    file = st.sidebar.file_uploader(upload_key, key="file_uploader", type=["csv"])
    if file is not None:
        # Create instances of recommendation systems and analysis
        client_product_rec_sys = ClientProductRecSys(file_path=file)
        product_client_rec_sys = ProductClientRecSys(interaction_matrix=client_product_rec_sys.interaction_matrix, embeddings=client_product_rec_sys.embeddings)
        rfm_analysis = RFMAnalysis(file_path=file)
        frequently_bought_items_rec_sys = FrequentlyBoughtItemsRecSys(interaction_matrix=client_product_rec_sys.interaction_matrix)

        # Create tabs
        tabs = ["Recommendations: Clients to Products",
                "Recommendations: Products to Clients",
                "RFM Analysis",
                "Frequently Bought Items"]
        selected_tab = st.sidebar.radio("Navigation", tabs)

        # Recommendations: Clients to Products
        if selected_tab == "Recommendations: Clients to Products":
            st.subheader("Client-Centric Product Recommendations")
            st.write("""
            This section provides discerning clients with personalized recommendations, meticulously crafted based on their historical product interactions.
            Leveraging sophisticated autoencoder techniques, our system discerns latent features to furnish tailored product suggestions of utmost relevance.
            """)
            for client_id in client_product_rec_sys.interaction_matrix.index:
                with st.expander(f"Client {client_id} Recommendations"):
                    recommended_products = client_product_rec_sys.recommend_products_autoencoder(client_id, k=15)
                    actual_products = client_product_rec_sys.interaction_matrix.loc[client_id]
                    recall_at_k = len(set(recommended_products).intersection(actual_products[actual_products > 0].index)) / len(actual_products[actual_products > 0])
                    table_data = []
                    for product_id in recommended_products:
                        product_name = client_product_rec_sys.df.loc[client_product_rec_sys.df['Product_ID'] == product_id, 'Product_Name'].iloc[0]
                        brand = client_product_rec_sys.df.loc[client_product_rec_sys.df['Product_ID'] == product_id, 'Brand_Name'].iloc[0]
                        table_data.append([product_id, product_name, brand])
                    st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                    st.dataframe(pd.DataFrame(table_data, columns=["Product ID", "Product Name", "Brand"]))

        # Recommendations: Products to Clients
        elif selected_tab == "Recommendations: Products to Clients":
            st.subheader("Product-Focused Client Recommendations")
            st.write("""
            This segment offers refined recommendations tailored for each product, pinpointing prospective clients likely to exhibit interest.
            By identifying target clientele, it empowers businesses with precise targeting strategies, optimizing marketing endeavors.
            """)
            for product_id in client_product_rec_sys.interaction_matrix.columns:
                with st.expander(f"Product {product_id} Recommendations"):
                    recommended_clients = product_client_rec_sys.recommend_clients_to_products(product_id, k=5)
                    if len(recommended_clients) > 0:
                        recall_at_k = len(set(recommended_clients).intersection(client_product_rec_sys.interaction_matrix.index)) / len(client_product_rec_sys.interaction_matrix.index)
                        table_data = []
                        for client_id in recommended_clients:
                            table_data.append([client_id])
                        st.write(f"Recall at K = 5: {recall_at_k:.2f}")
                        st.dataframe(pd.DataFrame(table_data, columns=["Client ID"]))
                    else:
                        st.write("No recommendations for this product.")

        # RFM Analysis
        elif selected_tab == "RFM Analysis":
            st.subheader("RFM Analysis: Unveiling Client Segmentation")
            st.write("""
            This section conducts meticulous RFM (Recency, Frequency, Monetary) analysis to unveil nuanced client segments.
            By clustering clients based on their behavior, it provides actionable insights, enabling tailored marketing strategies for diverse clientele.
            """)
            rfm_data = rfm_analysis.compute_rfm()
            rfm_data, silhouette_avg = rfm_analysis.perform_clustering(rfm_data)
            rfm_analysis.plot_rfm_clusters(rfm_data)
            rfm_analysis.print_rfm_results_table(rfm_data, silhouette_avg)

        # Frequently Bought Items
        elif selected_tab == "Frequently Bought Items":
            st.subheader("Frequently Bought Items Recommender System")
            st.write("""
            This cutting-edge recommender system suggests frequently bought items analogous to each product, amplifying cross-selling and upselling initiatives.
            By identifying correlated product pairs, it augments overall sales performance, fostering sustainable business growth.
            """)
            frequently_bought_items_rec_sys.instantiate_similarity_matrix()

            # Display recommendations for each product
            for product_id in client_product_rec_sys.interaction_matrix.columns:
                similar_products, recall_at_k = frequently_bought_items_rec_sys.recommend_products_for_product(product_id, k=5)
                st.subheader(f"Recommendations for Product ID: {product_id} (Recall at K: {recall_at_k:.2f})")
                recommended_product_info = client_product_rec_sys.df[client_product_rec_sys.df['Product_ID'].isin(similar_products)][['Product_ID', 'Product_Name', 'Brand_Name']].drop_duplicates()
                st.dataframe(recommended_product_info)

if __name__ == "__main__":
    main()
