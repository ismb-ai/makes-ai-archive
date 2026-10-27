import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages

# Function to read and preprocess data
def read_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'])
    return data

# Function to calculate RFM scores
def calculate_rfm(data):
    recency_df = data.groupby('Product_ID')['Sales_Date'].max().reset_index()
    recency_df.columns = ['Product_ID', 'Last_Purchase_Date']
    recency_df['Recency'] = (data['Sales_Date'].max() - recency_df['Last_Purchase_Date']).dt.days

    frequency_df = data.groupby('Product_ID')['Sales_Date'].count().reset_index()
    frequency_df.columns = ['Product_ID', 'Frequency']

    monetary_df = data.groupby('Product_ID')['Monetary_Value'].sum().reset_index()
    monetary_df.columns = ['Product_ID', 'Monetary']

    rfm_df = recency_df.merge(frequency_df, on='Product_ID')
    rfm_df = rfm_df.merge(monetary_df, on='Product_ID')

    return rfm_df

# Function to resample data and create time series of RFM scores
def resample_rfm(data):
    data.set_index('Sales_Date', inplace=True)
    daily_data = data.groupby('Product_ID').resample('D').agg({
        'Monetary_Value': 'sum',
        'Client_ID': 'count'
    }).reset_index()

    daily_data['Monetary_Value'].fillna(0, inplace=True)
    daily_data['Client_ID'].fillna(0, inplace=True)

    return daily_data

def daily_rfm(daily_data):
    rfm_time_series = {}

    for product in daily_data['Product_ID'].unique():
        product_data = daily_data[daily_data['Product_ID'] == product].copy()
        product_data['Recency'] = (product_data['Sales_Date'].max() - product_data['Sales_Date']).dt.days
        product_data['Frequency'] = product_data['Client_ID'].cumsum()
        product_data['Monetary'] = product_data['Monetary_Value'].cumsum()
        rfm_time_series[product] = product_data[['Sales_Date', 'Recency', 'Frequency', 'Monetary']]

    return rfm_time_series

# TCN Model and Dataset class
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, output_size, kernel_size)
        )

    def forward(self, x):
        return self.tcn(x)

class RFMDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return x, y

# Function to train TCN model for a product
def train_tcn_model(product_data):
    input_size = 3  # R, F, M
    output_size = 3  # R, F, M
    num_channels = 10
    kernel_size = 2
    dropout = 0.2
    seq_len = 30  # length of the input sequence

    model = TCN(input_size, output_size, num_channels, kernel_size, dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = RFMDataset(product_data[['Recency', 'Frequency', 'Monetary']].values, seq_len)

    if len(dataset) <= 0:
        return None

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    epochs = 20
    for epoch in range(epochs):
        for x, y in dataloader:
            x = x.permute(0, 2, 1).float()  # reshape for TCN: (batch_size, input_size, seq_len)
            y = y.float().unsqueeze(0)
            optimizer.zero_grad()
            output = model(x)
            output = output[:, :, -1]  # get the last output for comparison
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    return model

# Function to cluster data
def cluster_data(rfm_df):
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_features)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(scaled_rfm)
    rfm_df['Cluster'] = kmeans.labels_
    rfm_df['Profitability'] = rfm_df['Cluster'].map({0: 'Low', 1: 'Average', 2: 'High'})
    return rfm_df

# Function to visualize the results and save them to a PDF
def visualize_results(tcn_models, rfm_time_series, rfm_df, pdf_file_path):
    with PdfPages(pdf_file_path) as pdf:

        for product, model in tcn_models.items():
            product_data = rfm_time_series[product]
            x = product_data[['Recency', 'Frequency', 'Monetary']].values
            x = torch.tensor(x).permute(1, 0).unsqueeze(0).float()

            with torch.no_grad():
                forecast = model(x).squeeze().permute(1, 0).numpy()

            forecast = forecast[:len(product_data['Sales_Date'])]  # ensure forecast length matches Sales_Date length

            plt.figure(figsize=(15, 5))
            plt.plot(product_data['Sales_Date'], product_data['Recency'], label='Actual Recency')
            plt.plot(product_data['Sales_Date'][:len(forecast)], forecast[:, 0], label='Forecasted Recency', linestyle='--')
            plt.plot(product_data['Sales_Date'], product_data['Frequency'], label='Actual Frequency')
            plt.plot(product_data['Sales_Date'][:len(forecast)], forecast[:, 1], label='Forecasted Frequency', linestyle='--')
            plt.plot(product_data['Sales_Date'], product_data['Monetary'], label='Actual Monetary')
            plt.plot(product_data['Sales_Date'][:len(forecast)], forecast[:, 2], label='Forecasted Monetary', linestyle='--')
            plt.title(f'RFM Time Series for Product {product}')
            plt.legend()
            pdf.savefig()
            plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=rfm_df, x='Recency', y='Monetary', hue='Profitability', style='Profitability', palette='viridis')
        plt.title('Product Clustering Based on RFM Scores')
        plt.xlabel('Recency')
        plt.ylabel('Monetary')
        plt.legend()
        pdf.savefig()
        plt.close()

# Function to save RFM table to a text file
def save_rfm_table_to_txt(rfm_df, txt_file_path):
    rfm_table = rfm_df[['Product_ID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Profitability']]
    table_string = tabulate(rfm_table, headers='keys', tablefmt='grid')
    with open(txt_file_path, 'w') as f:
        f.write(table_string)

# Main execution function
def main(file_path, pdf_file_path, txt_file_path):
    data = read_and_preprocess_data(file_path)
    rfm_df = calculate_rfm(data)
    daily_data = resample_rfm(data)
    rfm_time_series = daily_rfm(daily_data)

    tcn_models = {}
    for product, product_data in rfm_time_series.items():
        model = train_tcn_model(product_data)
        if model is not None:
            tcn_models[product] = model

    rfm_df = cluster_data(rfm_df)
    visualize_results(tcn_models, rfm_time_series, rfm_df, pdf_file_path)
    save_rfm_table_to_txt(rfm_df, txt_file_path)

# File path and main function call
file_path = '/Users/igormol/Desktop/RecSysDNA/sample/Vaz.csv'
pdf_file_path = 'rfm_analysis_results.pdf'
txt_file_path = 'rfm_analysis_table.txt'
main(file_path, pdf_file_path, txt_file_path)
