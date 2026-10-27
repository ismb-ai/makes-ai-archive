import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def load_data(file_path):
    """Load data from CSV."""
    return pd.read_csv(file_path)

def calculate_product_sales(data):
    """Calculate product sales."""
    product_sales = data.groupby('Product_ID')['Monetary_Value'].sum().reset_index()
    product_sales = product_sales.sort_values(by='Monetary_Value', ascending=False)
    product_sales['Cumulative_Percentage'] = product_sales['Monetary_Value'].cumsum() / product_sales['Monetary_Value'].sum() * 100
    return product_sales

def classify_product(row):
    """Classify products into A, B, and C categories based on cumulative percentage."""
    if row['Cumulative_Percentage'] <= 70:
        return 'A'
    elif row['Cumulative_Percentage'] <= 90:
        return 'B'
    else:
        return 'C'

def plot_product_clustering(product_sales):
    """Plot the clustering of products according to their ABC class."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Product_ID', y='Cumulative_Percentage', hue='ABC_Class', data=product_sales)
    plt.title('Clustering of Products by ABC Class')
    plt.xlabel('Product ID')
    plt.ylabel('Cumulative Percentage of Sales')
    plt.legend(title='ABC Class')
    plt.show()

def calculate_client_sales(data):
    """Calculate client sales."""
    client_sales = data.groupby('Client_ID')['Monetary_Value'].sum().reset_index()
    client_sales = client_sales.sort_values(by='Monetary_Value', ascending=False)
    client_sales['Cumulative_Percentage'] = client_sales['Monetary_Value'].cumsum() / client_sales['Monetary_Value'].sum() * 100
    return client_sales

def classify_client(row):
    """Classify clients into A, B, and C categories based on cumulative percentage."""
    if row['Cumulative_Percentage'] <= 70:
        return 'A'
    elif row['Cumulative_Percentage'] <= 90:
        return 'B'
    else:
        return 'C'

def plot_client_clustering(client_sales):
    """Plot the clustering of clients according to their ABC class."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='ABC_Class', data=client_sales, order=['A', 'B', 'C'])
    plt.title('Clustering of Clients by ABC Class')
    plt.xlabel('ABC Class')
    plt.ylabel('Number of Clients')
    plt.show()

def add_reorder_and_frequency(product_sales):
    """Add re-order point and delivery frequency to the dataframe."""
    reorder_points = {'A': 5000, 'B': 2000, 'C': 1000}
    delivery_frequencies = {'A': 'Weekly', 'B': 'Bi-Weekly', 'C': 'Monthly'}
    product_sales['Reorder_Point'] = product_sales['ABC_Class'].map(reorder_points)
    product_sales['Delivery_Frequency'] = product_sales['ABC_Class'].map(delivery_frequencies)
    return product_sales

def print_results(product_sales):
    """Print the results in a formatted table."""
    print("ABC Classification of Products with Suggested Re-order Point and Delivery Frequency:")
    print(tabulate(product_sales, headers='keys', tablefmt='pretty'))

if __name__ == "__main__":
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    data = load_data(file_path)

    product_sales = calculate_product_sales(data)
    product_sales['ABC_Class'] = product_sales.apply(classify_product, axis=1)
    plot_product_clustering(product_sales)

    client_sales = calculate_client_sales(data)
    client_sales['ABC_Class'] = client_sales.apply(classify_client, axis=1)
    plot_client_clustering(client_sales)

    product_sales = add_reorder_and_frequency(product_sales)
    print_results(product_sales)

