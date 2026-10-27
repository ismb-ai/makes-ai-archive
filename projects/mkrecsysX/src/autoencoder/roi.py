import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def read_data(file_path):
    """Read CSV file and return DataFrame."""
    return pd.read_csv(file_path)

def calculate_monthly_metrics(df):
    """Calculate ROI and ROE per month for each product ID."""
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
    df['Total_Cost'] = df['Monetary_Value'] * (1 - df['Discount'] / 100)
    monthly_data = df.groupby([df['Sales_Date'].dt.to_period('M'), 'Product_ID']).agg({'Total_Cost': 'sum'}).reset_index()
    monthly_data = monthly_data.rename(columns={'Sales_Date': 'Month'})
    monthly_data['ROI'] = monthly_data.groupby('Product_ID')['Total_Cost'].pct_change()
    monthly_data['ROE'] = monthly_data.groupby('Product_ID')['Total_Cost'].diff()
    monthly_data.fillna(0, inplace=True)
    return monthly_data

def print_table(monthly_data):
    """Print table with ROI and ROE for each product ID."""
    for product_id, group in monthly_data.groupby('Product_ID'):
        print(f"Product ID: {int(product_id)}")
        print(group[['Month', 'ROI', 'ROE']].to_string(index=False))
        print()

def plot_metrics(monthly_data):
    """Plot ROI and ROE time series for each product ID."""
    for product_id, group in monthly_data.groupby('Product_ID'):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        for i, metric in enumerate(['ROI', 'ROE']):
            X = np.arange(len(group)).reshape(-1, 1)
            y = group[metric].values.reshape(-1, 1)

            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)

            # Plot original time series and regression model
            axs[i].plot(group['Month'].astype(str), y, label=f'{metric} - Original')
            axs[i].plot(group['Month'].astype(str), y_pred, linestyle='--', label=f'{metric} - Predicted')
            axs[i].set_xlabel('Month')
            axs[i].set_ylabel(metric)
            axs[i].legend()

        fig.suptitle(f'Product ID {int(product_id)}')
        plt.tight_layout()
        plt.show()

def main():
    file_path = "/Users/igormol/Desktop/recsys/data/sales-CPP-updated.csv"
    df = read_data(file_path)
    monthly_data = calculate_monthly_metrics(df)
    print_table(monthly_data)
    plot_metrics(monthly_data)

if __name__ == "__main__":
    main()

