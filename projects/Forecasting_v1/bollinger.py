import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

def calculate_bollinger_bands(df, column, window, horizon):
    """
    Function to calculate Bollinger bands
    'df' is the DataFrame, 'column' is the column to consider for band calculation
    'window' is the window size, 'horizon' is the horizon window for moving standard deviation
    """
    df = df.copy()
    df['Rolling_Mean'] = df[column].rolling(window=window).mean()
    df['Bollinger_High'] = df['Rolling_Mean'] + horizon * df[column].rolling(window=window).std()
    df['Bollinger_Low'] = df['Rolling_Mean'] - horizon * df[column].rolling(window=window).std()
    return df

def bollinger_outliers(df, sum_quant_item='sum_quant_item', bollinger_high='Bollinger_High', bollinger_low='Bollinger_Low'):
    """
    Function to remove outliers based on Bollinger Bands
    """
    no_outliers = df[(df[sum_quant_item] <= df[bollinger_high]) &
                     (df[sum_quant_item] >= df[bollinger_low])]
    return no_outliers

# Calculate Bollinger Bands
training_data = calculate_bollinger_bands(training_data, 'sum_quant_item', window=20, horizon=2)

# Create a new_data DataFrame by removing the outliers
new_data = bollinger_outliers(training_data)

# Plot the original data and cleaned data for comparison
fig, ax = plt.subplots(2, 1, figsize = (12, 8), sharex=True)

ax[0].plot(training_data['time_scale'], training_data['sum_quant_item'], label='Original data')
ax[0].plot(training_data['time_scale'], training_data['Bollinger_High'], label='Bollinger High', color='r')
ax[0].plot(training_data['time_scale'], training_data['Bollinger_Low'], label='Bollinger Low', color='r')
ax[0].legend()
ax[0].set_title('Original Data with Bollinger Bands')

ax[1].plot(new_data['time_scale'], new_data['sum_quant_item'], label='Cleaned data', color='g')
ax[1].legend()
ax[1].set_title('Cleaned Data (Outliers Removed)')

plt.tight_layout()
plt.show()
