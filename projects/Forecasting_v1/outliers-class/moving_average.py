# moving_average.py
# Calcula média móvel e desvio padrão em uma janela deslizante na série temporal.
# Sinaliza como outliers pontos de dados que se desviam da média móvel por um limite (z-score > 2).

# Ajuste o tamanho da janela deslizante para determinar média móvel na variável 'window'.
# Ajuste o threshold para classificação de outliers na variável 'z_threshold'.
 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# detect_outliers_moving_avg
# Calcula médias móveis da coluna 'sum_quant_item' com rolling().mean() do pandas.
# Calcula a diferença absoluta entre valores originais e a média móvel na coluna 'MAD'.
# Calcula o desvio padrão móvel da diferença absoluta na coluna 'MAD_std'.
# Identifica outliers quando 'MAD' excede a média mais 'z_threshold' desvios padrões móveis.

def detect_outliers_moving_avg(data, window, z_threshold=2):
    # Calculate moving averages
    data['MA'] = data['sum_quant_item'].rolling(window).mean()

    # Calculate absolute difference from moving average 
    data['MAD'] = np.abs(data['sum_quant_item'] - data['MA'])

    # Calculate rolling standard deviation
    data['MAD_std'] = data['MAD'].rolling(window).std()

    # Identify outliers
    data['outliers'] = np.where(data['MAD'] > data['MA'] + z_threshold*data['MAD_std'], 1, 0)

    return data

# plot_data()
# Plota dados originais, média móvel e outliers.

def plot_data(data):
    plt.figure(figsize=(10,6))

    # plot original data
    plt.plot(data['time_scale'], data['sum_quant_item'], 'k-', label='Original data')

    # plot moving average
    plt.plot(data['time_scale'], data['MA'], 'b-', label='Moving Average')

    # plot outliers
    plt.plot(data.loc[data.outliers==1, 'time_scale'], data.loc[data.outliers==1, 'sum_quant_item'], 'ro', label='Outliers')

    plt.legend()
    plt.title('Original Data vs Cleaned Data with Outliers')
    plt.show()

# apply_outlier_detection()
# Aplica "detect_outliers_moving_avg" com 'window' e 'z_threshold'.
# Extrai outliers.
# Imprime número de outliers e dados correspondentes.
# Plota dados, média móvel e outliers.
# Remove outliers, substituindo por np.nan e usando dropna() para removê-los do dataframe.    

def apply_outlier_detection(window, z_threshold, training_data):
    training_data_with_outliers = detect_outliers_moving_avg(training_data.copy(), window, z_threshold)
    outliers = training_data_with_outliers.loc[training_data_with_outliers.outliers == 1]
    print('Number of outliers detected: ', len(outliers))
    print('Outliers:\n', outliers)
    plot_data(training_data_with_outliers)
    
    cleaned_data = training_data.copy()
    cleaned_data['sum_quant_item'] = np.where(training_data['time_scale'].isin(outliers['time_scale']), np.nan, cleaned_data['sum_quant_item'])
    cleaned_data.dropna(inplace=True)
    return cleaned_data

window = 20; # size of the moving window for moving average calculation
z_threshold = 2; # z-score threshold for outlier detection
cleaned1_data = apply_outlier_detection(window, z_threshold, training_data)
