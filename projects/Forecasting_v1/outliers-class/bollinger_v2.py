# bollinger_v2.py
# Implementa-se o método de Bandas de Bollinger para detectar outliers.
# Quando o valor sai das Bandas de Bollinger, é considerado um outlier.
# Os outliers são substituídos pela banda superior ou inferior, 
# dependendo se estão acima ou abaixo da banda, respectivamente.

# Este código inclui funções necessárias para processar os dados 
# (`prepare_training_data()`), treinar o modelo de regressão linear
# (`train_linear_regression()`), calcular as Bandas de Bollinger 
# (`calculate_bollinger_bands()`), detectar e substituir outliers
# usando as Bandas de Bollinger (`detect_replace_outliers_bollinger()`), 
# e plotar os dados originais e limpos (`plot_data()`).

# Na função main(), essas funções são chamadas em ordem, e os dados limpos são salvos em um arquivo .csv.
# Substitua o DataFrame (`training_data`) conforme necessário antes de executar o código.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2

# prepare_training_data()
# Prepara os dados de treinamento selecionando as colunas
# ('time_scale' e 'sum_quant_item'), removendo linhas com valores faltantes e ordenando-os
# com base em 'time_scale'. Retorna os dados preparados.

def prepare_training_data():
    global training_data
    training_data = training_data[['time_scale','sum_quant_item']]
    training_data = training_data.dropna()
    training_data.sort_values(['time_scale'], axis = 0, inplace = True)  
    return training_data

# train_linear_regression(X, y)
# Treina um modelo de regressão linear com variável
# independente `X` e dependente `y`. Retorna o modelo treinado.

def train_linear_regression(X, y):
    model = LinearRegression()  
    model.fit(X, y)
    return model

# calculate_bollinger_bands(mean, std)
# Calcula as bandas superiores e inferiores de Bollinger 
# a partir da média e desvio padrão dos resíduos. Bandas = média ± 2 * desvio padrão.
# Retorna as bandas superior e inferior.

def calculate_bollinger_bands(mean, std):
    upper_band = mean + 2*std
    lower_band = mean - 2*std
    return upper_band, lower_band

# detect_replace_outliers_bollinger(X, y, training_data)
# Detecta e substitui outliers usando as bandas de Bollinger.
# Treina um modelo de regressão linear, calcula os resíduos e as bandas.
# Resíduos além das bandas são considerados outliers e substituídos pelas bandas.
# Retorna os dados limpos.

def detect_replace_outliers_bollinger(X, y, training_data):
    model = train_linear_regression(X, y)
    residuals = pd.Series(y.flatten()) - model.predict(X).flatten()

    residuals_mean = residuals.mean()
    residuals_std = residuals.std()

    upper_band, lower_band = calculate_bollinger_bands(residuals_mean, residuals_std)

    outliers = residuals[(residuals > upper_band) | (residuals < lower_band)]
    
    print('Number of outliers:', len(outliers))
    print('Outliers:\n', outliers)
    
    corrected_residuals = residuals.copy()
    corrected_residuals[residuals > upper_band] = upper_band
    corrected_residuals[residuals < lower_band] = lower_band

    cleaned_data = training_data.copy()
    cleaned_data['sum_quant_item'] = model.predict(X).flatten() + corrected_residuals.values
    return cleaned_data

# plot_data(original_data, cleaned_data)
# Plota dados originais em azul ('bo') e dados limpos em vermelho ('ro').

def plot_data(original_data, cleaned_data):
    plt.figure(figsize=(14, 6))
    plt.plot(original_data[['time_scale']], original_data[['sum_quant_item']], 'bo', label='Original')
    plt.plot(cleaned_data[['time_scale']], cleaned_data[['sum_quant_item']], 'ro', label='Cleaned')
    plt.legend()
    plt.show()

# Função principal que chama todas as outras funções. Prepara dados de treinamento,
# separa variáveis, detecta/substitui outliers, plota dados e os salva em um arquivo CSV.

def main():
    training_data = prepare_training_data()
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, 1].values
    
    cleaned_data = detect_replace_outliers_bollinger(X, y, training_data)
    plot_data(training_data, cleaned_data)

    cleaned_data.to_csv('cleaned1_data.csv', index=False)
    cleaned1_data = pd.read_csv('cleaned1_data.csv')
    
if __name__ == "__main__":
    main()
