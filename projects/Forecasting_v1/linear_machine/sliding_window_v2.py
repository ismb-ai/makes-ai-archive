# sliding_window_v2.py
#
# Este código define e treina um modelo de regressão linear usando uma abordagem 
# de janela deslizante para prever séries temporais. Primeiro, as variáveis de 
# tempo e quantidade são extraídas dos dados de treinamento e convertidas em 
# arrays unidimensionais. Em seguida, define-se uma função que implementa a 
# abordagem de janela deslizante, treinando o modelo de regressão em cada janela 
# de dados e fazendo previsões para o horizonte seguinte. Este processo é repetido 
# continuamente, deslocando a janela de treinamento ao longo do conjunto de dados.
# Ao final, o código plota dois gráficos: um que compara as previsões do modelo 
# com a série temporal original e outro que mostra a tendência do erro quadrático 
# médio ao longo das diferentes janelas.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Definição das variáveis 'window_size' e 'horizon'.
# Ajuste conforme a necessidade.

window_size = 50
horizon = 10

# Preparação dos dados: As variáveis X e y são definidas como 'time_scale' e
# 'sum_quant_item', respectivamente. Ambas são reformatadas para um array 1-D.

X = training_data['time_scale'].values.reshape(-1,1)
Y = training_data['sum_quant_item'].values.reshape(-1,1)

# 'rolling_window_approach(X, Y, window_size, horizon)'
# A função rolling_window_approach() recebe as variáveis X e Y, o tamanho da 
# janela e o horizonte como parâmetros. A função inicia um objeto de regressão 
# linear. Em seguida, inicializa duas listas vazias para armazenar as previsões
# e os valores de MSE.

def rolling_window_approach(X, Y, window_size, horizon):
    """
    This function trains a model by shifting the window of training data and making predictions.
    """
    regressor = LinearRegression()
    predictions, mse_values = [], []
    
    # O seguinte laço percorre o índice do primeiro elemento até o último elemento menos o
    # horizonte (para garantir que sempre haja um horizonte para previsão). 
    #
    # Para cada iteração do laço:
    #
    # - Define as janelas de treinamento e teste.
    # - Treina o modelo de regressão com os dados de treinamento.
    # - Faz a previsão para a janela de teste.
    # - Adiciona a primeira previsão à lista 'predictions' e calcula o erro 
    # médio quadrático (MSE) entre os valores previstos e reais.
    # - Adiciona o MSE calculado à lista 'mse_values'.
    
    for i in range(window_size, len(X) - horizon):
        X_train = X[i-window_size:i] 
        Y_train = Y[i-window_size:i]
        X_test = X[i:i+horizon] 
        regressor.fit(X_train, Y_train.ravel())
        Y_pred = regressor.predict(X_test)
        predictions.append(Y_pred[0]) # Armazena apenas a primeira previsão
        mse = mean_squared_error(Y[i:i+1], Y_pred[:1]) 
        mse_values.append(mse)
    
    # Finalmente, a função retorna as listas 'predictions' e 'mse_values'.
    return predictions, mse_values

# Aplicação da função 'rolling_window_approach' para obter as previsões e os valores de MSE.
predictions, mse_values = rolling_window_approach(X, Y, window_size, horizon)

# Traçado dos gráficos

# O primeiro gráfico compara as previsões do modelo de regressão em janela com a série temporal original. 
plt.plot(X[window_size:len(X)-horizon], Y[window_size:len(X)-horizon], color='lightcoral', label='Original time series')
plt.plot(X[window_size:len(X)-horizon], predictions, color='blue', label='Windowed Linear Regression model')
plt.grid(True)
plt.xlabel('time_scale')
plt.ylabel('sum_quant_item')
plt.legend()
plt.title('Windowed Linear Regression model vs Original time series')
plt.show()

# O segundo gráfico mostra o MSE ao longo das diferentes janelas.plt.figure(figsize=(12,6))
plt.figure(figsize=(12,6))
plt.plot(X[window_size:len(X)-horizon], mse_values, color='green')
plt.grid(True)
plt.xlabel('end point of each window')
plt.ylabel('mean squared error')
plt.title('MSE trend over the windowed Linear Regression models')
plt.show()

