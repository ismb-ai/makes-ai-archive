# cnn.py
#
# O código implementa um modelo de aprendizado de máquina para previsão de séries
# temporais usando Redes Neurais Convolucionais (CNN). Primeiro, definimos a janela
# de dados e o horizonte de previsão, e transforma os dados em uma sequência de janelas
# de pontos de dados. O núcleo do código é a construção e treinamento do modelo CNN,
# que é feito usando a biblioteca Keras. Após o treinamento, o modelo é usado para
# fazer previsões nos dados de treinamento. Além disso, o código executa uma análise
# visual dos resultados, comparando os dados de previsão com os dados originais em 
# um gráfico. Finalmente, é calculado o Erro Quadrático Médio (MSE), uma métrica de 
# avaliação do desempenho do modelo, que também é apresentado graficamente.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten
from sklearn.metrics import mean_squared_error

# Definindo a janela e o horizonte: A janela é o 
# número de pontos de dados passados que a rede usará para fazer 
# a previsão, e o horizonte é o número de pontos futuros que a 
# rede tentará prever.
window_size = 10
horizon = 1

# transform_series: Esta função 
# converterá a série de tempo em uma sequência de janelas de dados.
# A função irá iterar pela série e criar janelas de tamanho 'window_size'.
# Cada uma dessas janelas será armazenada em 'X' e o valor 
# correspondente de 'Y' será o ponto de dado no horizonte.
# Essa função é usada para transformar os dados de treinamento 
# 'sum_quant_item' de um Pandas DataFrame em uma matriz X e um 
# vetor Y.
def transform_series(series, window_size, horizon):
    X, Y = list(), list()
    length = len(series)
    for i in range(length):
        end = i + window_size
        out_end = end + horizon
        if out_end > length:
            break
        X.append(series[i:end])
        Y.append(series[end:out_end])
    return np.array(X), np.array(Y)

# Convertendo a série em uma sequência:
X, Y = transform_series(training_data['sum_quant_item'].values, 
                        window_size, horizon)

# Definindo o modelo CNN:
# A função "build_CNN_model" define a rede neural convolucional.
def build_CNN_model(window_size, horizon):
    model = Sequential()
    model.add(Conv1D(64, (3), activation='relu', 
                     input_shape=(window_size, 1)))
    model.add(Flatten())
    model.add(Dense(horizon))

    model.compile(optimizer='adam', loss='mse')
    return model

# Treinando o modelo:
# Em seguida, o código constroi e treina o modelo CNN. Utiliza o 
# modelo sequencial do Keras, adicionando as camadas de forma 
# sequencial. A primeira camada é uma camada de convolução 
# unidimensional (CNN), seguida por uma camada de achatamento que 
# converterá a saída da camada CNN em uma matriz unidimensional, 
# finalizando com uma camada densa que vai produzir a previsão final.
CNN_model = build_CNN_model(window_size, horizon)
CNN_model.fit(np.expand_dims(X, axis=2), Y, epochs=200, verbose=0)

# Predizendo valores:
# A variável "predictions" armazena as previsões do modelo.
predictions = CNN_model.predict(np.expand_dims(X, axis=2))

# Plotando a comparação entre o modelo e os dados originais
plt.figure(figsize=(10, 6))
plt.plot(training_data['sum_quant_item'].values, label='Original Data')
plt.plot(np.arange(window_size, len(predictions) + window_size), 
         predictions, color='r', label='Predicted Data')
plt.title('Comparação entre o modelo CNN e os dados originais')
plt.xlabel('Time Scale')
plt.ylabel('Sum Quantity Item')
plt.legend()
plt.show()

# Plotando a tendência de MSE
mse = [mean_squared_error(Y[i], predictions[i]) 
       for i in range(len(predictions))]
plt.figure(figsize=(10, 6))
plt.plot(mse, label='MSE Trend')
plt.title('Tendência de MSE ao longo do tempo')
plt.xlabel('Time Scale')
plt.ylabel('MSE')
plt.legend()
plt.show()
