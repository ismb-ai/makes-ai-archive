# random_forests.py
#
# Esta implementação utiliza a técnica "Floresta Aleatória" (Random Florest)
# para fazer previsões em séries temporais. Primeiramente, prepara-se 
# os dados para a previsão, criando grupos contíguos chamados 'janelas' 
# e retornando um vetor-de-recursos e um vetor-de-alvos. Em seguida, 
# realiza-se uma validação cruzada, para dividir a série em conjuntos de 
# treino e teste, respeitando a ordem temporal dos dados. Posteriormente, 
# inicia-se a Floresta Aleatória, treinando o modelo e realizando previsões. 
# Valores previstos e os dados originais são armazenados, juntamente com o 
# Erro Médio Quadrático de cada previsão. Findamos traçando um gráfico dos 
# resultados obtidos.

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# prepare_data()
# Nesta função, a série temporal é preparada para a previsão. A função vai criar
# conjuntos de séries deslizantes chamadas "janelas" e retornar um vetor de 
# características X e um vetor de alvos y. A janela deslizante é um conjunto
# contíguo dos dados da série temporal recebida como entrada. A função 
# iterativamente move a janela pelos dados e captura a configuração da série.

def prepare_data(data, window_size, horizon):
    """Function to prepare data for time-series prediction"""
    X, y = list(), list()
    start = 0
    for _ in range(len(data)):
        end = start + window_size
        if end + horizon <= len(data):
            X.append(data[start:end])
            y.append(data[end:end+horizon])
        start += 1
    return np.array(X), np.array(y)


# Obtém a estrutura de dados da série temporal.
time_series = training_data['sum_quant_item'].values

# Ajuste o tamanho da janela e horizonte conforme a necessidade.
window_size = 10
horizon = 1

# Realização da validação cruzada:
# A divisão da série em conjuntos de treino e teste deve ser feita de maneira a
# respeitar a ordem temporal dos dados. Isto é feito com a classe 
# TimeSeriesSplit, do sklearn, que implementa um esquema de validação cruzada 
# para séries temporais.

X, y = prepare_data(time_series, window_size, horizon)
tscv = TimeSeriesSplit(n_splits=5)

# Definição do modelo:
# Inicia-se um modelo de Random Forest, com 100 árvores de decisão.

rf = RandomForestRegressor(n_estimators = 100)
predictions, actuals, mse = [], [], [] 

# Treinamento e previsão:
# Para cada divisão de treino e teste realizada pela validação cruzada, o modelo
# é treinado com o conjunto de treino e realiza previsões no conjunto de teste.
# Valores preditos e reais são armazenados, assim como o MSE de cada previsão.

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit model
    rf.fit(X_train, y_train.ravel())

    # Faz previsões
    prediction = rf.predict(X_test)
    predictions.extend(prediction)

    # Salva actuals
    actuals.extend(y_test.ravel())

    # Computa MSE
    mse.append(mean_squared_error(y_test, prediction))

# Plot the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(actuals, 'darksalmon', label='Actual values')
plt.plot(predictions, 'navy', label='Predictions')
plt.title('RF model Predictions vs Actual values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot the MSE
plt.figure(figsize=(12, 6))
plt.plot(mse)
plt.title('Trend of Mean Squared Error (MSE) through each fold')
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.show()
