# Hybrid_CNN_GRU.py
#
# O código-fonte a seguir apresenta uma implementação de modelagem de séries 
# temporais utilizando uma abordagem híbrida, combinando Redes Neurais Convolu-
# cionais (CNN) e Unidades Recorrentes de Gated (GRU). Esta técnica é empregada 
# para prever futuros valores em uma série temporal representada pelos dados 
# em ``training_data''. Esta abordagem híbrida, combinando características 
# aprendidas por meio de convoluções com mecanismos recorrentes, é particular-
# mente útil para capturar padrões temporais complexos em séries temporais.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense
import seaborn as sns

# Extração e Normalização dos Dados:
# Extrai os dados de treinamento e realiza a normalização usando Min-Max Scaling.

training_set = training_data.iloc[:, 1:2].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Criação de Conjunto de Dados Temporais:
# Define uma função para criar pares de entrada e saída para o treinamento do 
# modelo, considerando o número de observações passadas (look_back), o horizonte 
# de previsão (forecast_horizon), e o tamanho do lote (batch_size).

def create_dataset(dataset, look_back=1, forecast_horizon=1, batch_size=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-forecast_horizon+1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back: i + look_back + forecast_horizon, 0])

    return np.array(dataX), np.array(dataY)

# Parâmetros e Criação do Conjunto de Dados Temporais:
# Especifica os parâmetros para a função e cria o conjunto de dados temporais.

look_back = 30
forecast_horizon = 1
batch_size = 32
X_train, y_train = create_dataset(training_set_scaled, look_back, forecast_horizon, batch_size)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Função para Criação do Modelo:
# Define uma função para construir o modelo, incluindo uma camada de convolução 
# (CNN) seguida por camadas GRU e uma camada densa de saída.

def create_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(GRU(units=50, return_sequences=True))
    model.add(GRU(units=50))
    model.add(Dense(units = forecast_horizon))

    return model

# Criação e Compilação do Modelo:
# Cria o modelo usando a função definida e o compila com o otimizador 'adam' e a 
# função de perda sendo o erro quadrático médio.

model = create_model()
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = 100, batch_size = batch_size, verbose=0)

# Treinamento do Modelo:
# Treina o modelo com os dados de treino, utilizando 100 épocas e um tamanho de 
# lote de 32.

y_pred = model.predict(X_train)
training_predicted = sc.inverse_transform(y_pred)

# Exibição do Gráfico:

sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
sns.lineplot(data=training_data["sum_quant_item"].values, color='red', label='Real Data')
sns.lineplot(data=training_predicted, color='blue', label='Predicted Data')
plt.title("Prediction", fontsize=20)
plt.xlabel("Time", fontsize=16)
plt.ylabel("Values", fontsize=16)
plt.legend(fontsize=10)
plt.show()
