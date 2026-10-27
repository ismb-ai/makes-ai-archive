# cnn_lstm.py
#
# O código acima implementa uma abordagem híbrida para modelagem de séries 
# temporais, utilizando redes neurais convolucionais (CNN) e o método 
# "Long Short-Term Memory" (LSTM). Primeiro, o conjunto de dados é 
# escalonado e reformatado com a função 'create_dataset' para ser compatível
# com o modelo LSTM. Em seguida, um modelo de rede neural sequencial 
# contendo uma camada convolucional, uma camada LSTM e uma camada de 
# saída é criado utilizando a função 'create_model'. Este modelo é treinado 
# ao longo de 30 épocas com dados de treino e validação.
# Após o treinamento, o modelo é usado para prever os valores desejados 
# para os conjuntos de treinamento e validação. As perdas durante o 
# treinamento e as previsões são visualizadas utilizando as funções 
# 'plot_history' e 'plot_prediction'.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten

# create_dataset
# A função create_dataset cria a estrutura de dados de entrada,
# que são pré-processados para criar sequências adequadas para treinar 
# um modelo de série temporal.

def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    X = X.to_numpy()  # Convert the Pandas Series X to a Numpy array
    y = y.to_numpy()  # Convert the Pandas Series y to a Numpy array
    for i in range(0, len(X) - time_steps, step):
        v = X[np.newaxis,i:(i + time_steps), np.newaxis]
        Xs.append(v)        
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# create_model
# A função create_model cria a arquitetura para o modelo de rede neural.
# O modelo é sequencial, composto por uma camada convolucional envolvida em uma 
# camada TimeDistributed, uma camada de pooling máximo envolvida em outra camada 
# TimeDistributed, uma camada LSTM, uma camada dropout (para regularização) e 
# uma camada densa de saída.

def create_model(time_steps):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, time_steps, 1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
  
    return model

# A função plot_history plota a métrica de perda.
# A função plot_prediction plota os valores reais em relação às previsões feitas
# pelo modelo.

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_prediction(y, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y, 'b', label="History")
    plt.plot(y_pred, 'r', label="Prediction")
    plt.title('Prediction vs Actual value')
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()

# O MinMaxScaler é usado para normalizar o 'sum_quant_item' no conjunto de dados
# de entrada.
scaler = MinMaxScaler(feature_range=(0, 1))
training_data['sum_quant_item_scaled'] = scaler.fit_transform(training_data['sum_quant_item'].values.reshape(-1,1))

# Em seguida, a função create_dataset é usada com 'sum_quant_item' normalizado 
# como recursos e alvo. O comprimento da sequência é 10, e a mudança é 1.
X, y = create_dataset(training_data['sum_quant_item_scaled'], training_data['sum_quant_item_scaled'], 10, 1)
y = y.reshape((y.shape[0], 1))

# Os conjuntos de dados de treinamento e validação são criados ao cortar as 
# primeiras 1000 e as próximas 300 instâncias dos arrays de recursos e alvos
# normalizados.
X_train, X_val = X[0:1000], X[1000:1300]
y_train, y_val = y[0:1000], y[1000:1300]

# Após o treinamento, as previsões são feitas nos conjuntos de dados de
# treinamento e validação. Essas previsões, juntamente com as perdas ao longo
# das épocas, são traçadas em um gráfico, chamando as funções plot_history e 
# plot_prediction. 
model = create_model(10)
history = model.fit(X_train, y_train, epochs=30, verbose=1, validation_data=(X_val, y_val), batch_size=32)
train_predict = model.predict(X_train)
val_predict = model.predict(X_val)

# Gráficos
plot_history(history)
plot_prediction(y_train, train_predict)
plot_prediction(y_val, val_predict)
