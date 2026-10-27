# hybrid exponential smoothing plus CNN

# Importando as bibliotecas necessárias
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from tensorflow.keras.layers import Conv1D, MaxPooling1D

# Função para preparar os dados para o modelo CNN
def preparar_dados(dados, janela):
    X, y = [], []
    for i in range(len(dados) - janela - 1):
        X.append(dados[i:(i + janela), 0])
        y.append(dados[i + janela, 0])
    return np.array(X), np.array(y)

# Função para criar a CNN
def criar_CNN(janela):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(janela, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Ajustando o modelo de suavização exponencial
model_exp_smooth = SimpleExpSmoothing(training_data['sum_quant_item']).fit(smoothing_level=0.2, optimized=False)
training_data['suav_exponencial'] = model_exp_smooth.fittedvalues

# Preparação dos dados para o modelo CNN
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(training_data['suav_exponencial'].values.reshape(-1, 1))
janela = 3
X, y = preparar_dados(dataset, janela)

# Criando e ajustando o modelo CNN
model_cnn = criar_CNN(janela)
model_cnn.fit(X.reshape((X.shape[0], X.shape[1], 1)), y, epochs=200, verbose=0)

# Fazendo a previsão com a CNN
previsao_cnn = model_cnn.predict(X.reshape((X.shape[0], X.shape[1], 1)))
previsao_cnn = scaler.inverse_transform(previsao_cnn)

# Plotando os dados originais e a previsão do modelo
plt.figure(figsize=(15,8))
plt.plot(training_data['time_scale'].values, training_data['sum_quant_item'].values, label='Original data')
plt.plot(training_data['time_scale'].values[3:-1], previsao_cnn, label='CNN prediction')
plt.legend()
plt.show()
