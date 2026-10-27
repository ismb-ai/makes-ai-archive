# Anomaly_Detection_Hybrid_RNN_LSTM.py
#
# O código-fonte a seguir apresenta uma implementação de uma técnica de aprendi-
# zagem profunda para modelar uma série temporal, detectar e substituir anoma-
# lias na mesma, combinando uma abordagem de janelas deslizantes e uma Rede Neu-
# ral Recorrente (RNN) do tipo Long-Short Term Memory (LSTM).
#     A função `create_train_model(data)` utiliza uma rede neural recorrente 
# Long Short-Term Memory para modelar a série e retorna o modelo treinado.
#     O método `detect_replace_anomalies(model, series, window_size, sigma=1.0)`
# detecta as anomalias nos dados, considerando anomalias pontos que se encontram 
# além de um número `sigma` de desvios padrão da média em uma janela deslizante. 
#     As anomalias detectadas são substituídas pelos valores previstos pelo mo-
# delo treinado. A função `plot_data(orig_data, cleaned_data, anomalies)` cria 
# um gráfico para visualizar os dados originais, os dados limpos e as anomalias 
# substituídas. No código principal, a série de dados é extraída do dataframe 
# `training_data`, o modelo é treinado e, em seguida, é usado para detectar e 
# substituir as anomalias. Finalmente, os dados originais e as anomalias são 
# plotados e os dados limpos são salvos em um arquivo CSV chamado 
# 'cleaned_data.csv'. 
#    Portanto, este código apresenta uma abordagem eficaz para a detecção e 
# substituição de anomalias em dados de séries temporais utilizando redes neu-
# rais LSTM.

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Função para criar e treinar o modelo LSTM:
# Esta função cria e treina uma rede LSTM (Long Short-Term Memory). Primeiro, os
# dados são normalizados para estarem entre 0 e 1. Em seguida, são remodelados 
# para o formato necessário para a rede LSTM, ou seja, amostras, etapas de tempo 
# e recursos. Um modelo sequencial é criado usando camadas LSTM e Dense do K
# eras. A função de perda escolhida é o erro quadrático médio e o otimizador é 
# o Adam. O modelo é então treinado com os dados de entrada por 10 épocas. 
# A função retorna o modelo treinado e o objeto de normalização.

def create_train_model(data):
    # Normalizando os dados
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))
    data = data[:-1]

    # Redefinindo os dados para o formato adequado
    dataX = [data[n] for n in range(len(data))]
    dataX = np.reshape(dataX, (len(dataX), 1, 1))

    # Criando o modelo LSTM
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(dataX, data, epochs=100, batch_size=1, verbose=2)

    return model, scaler

# Função para detectar e substituir anomalias:
# Esta função detecta e substitui anomalias nos dados originais. Primeiro, a 
# série de dados é normalizada para estar entre 0 e 1. Em seguida, a média móvel
# e o desvio padrão móvel são calculados para janelas dos dados. Pontos que se 
# encontram além de um número `sigma` de desvios padrão da média são considera-
# dos anomalias. As anomalias são substituídas pelos valores previstos pelo mo-
# delo LSTM. A função retorna a série com anomalias substituídas e os índices 
# das anomalias.

def detect_replace_anomalies(model, series, window_size, sigma=1.0):
    # Normalizando a série
    scaler = MinMaxScaler()
    series = scaler.fit_transform(series.reshape(-1, 1))
    
    # Criando um data frame para a série
    series_df = pd.DataFrame(series)

    # Calculando a média móvel e o desvio padrão móvel
    rolling_mean = series_df.rolling(window=window_size).mean()
    rolling_std = series_df.rolling(window=window_size).std()

    # Identificando as anomalias
    anomalies = series_df[(series_df < (rolling_mean - sigma * rolling_std)) | (series_df > (rolling_mean + sigma * rolling_std))].index

    # Substituindo as anomalias pelos valores preditos pelo modelo
    for anom in anomalies:
        series[anom] = model.predict(series[anom].reshape(1, 1, 1))
        
    return scaler.inverse_transform(series), anomalies

# Função para plotar os dados originais e limpos:

def plot_data(orig_data, cleaned_data, anomalies):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(len(orig_data)), orig_data, color='blue', lw=2, label='Original Data')
    plt.plot(np.arange(len(cleaned_data)), cleaned_data, color='red', lw=2, label='Cleaned Data')
    plt.scatter(anomalies, orig_data[anomalies], color='green', s=100, edgecolor='black', label='Anomalies')
    plt.title('Original Data vs Cleaned Data', fontsize=20)
    plt.xlabel('Index', fontsize=16)
    plt.ylabel('Values', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.show()

# Carregando os dados
series = training_data['sum_quant_item'].values

# Criando e treinando o modelo
model, scaler = create_train_model(series)

# Detectando e substituindo as anomalias
cleaned_data, anomalies = detect_replace_anomalies(model, series, window_size=5, sigma=3)

# Plotando os dados originais e os dados limpos
plot_data(series, cleaned_data, anomalies)

# Salvando os dados limpos em um novo arquivo CSV
cleaned_data_df = pd.DataFrame(cleaned_data, columns=['sum_quant_item'])
cleaned_data_df.to_csv('cleaned_data.csv', index=False)
