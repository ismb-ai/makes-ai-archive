# recurrent_neural_networks.py
#
# Este código implementa uma abordagem de embedding para modelar a série 
# temporal "training_data" utilizando um modelo RNN. Primeiro, ele normaliza 
# os dados e os prepara para treinamento, depois cria e treina o modelo RNN,
# faz previsões e calcula o erro quadrático médio. Por fim, ele exibe os 
# resultados, comparando as previsões do modelo com os dados originais e 
# mostrando a tendência do erro ao longo do tempo.

# Libraries

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Definindo a função para compor o modelo:
# A função create_model define a arquitetura da rede neural. Esta é uma rede 
# Recorrente Simples com 4 neurônios na camada oculta e uma densa com 1 
# neurônio de saída.

def create_model(window_size):
    model = Sequential()
    model.add(SimpleRNN(4, input_shape=(1, window_size)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Definindo a função para criar a "embedded series".
# A função create_embedded_series é responsável por reformatar a série 
# temporal de entrada em uma matriz que possa ser usada para treinar a RNN.

def create_embedded_series(data, window_size):
    data_s = np.array(data)
    x = []
    y = []
    for i in range(len(data_s)-window_size):
        x.append(data_s[i:i+window_size])
        y.append(data_s[i+window_size])
    x = np.reshape(np.array(x), (len(x), 1, window_size))
    y = np.reshape(np.array(y), (len(y), 1))
    return x, y

# Definindo a função principal:
# A função apply_model define os passos necessários para treinar e testar 
# o modelo RNN. Isso inclui a normalização dos dados, a preparação dos dados,
# a construção do modelo, o treinamento do modelo, a execução de previsões, 
# o cálculo do MSE e a plotagem dos resultados.

def apply_model(training_data, window_size, horizon):
    
    # Normalizando os dados
    # Os dados são normalizados para o intervalo [0,1] utilizando MinMaxScaler.

    scaler = MinMaxScaler(feature_range=(0, 1))
    sum_quant_item_scaled = scaler.fit_transform(training_data['sum_quant_item'].values.reshape(-1,1))
    
    # Preparando os dados
    # Os dados preparados são criados chamando a função create_embedded_series.

    x, y = create_embedded_series(sum_quant_item_scaled, window_size)

    # Construindo o modelo
    # O modelo é construído chamando a função create_model.

    model = create_model(window_size)

    # Treinamento do modelo e suas respectivas previsões:
    # Primeiro, o modelo é treinado usando os dados preparados.
    # Subsequentemente, as previsões são feitas usando o modelo treinado. 
    # As previsões são então revertidas para a escala original chamando 
    # a função inverse_transform.

    model.fit(x, y, epochs=200, batch_size=32)

    predictions = model.predict(x)
    predictions = scaler.inverse_transform(predictions)
    y = scaler.inverse_transform(y)

    # Calculando o Erro Quadrado Médio
    # O MSE é calculado entre os valores reais e previstos.

    mse = mean_squared_error(y, predictions)

    # Plotando dados originais e as previsões do modelo
    # Os dados originais e as previsões do modelo são plotados em um gráfico.

    plt.figure(figsize=(20,10))
    plt.plot(y, label='Original')
    plt.plot(predictions, label='Model')
    plt.title('Comparison of Model with Original Data')
    plt.xlabel('Time Scale')
    plt.ylabel('Sum Quant Item')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotando tendência MSE
    # A tendência do MSE ao longo do tempo é plotada em um gráfico.

    plt.figure(figsize=(20,10))
    plt.plot(range(len(y)), [mse]*len(y), label='MSE')
    plt.title('MSE Trend')
    plt.xlabel('Time Scale')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

# Por fim, a função apply_model é chamada para executar o processo de modelagem.

apply_model(training_data, window_size=10, horizon=1)
