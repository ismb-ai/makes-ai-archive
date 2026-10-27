# Anomaly_Detection_Exponential_Smoothing_CNN.py
#
# O programa a seguir consiste em uma implementação de aprendizado profundo que
# combina a capacidade da suavização exponencial em analisar as oscilações tem-
# porais com uma rede de CNN para modelar padrões locais, propondo um tratamento 
# eficaz das anomalias na série temporal e mantendo os dados consistentes para 
# análises futuras.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential

# A função `exponential_smoothing` aplica suavização exponencial ao conjunto de 
# dados. Esta técnica pondera as observações da série temporal, dando mais 
# importância aos pontos mais recentes. O resultado é uma série suavizada, que 
# idealmente apresenta menos ruído e destaca a tendência dos dados. O parâmetro 
# `alpha` controla o grau de suavização.

def exponential_smoothing(data, alpha):
    result = [data[0]] 
    for n in range(1, len(data)):
        result.append(alpha * data[n] + (1 - alpha) * result[n-1])
    return result

# A função `cnn_model` constrói o modelo de CNN, definindo a arquitetura sequen-
# cial. A rede consiste de uma camada convolucional `Conv1D`, que extrai recursos 
# locais da sequência de entrada, seguida por uma camada de `MaxPooling1D`, que 
# reduz a dimensão espacial dos recursos. Após o achatamento dos dados por uma 
# camada `Flatten`, segue-se uma `Dense` com 50 unidades e ativação ReLU para 
# maior processamento, culminando em uma camada de saída `Dense` com uma unidade 
# para previsão do próximo valor na série temporal.

def cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# A função `remove_anomalies` executa o processo de detecção e remoção de anoma-
# lias. Primeiro, aplica a suavização exponencial aos dados e, em seguida, norma-
# liza a série suavizada com o `MinMaxScaler`. Os dados são então formatados pa-
# ra se adequarem à entrada da CNN, representando os últimos 10 pontos preceden-
# tes para prever o próximo ponto. O modelo CNN é treinado com esses dados, e 
# após realizar previsões, identifica-se e remove-se os pontos residuais que ex-
#cedam o limiar definido de 2.5 vezes o desvio padrão dos resíduos.

def remove_anomalies(training_data):
    smoothed = exponential_smoothing(training_data['sum_quant_item'], 0.9)
    training_data['smoothed'] = smoothed

    # apply MinMaxScaler
    scaler = MinMaxScaler()
    training_data['scaled_smoothed'] = scaler.fit_transform(training_data['smoothed'].values.reshape(-1, 1))

    # preparing data for CNN
    X = np.array([training_data['scaled_smoothed'].values[i-10:i] for i in range(10,len(training_data))])
    y = training_data['scaled_smoothed'].values[10:]

    # training the CNN model
    model = cnn_model((10, 1))
    model.fit(X, y, epochs=20, verbose=0)

    # prediction
    pred = model.predict(X)
    pred_inverse_scaled = scaler.inverse_transform(pred)

    # remove anomalies
    residuals = y - pred_inverse_scaled.reshape(-1)
    anomalies_index = np.where(np.abs(residuals) > 2.5*np.std(residuals))[0] + 10
    training_data.drop(anomalies_index, inplace=True)

    # return cleaned data
    cleaned_data = training_data.drop(['smoothed', 'scaled_smoothed'], axis=1)
    return cleaned_data

# As últimas linhas do programa carregam os dados de treinamento e aplicam a função `remove_anomalies` para limpar a série temporal. Finalmente, um gráfico é gerado para visualizar a série temporal original após a limpeza das anomalias.

# load the data
training_data = data
cleaned_data = remove_anomalies(training_data)

# plot original vs cleaned data
plt.figure(figsize=(10,6))
plt.plot(cleaned_data['time_scale'], cleaned_data['sum_quant_item'], label='Cleaned data')
plt.legend()
plt.show()
