# Exponential_Smoothing_RNN_v2.py
#
# Este código implementa um sistema híbrido de aprendizado profundo consistindo
# em uma técnica de suavização exponencial combinada com uma rede neural 
# recorrente (RNN) para modelar uma série temporal chamada "training_data".

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Suavização Exponencial

# A função "aplicar_suavizacao_exponencial" recebe como parâmetro um conjunto de 
# dados ('data') e aplica a técnica de suavização exponencial, que modela a
# tendência e sazonalidade dos dados. O modelo ajustado é então utilizado para
# calcular os valores suavizados, retornados pela função.

def aplicar_suavizacao_exponencial(data):
    modelo_suavizacao = ExponentialSmoothing(data)
    modelo_ajustado = modelo_suavizacao.fit()
    suavizado = modelo_ajustado.fittedvalues
    return suavizado

# Preparar dados para a RNN

# A função "preparar_dados_rnn" recebe um conjunto de dados e um valor 
# 'look_back', que representa quantos passos no tempo serão considerados pela 
# RNN. Ela gera duas listas, onde 'dataX' contém sequências de 'look_back' 
# pontos de dados e 'dataY' contém o ponto subsequentes a cada sequência. Ambas 
# as listas são convertidas em arrays numpy antes de serem retornadas.

def preparar_dados_rnn(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return np.array(dataX), np.array(dataY)

# Criação da RNN

# A função "criar_rnn_model" cria e configura um modelo de RNN. Adiciona-se uma 
# camada LSTM com 4 unidades e então uma camada densa. O modelo é então 
# compilado com a função de perda mean_squared_error e o otimizador Adam.

def criar_rnn_model(look_back):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(4, input_shape=(1, look_back)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Lendo os dados
# Para fim de exemplo, vou supor que foi um DataFrame do pandas
# training_data = pd.read_csv('seu_arquivo.csv')

# Aplicando suavização exponencial:
# A coluna 'sum_quant_item' do dataframe 'training_data' é suavizada e os 
# valores suavizados são armazenados na coluna 'suavizado'.

training_data['suavizado'] = aplicar_suavizacao_exponencial(training_data['sum_quant_item'])

# Ajustando a escala dos dados:
# Os valores são então normalizados para o intervalo [0,1] utilizando o 
# MinMaxScaler, e são preparados para a RNN, com 'look_back' igual a 1.

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(training_data['suavizado'].values.reshape(-1,1))

# Preparando os dados para a RNN
look_back = 1
trainX, trainY = preparar_dados_rnn(dataset, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# Criação e ajuste da RNN
# Em seguida, a RNN é criada e ajustada com os dados preparados, ao longo de 50 
# épocas, com tamanho de lote igual a 1 e mensagens de progresso ativadas.

model = criar_rnn_model(look_back)
model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)

# Gerando predições
# As predições do modelo são então calculadas para os dados de treino e 
# transformadas de volta para a escala original usando o inverso da 
# transformação MinMaxScaler.
trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)

# Cálculo do erro do modelo
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))

# Definindo o estilo dos gráficos
sns.set_style("whitegrid")

# Criação da figura e dos eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Plotando os dados originais
ax.plot(training_data['sum_quant_item'].values, color='blue', label='Dados originais')

# Plotando as previsões do modelo
ax.plot(trainPredict, color='red', label='Predições do modelo')

# Adicionando a legenda
ax.legend(loc='upper left')

# Adicionando um título ao gráfico
ax.set_title('Comparação dos dados originais e as predições do modelo')

# Adicionando rótulos aos eixos
ax.set_xlabel('Índice de tempo')
ax.set_ylabel('Valor')

# Mostrando o gráfico
plt.show()
