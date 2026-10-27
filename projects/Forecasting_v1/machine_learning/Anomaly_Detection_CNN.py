# Anomaly_Detection_CNN.py
#
# O código em questão implementa uma abordagem para detecção e correção de 
# anomalias em séries temporais, utilizando uma Rede Neural Convolucional (CNN).
# As principais etapas do código são as seguintes:
#
# - Normalização de Dados: Os dados da série temporal, representados pela 
# variável sum_quant_item, são normalizados para possuírem média nula e desvio 
# padrão unitário.
# - Criação e Treinamento do Modelo: A função create_model cria e treina um 
# modelo de rede neural. Os dados normalizados são organizados em formato bidi-
# mensional, adequado para modelagem convolucional. O modelo inclui uma camada 
# convolucional, seguida por uma camada Flatten e uma camada Dense para a saída. 
# O treinamento ocorre ao longo de 10 épocas.
# - A função detect_anomalies utiliza o modelo treinado para detectar anomalias 
# comparando resíduos (diferença entre o valor real e o previsto) com um limiar. 
# Anomalias são identificadas e substituídas pela mediana dos dados normalizados, 
# sendo posteriormente retransformadas para a escala original.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import TensorBoard
import datetime

# create_model(training_data):
# A função cria e treina o modelo de rede neural. Dá-se início pelo procedimento
# de normalização dos dados, no qual os valores da variável `sum_quant_item`
# são transformados a fim de possuírem média nula e desvio padrão unitário.
# Sequencialmente, os dados são redistribuídos em formato bidimensional,
# adequando-se ao paradigma da modelagem convolucional. O modelo é então montado
# com a utilização da biblioteca Keras, englobando uma camada convolucional que
# recebe os dados já reformulados, uma camada Flatten encarregada pela 
# linearização dos dados para entrada na camada Dense, a qual realiza a saída
# final da rede. A finalização do modelo faz-se por meio da sua compilação 
# com o otimizador `adam` e com a função de perda do erro quadrático médio (`mse`).
# O treinamento ocorre ao longo de 10 épocas, com os logs do treinamento sendo
# armazenados para inspeções futuras. A função retorna o modelo já treinado,
# o Scaler usado na normalização e os históricos de treinamento e perda.

def create_model(training_data):
    scaler = StandardScaler()
    training_data['scaled_sum_quant_item'] = scaler.fit_transform(training_data[['sum_quant_item']])
    X = []
    Y = []
    
    for i in range(1, len(training_data)):
        X.append(training_data.loc[i-1:i, 'scaled_sum_quant_item'].values)
        Y.append(training_data.loc[i, 'scaled_sum_quant_item'])

    X = np.array(X).reshape((-1, 2, 1))
    Y = np.array(Y)
    
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape=X[0].shape))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    history = model.fit(X, Y, batch_size=1, epochs=10, callbacks=[tensorboard_callback])

    mse_history = history.history['loss']
    
    return model, scaler, history, mse_history
    
# detect_anomalies(model, training_data, scaler): A função é responsável pela
# detecção e correção de anomalias em `training_data` com base no modelo
# prévio. Tal detecção ocorre por meio da comparação dos resíduos (i.e., a 
# diferença entre o valor efetivo e o previsto) com um limiar, estabelecido como
# sendo três vezes o desvio padrão desses resíduos. Pontos cujos resíduos 
# absolutos ultrapassam tal limiar são identificados como anomalias. Tais 
# anomalias são então substituídas pela mediana dos dados normalizados e
# retransformadas ao valor original através do método `inverse_transform` do
# Scaler. A função finalmente retorna os dados com as anomalias tratadas.

def detect_anomalies(model, training_data, scaler):
    X_test = []
    Y_test = training_data.loc[1:, 'scaled_sum_quant_item'].values
    
    for i in range(1, len(training_data)):
        X_test.append(training_data.loc[i-1:i, 'scaled_sum_quant_item'].values)
        
    X_test = np.array(X_test).reshape((-1, 2, 1))

    predictions = model.predict(X_test)
    residuals = Y_test - predictions.reshape(-1)

    anomalies = np.where(np.abs(residuals) > 3*np.std(residuals))[0]

    training_data.loc[training_data.index[1 + anomalies], 'scaled_sum_quant_item'] = training_data['scaled_sum_quant_item'].median()
    training_data["clean_sum_quant_item"] = scaler.inverse_transform(training_data[['scaled_sum_quant_item']])

    return training_data
    
# A função `plot_data(training_data, cleaned_data)` destina-se à visualização do
# `training_data` original e dos dados pós-tratamento das anomalias. Esta gera 
# um gráfico contendo duas linhas, uma para cada conjunto de dados.

def plot_data(training_data, cleaned_data):
    plt.figure(figsize=(12,6))
    plt.plot(training_data['time_scale'], training_data['sum_quant_item'], 'b', label = 'Original data')
    plt.plot(cleaned_data['time_scale'], cleaned_data['clean_sum_quant_item'], 'r', label = 'Cleaned data')
    plt.legend(loc='upper left')
    plt.xlabel('Time scale')
    plt.ylabel('Sum quantity item')
    plt.title('Original vs Cleaned data')
    plt.grid(True)
    plt.show()

# A função `plot_loss(history)` objetiva a apresentação gráfica do histórico 
# de perda decorrente do treinamento do modelo, mostrando a variação da perda a
# cada época de treino.

def plot_loss(history):
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()

# Por último, a função `plot_mse_trend(mse_history)` proporciona a visualização
# da tendência global do erro quadrático médio (MSE) a cada época do treinamento.

def plot_mse_trend(mse_history):
    plt.figure(figsize=(12,6))
    plt.plot(mse_history)
    plt.title('Overall Trend of MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()

# Ao final do código, as funções `create_model`, `detect_anomalies`, `plot_data`,
# `plot_loss` e `plot_mse_trend` são invocadas a fim de demonstrar os resultados
# gerados pelo modelo.

modelo, escalador, historico, mse_historico = create_model(training_data)
cleaned_data = detect_anomalies(modelo, training_data, escalador)
plot_data(training_data, cleaned_data)
plot_loss(historico)
plot_mse_trend(mse_historico)
