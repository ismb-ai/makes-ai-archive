# Anomaly_Detection_Moving_Average_RNN.py
#
#     O código aqui apresentado propõe um mecanismo de detecção e tratamento de 
# discrepâncias em dados sequenciais temporais, aplicando o conceito de médias 
# móveis em conjunto com uma rede neural recorrente, especificamente a Long 
# Short-Term Memory (LSTM). Inicialmente, as dependências e bibliotecas necessá-
# rias são importadas, com destaque para as destinadas ao processamento de dados 
# e construção de redes neurais. O tratamento dos dados inicia-se pela sua nor-
# malização, de maneira a adequar a amplitude das variáveis de entrada para um 
# intervalo padrão que otimiza a performance do modelo de aprendizagem profunda.
#     Posteriormente, a função `prepare_data` é encarregada de organizar a série 
# temporal em porções sequenciais que servirão de entrada para o treinamento da 
# LSTM, cada segmento é acompanhado pelo valor subsequente da série que servirá 
# como alvo a ser predito pelo modelo. A LSTM, uma estrutura projetada para cap-
# tar dependências de longo prazo em dados sequenciais, é treinada com as se-
# quências modeladas, induzindo-a a reconhecer padrões subjacentes na série 
# temporal. O treino é realizado através da função `train_lstm`, que configura 
# e executa o processo de otimização dos parâmetros internos da rede.
#     Com a LSTM devidamente ajustada, avança-se para a etapa de detecção de 
# anomalias com a função `detect_anomalies`, que, por meio do cálculo de erros 
# entre previsões e valores reais, identifica pontos discrepantes com base em um 
# limiar estatístico. Esses pontos são classificados como anômalos caso seu erro 
# supere a média acrescida de um múltiplo do desvio padrão. Dessa forma, é 
# possível isolar as observações que desviam do comportamento padrão aprendido 
# pela LSTM.
#     Por fim, exclui-se do conjunto original os dados identificados como 
# anormais, resultando em uma série temporal limpa, desprovida de flutuações 
# atípicas que poderiam mascarar ou distorcer análises futuras. As séries, 
# original e depurada, são então representadas graficamente através da função 
# `plot_data`, que ilustra ambas numa mesma figura para fácil comparação visual. 
# Esse processo de limpeza de dados é fundamental para garantir a integridade e 
# confiabilidade de modelos preditivos subsequentes ou para a correta interpre-
# tação da série temporal em análises exploratórias ou inferenciais.

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# prepare_data(training_data, timesteps=1): 
# Esta função é responsável por preparar os dados para o treinamento do modelo 
# LSTM. Primeiro, escala os dados para o intervalo [0, 1] usando `MinMaxScaler`. 
# Em seguida, estrutura os dados de entrada `X` e saída `Y` para que o modelo 
# LSTM possa processá-los. Aqui, `X` conterá sequências das observações da série 
# temporal e `Y` conterá a observação subsequente para cada sequência de 
# entrada.

def prepare_data(training_data, timesteps=1):
    # Convert dataframe to numpy array and scale values to range [0,1]
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(training_data)
    
    # Prepare inputs for LSTM
    X, Y = [], []
    for i in range(len(dataset)-timesteps-1):
        a = dataset[i:(i+timesteps), 0]
        X.append(a)
        Y.append(dataset[i + timesteps, 0])
    return np.array(X), np.array(Y), scaler

# train_lstm(X, Y, timesteps=1): 
# Nesta função, os dados `X` são redimensionados para o formato requerido pelo 
# LSTM ([amostras, passos de tempo, características]). Um modelo sequencial é 
# construído com uma camada LSTM seguida por uma camada densa. O modelo é compi-
# lado com a função de perda 'mean_squared_error' e otimizado pelo algoritmo 
# 'adam'. O modelo é então treinado com os dados `X` e `Y`.

def train_lstm(X, Y, timesteps=1):
    # Reshape X to [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], timesteps, X.shape[1]))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(4, input_shape=(timesteps, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=50, batch_size=1, verbose=2)
    
    return model

# detect_anomalies(model, X, Y, original_df, timesteps=1): 
# Com o modelo treinado, essa função realiza previsões nos dados `X` e compara 
# as previsões com os dados reais `Y` para calcular o erro que é usado para 
# detectar anomalias. Anomalias são identificadas quando o erro de uma determi-
# nada observação excede um limiar definido pela média dos erros mais duas vezes 
# o desvio padrão. As anomalias detectadas são retornadas como `anomalous_data`, 
# que é um subset dos dados originais.

def detect_anomalies(model, X, Y, original_df, timesteps=1):
    # Make predictions
    X = np.reshape(X, (X.shape[0], timesteps, X.shape[1]))
    pred = model.predict(X)
    
    # Calculate errors
    error = np.square(np.subtract(Y, pred)).mean()
    std_dev = np.std(np.subtract(Y, pred))
    
    # Identify anomalies
    anomalies = np.where(np.abs(np.subtract(Y, pred)) > error+2*std_dev)[0]
    anomalous_data = original_df.iloc[anomalies, :]
    
    return anomalous_data

# plot_data(original_data, cleaned_data): 
# Esta função gera uma visualização onde plota os dados originais 
# (`original_data`) em azul e os dados após a remoção de anomalias 
# (`cleaned_data`) em vermelho, permitindo uma comparação visual entre eles.

def plot_data(original_data, cleaned_data):
    plt.figure(figsize=(14,7))
    plt.plot(original_data, color='blue', label='Original data')
    plt.plot(cleaned_data, color='red', label='Cleaned data')
    plt.legend()
    plt.show()
    
# No resto do código:
# - É carregado o conjunto de dados `training_data`.
# - As funções são chamadas na sequência, começando pelo preparo dos dados, 
# seguido pelo treinamento do LSTM, e, depois, pela detecção de anomalias.
# - Os dados anomalous detectados são removidos do conjunto de dados original, 
# resultando em `cleaned_data`.
# - Por fim, os resultados são plotados, mostrando os dados originais e os dados limpos.    

# Load your training_data
training_data = data

# Call functions
X, Y, scaler = prepare_data(training_data[['sum_quant_item']])
model = train_lstm(X, Y)
anomalous_data = detect_anomalies(model, X, Y, training_data)
cleaned_data = training_data.drop(anomalous_data.index)

# Plot results
plot_data(training_data, cleaned_data)
