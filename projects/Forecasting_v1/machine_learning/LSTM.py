# LSTM.py
#
# O código implementa um modelo LSTM para prever séries temporais a partir dos dados
# de treinamento. Primeiro, é realizado o pré-processamento dos dados, incluindo a
# normalização e a criação de conjuntos de treinamento e teste. Em seguida, o modelo
# é treinado e as previsões são feitas, que são então plotadas para comparação com os 
# dados originais.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.metrics import mean_squared_error

# A função `create_window(dataframe, window_size, horizon)` é utilizada 
# para converter a série temporal em múltiplos pequenos blocos, ou 
# janelas, de dados. Cada janela contém uma quantidade definida de 
# observações consecutivas (o "window_size") e é empurrada para a frente
# ao longo da série temporal a cada iteração do laços, sempre avançando
# n-passos na série, onde n:="horizons".

def create_window(dataframe, window_size, horizon):
    dataX, dataY = [], []
    for i in range(len(dataframe) - window_size-horizon):
        a = dataframe.iloc[i:(i+window_size)].copy()
        dataX.append(a)
        dataY.append(dataframe.iloc[i + window_size + horizon-1])
    return np.array(dataX), np.array(dataY)


# A função `plot_results(original_data, predicted_data, window_size)` é
# usada para visualizar os resultados das previsões do modelo em relação 
# aos dados originais.

def plot_results(original_data, predicted_data, window_size):
    plt.figure(figsize=(15,10))
    plt.plot(original_data[window_size:],color='blue',label='Dados Originais')
    plt.plot(predicted_data,color='red',label='Previsoes')
    plt.legend(loc='best')
    plt.title('Dados Originais vs Previsoes')
    plt.show()

# A função `plot_mse(history)` é utilizada para visualizar a evolução do 
# erro quadrático médio (MSE) ao longo dos "epochs" de treinamento. É um 
# indicativo da performance e convergência do modelo ao longo do tempo.

def plot_mse(history):
    plt.figure(figsize=(15,10))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Tendência do MSE')
    plt.legend()
    plt.show()

# A função `main(training_data, window_size=5, horizon=1)` é a função 
# principal do programa e onde os dados são de fato processados. Sua
# execução se dá mediante os seguintes passos:
#
# 1. Extrai a série temporal do objeto "training_data" em um DataFrame
# Pandas.
# 2. Usa a classe MinMaxScaler para normalizar a série temporal para ter um
# intervalo entre 0 e 1.
# 3. Converte a série temporal em janelas de dados usando a função 
# `create_window`.
# 4. Divide os dados em conjuntos de treino e de teste.
# 5. Constrói um modelo sequencial LSTM com uma única camada oculta de 50
# neurônios.
# 6. Compila e treina o modelo usando o algoritmo de otimização Adam e o erro
# quadrático médio como função de perda.
# 7. Usa o modelo treinado para fazer previsões nos conjuntos de treino e de
# teste.
# 8. Reverte a normalização aplicada aos dados para trazer as previsões de 
# volta à escala original.
# 9. Plota os resultados das previsões vs os dados originais, e a evolução
# do MSE.
    
def main(training_data, window_size=5, horizon=1):

    sum_quant_item = pd.DataFrame(training_data['sum_quant_item'])

    # Normalização dos dados
    scaler = MinMaxScaler(feature_range=(0,1))
    sum_quant_item = scaler.fit_transform(sum_quant_item)

    # Criação das janelas
    X, y = create_window(pd.DataFrame(sum_quant_item), window_size, horizon)

    # Divisão entre dados de treino e de teste
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Criação da arquitetura LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[MeanSquaredError()])
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, y_test))

    # Previsão dos dados
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Volta dos dados para a escala original
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)
    original_data = scaler.inverse_transform(sum_quant_item)

    # Plot do resultado
    plot_results(original_data, np.append(y_train_pred, y_test_pred), window_size)

    # Plot da tendência do MSE
    plot_mse(history)


if __name__ == '__main__':
    #training_data = pd.read_csv('yourfilepath.csv')    # Substitua com o CSV a ser usado!
    main(training_data)
