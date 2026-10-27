# ARIMA_RNN.py
#
# O código-fonte a seguir realiza a modelagem de uma série temporal, representa-
# da pelo conjunto de dados "training_data", empregando uma abordagem híbrida de
# aprendizado profundo utilizando ARIMA (Médias Móveis Integradas Auto-
# Regressivas) em conjunto com Redes Neurais Recorrentes (RNN).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

# Preparação de Dados:
# Divide os dados em conjuntos de treino e teste, escalona os dados usando 
# Min-Max scaling, e retorna os dados de treino, teste e o objeto scaler.

def prepare_data(df, test_size):
    train, test = train_test_split(df, test_size=test_size, shuffle=False)
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train, test, scaler
    
# Execução do Modelo ARIMA:
# Ajusta e treina um modelo ARIMA utilizando a biblioteca "pmdarima" e retorna o 
# modelo ARIMA e um objeto scaler para a série temporal.

def run_arima(train, scaler):
    scaler_arima = MinMaxScaler()
    scaler_arima.fit(train[:, 1].reshape(-1,1))
    
    arima_model = auto_arima(train[:, 1], start_p=1, start_q=1, max_p=3, max_q=3, m=12, d=1, D=1, 
                             trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    arima_model.fit(train[:, 1])
    return arima_model, scaler_arima
    
# Escalonamento das Previsões do ARIMA:
# Realiza previsões utilizando o modelo ARIMA ajustado e escala as previsões 
# usando o scaler associado ao ARIMA.

def scale_predictions(arima_model, test, scaler_arima):
    arima_predictions = arima_model.predict(n_periods=len(test))
    arima_predictions = scaler_arima.transform(arima_predictions.reshape(-1,1))
    return arima_predictions
    
# Construção do Modelo LSTM:
#     Define e compila um modelo LSTM com uma camada LSTM de 100 unidades, 
# ativação ReLU, e uma camada densa de saída.

def build_lstm_model(n_input, n_features):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Previsões do LSTM e Visualização:
# Inverte a escala das previsões ARIMA e LSTM para a escala original, e plota os
# valores originais versus os valores previstos pelo ARIMA e LSTM.

def make_predictions_and_plot(test, arima_predictions, lstm_predictions, scaler):
    arima_predictions = scaler.inverse_transform(arima_predictions)
    scaler_lstm = MinMaxScaler()
    scaler_lstm.min_, scaler_lstm.scale_ = scaler.min_[1], scaler.scale_[1]
    lstm_predictions = scaler_lstm.inverse_transform(lstm_predictions)

    plt.figure(figsize=(10,6))
    plt.plot(test[:, 1], label='Original')
    plt.plot(arima_predictions, label='ARIMA Predictions')
    plt.plot(lstm_predictions, label='LSTM Predictions')
    plt.title('Original vs Predicted Values')
    plt.legend()
    plt.show()

# Carrega ou gera o conjunto de dados, realiza a preparação dos dados, executa o
# modelo ARIMA, escala as previsões ARIMA, constrói e treina o modelo LSTM, gera
# previsões LSTM, e finalmente, visualiza as previsões comparadas com os valores
# reais.

if __name__ == "__main__":
    training_data = pd.DataFrame()  # Load or generate your dataframe here
    train, test, scaler = prepare_data(training_data, 0.2)
    arima_model, scaler_arima = run_arima(train, scaler)
    arima_predictions = scale_predictions(arima_model, test, scaler_arima)
    
    n_input = 12 
    n_features = 1 
    generator = TimeseriesGenerator(data=arima_predictions, targets=arima_predictions, length=n_input, batch_size=1)
    
    model = build_lstm_model(n_input, n_features)
    model.fit(generator, epochs=50) 

    lstm_predictions = []
    batch = arima_predictions[-n_input:].reshape((1, n_input, n_features))
    for i in range(n_input): 
        lstm_predictions.append(model.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[lstm_predictions[i]]],axis=1)
    
    make_predictions_and_plot(test, arima_predictions, lstm_predictions, scaler)
