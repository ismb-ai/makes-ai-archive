# Anomaly_Detection_Hybrid_ARIMA_CNN.py
#
#     O código em questão exemplifica uma metodologia inovadora para a detecção 
# e manipulação de anomalias em séries temporais combinando modelos ARIMA com 
# redes neurais convolucionais (CNN). Inicialmente, o código importa as biblio-
# tecas necessárias para a manipulação de dados, processamento estatístico, mo-
# delagem de aprendizado de máquina e visualização gráfica. A estacionariedade 
# da série é verificada através do teste Dickey-Fuller aumentado, sendo um pré-
# requisito para a aplicação eficaz do modelo ARIMA.
#     Após assegurar que a série é estacionária, o modelo ARIMA é ajustado auto-
# maticamente à série utilizando a função auto_arima, que seleciona os parâme-
# tros ótimos. Utilizando as previsões geradas pelo modelo ARIMA, calculam-se os 
# resíduos, que são então analisados em busca de desvios que caracterizam as 
# anomalias, através de um limiar estabelecido pelo escore Z. Normaliza-se a 
# série temporal para treinar a rede neural convolucional, permitindo que a rede 
# processe os dados de forma mais eficiente.
#     O modelo de CNN é composto por uma camada convolucional, uma camada de 
# achatamento e uma camada densa, e é compilado com um otimizador e uma função 
# de perda específicos para tarefas de regressão. Após o treinamento da CNN, 
# fazem-se previsões sobre os dados normalizados. Novamente, as anomalias são 
# detectadas comparando as previsões da CNN com os dados originais, levando em 
# conta um múltiplo do desvio padrão dos resíduos.
#     O programa realiza uma combinação das anomalias detectadas tanto pelo 
# modelo ARIMA quanto pela CNN, criando um conjunto unificado de pontos anômalos. 
# Subsequentemente, os dados são limpos pela remoção das referidas anomalias. 
# Finalmente, o código provê a visualização gráfica da série original e da série 
# já tratada, permitindo a comparação entre ambas e a avaliação dos resultados do 
# processo de limpeza.

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from pmdarima import auto_arima
from scipy.stats import zscore
import matplotlib.pyplot as plt

# test_stationarity(timeseries): 
# Verifica se a série temporal é estacionária utilizando o teste Dickey-Fuller 
# aumentado (`adfuller`). Retorna `True` se o p-valor for menor ou igual a 0.05, 
# indicando estacionariedade.

def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1] <= 0.05

# fit_ARIMA(series): 
# Ajusta um modelo ARIMA na série temporal usando a função `auto_arima` que 
# determina automaticamente a melhor combinação de parâmetros (p, d, q) para 
# o modelo ARIMA.

def fit_ARIMA(series):
    arima_model = auto_arima(series)
    return arima_model
    
# create_CNN(input_shape): 
# Cria um modelo de Rede Neural Convolucional Sequential que possui uma camada 
# convolucional (`Conv1D`), uma camada para "achatar" os dados (`Flatten`) e uma 
# camada densa (`Dense`) para a saída linear. O modelo é compilado com um 
# otimizador 'adam' e função de perda 'mean squared error' (mse).    

def create_CNN(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
    
# detect_anomalies(original_data, arima_preds, threshold): 
# Detecta anomalias subtraindo as previsões do ARIMA dos dados originais para 
# obter os resíduos. Aplica o escore Z (`zscore`) e considera como anomalias 
# aqueles pontos onde o valor absoluto do escore Z excede um limite (threshold).    

def detect_anomalies(original_data, arima_preds, threshold):
    residuals = original_data - arima_preds
    outliers = residuals[np.abs(zscore(residuals)) > threshold]
    return outliers
    
# plot_series(series, name):
# Plota a série temporal usando `matplotlib`, onde `series` é a série a ser 
# plotada e `name` é a legenda da série no gráfico.    

def plot_series(series, name):
    plt.figure(figsize=(10, 5))
    plt.plot(series, label=name)
    plt.legend(loc='best')
    plt.show()

# O restante do código executa o processo de detecção e tratamento de anomalias:
# - Carrega os dados da série temporal na variável `training_data`.
# - Verifica a estacionariedade dos dados. Se não estacionários, é emitido um aviso.
# - Ajusta um modelo ARIMA usando a função `fit_ARIMA` se a série for estacionária.
# - Gera previsões com o modelo ARIMA.
# - Detecta anomalias com base nos resíduos do modelo ARIMA usando a função `detect_anomalies`.
# - Normaliza a série temporal para entrada na CNN com o `MinMaxScaler`.
# - Redimensiona os dados normalizados para o formato de entrada aceito pela CNN.
# - Cria o modelo CNN com a função `create_CNN`.
# - Treina a CNN com os dados normalizados.
# - Gera previsões usando a CNN treinada.
# - Detecta anomalias baseadas nos resíduos da CNN.
# - Combina anomalias detectadas pelo ARIMA e pela CNN, removendo duplicatas.
# - Limpa os dados originais removendo as anomalias combinadas.
# - Plota a série original e a série limpa usando a função `plot_series`.

# load your dataset
training_data = data

# ensure your data is stationary
if not test_stationarity(training_data['sum_quant_item']):
    print('The series is not stationary. Please do differencing or box-cox transformation.')
else:
    # fit ARIMA on the series
    arima_model = fit_ARIMA(training_data['sum_quant_item'])

    # use ARIMA for predictions
    arima_preds = arima_model.predict(n_periods=len(training_data['sum_quant_item']))

    # Detect anomalies using ARIMA residuals
    anomalies = detect_anomalies(training_data['sum_quant_item'], arima_preds, threshold=3)

    # Normalize the series for CNN
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(np.array(training_data['sum_quant_item']).reshape(-1, 1))

    # reshape the data for CNN input
    series_scaled = series_scaled.reshape((series_scaled.shape[0], series_scaled.shape[1], 1))

    # create CNN model
    cnn_model = create_CNN(series_scaled.shape[1:])
    cnn_model.fit(series_scaled, series_scaled, epochs=10)
    cnn_preds = cnn_model.predict(series_scaled)

    # Detect anomalies using CNN
    residuals = series_scaled.flatten() - cnn_preds.flatten()
    outliers_cnn = training_data[residuals > 3 * np.std(residuals)]

    # Combine anomalies detected by ARIMA and CNN
    anomalies_combined = pd.concat([anomalies, outliers_cnn], axis=0).drop_duplicates()

    # Remove the anomalies from the original data
    cleaned_data = training_data.drop(anomalies_combined.index)

    # plot original and cleaned data
    plot_series(training_data['sum_quant_item'], 'Original data')
    plot_series(cleaned_data['sum_quant_item'], 'Cleaned data')
