# ARIMA.py
#
# O código apresenta uma implementação do modelo ARIMA para modelar uma 
# série temporal, onde as funções para executar o modelo, plotar o gráfico e 
# fazer previsões estão definidas. O modelo ARIMA é executado nos dados de 
# treinamento e depois usado para prever os próximos 50 passos da série temporal. 
# O resultado então é representado graficamente, mostrando tanto os dados 
# originais como o modelo ajustado e as previsões para futuros passos.

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# A função run_arima recebe como entrada os dados da série temporal e a ordem do
# modelo ARIMA. Nela, o modelo ARIMA é instanciado com os dados e ajustado. 
# A função retorna o modelo ajustado.

def run_arima(data, order=(1, 1, 1)):
    model = ARIMA(data['sum_quant_item'].values, order=order)
    model_fit = model.fit()
    return model_fit

# plot_arima é a função que plota os dados originais e o modelo ajustado da ARIMA.
# Ela recebe como entrada os dados e o modelo ajustado, plota os valores originais 
# em azul e os valores ajustados em vermelho. 

def plot_arima(data, model_fit):
    plt.plot(data['sum_quant_item'].values, 'b', label='Original')
    plt.plot(model_fit.predict(), 'r', label='Fitted')
    plt.title('Original data and Fitted ARIMA model')
    plt.legend()
    plt.show()

# A função predict_arima é utilizada para fazer previsões usando o modelo ARIMA 
# ajustado. Ela recebe como entrada o modelo ajustado e o número de passos que você 
# deseja prever no futuro. A função retorna as previsões.
    

def predict_arima(model_fit, steps):
    forecast = model_fit.forecast(steps)
    return forecast

# Aqui chamamos a função run_arima com os dados de treino, ajustamos o modelo ARIMA
# e armazenamos em model_fit.
model_fit = run_arima(training_data)

# Em seguida, usamos a função plot_arima para traçar o gráfico da série temporal
# e do modelo ajustado.
plot_arima(training_data, model_fit)

# Finalmente, preveremos os próximos 50 passos com a função predict_arima
# e imprimir os resultados.forecast = predict_arima(model_fit, 50)
print('Forecast: ', forecast)
