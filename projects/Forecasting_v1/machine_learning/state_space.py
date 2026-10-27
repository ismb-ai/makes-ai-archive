# state_space.py
#
# Este programa modela uma série-temporal usando um método de espaço de fase,
# dividindo os dados em conjuntos de treinamento e teste, ajustando o modelo 
# nos dados de treino e prevendo valores para os dados de teste. O erro 
# quadrático médio entre as previsões e os dados de teste é calculado e 
# plotado para cada tamanho de conjunto de teste diferente. Os dados 
# originais, as previsões e o erro quadrático médio são visualizados em 
# gráficos.

# Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from pmdarima.model_selection import train_test_split

# Função para ajustar o modelo de espaço de estados:
# A função fit_and_forecast_state_space_model é usada para ajustar o modelo 
# de espaço de estados aos dados e fazer previsões.

def fit_and_forecast_state_space_model(training_data, test_size):
	
    # Separação de Dados:
    # Os dados são separados em treino e teste usando a função train_test_split.
    
    train, test = train_test_split(training_data['sum_quant_item'], test_size=test_size)
    
    # Ajuste do Modelo ARIMA:
    # O modelo ARIMA é ajustado aos dados usando a função auto_arima. Esta função 
    # escolhe os melhores valores para os parâmetros do modelo ARIMA.
    
    model = pm.auto_arima(train, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=1, d=None, seasonal=False,
                          start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    print(model.summary())
    
    # Previsão:
    # As previsões são feitas usando a função predict do modelo ajustado.
    
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
    
    # Plotando os Dados e o Modelo:
    # Os dados de treino, teste e as previsões são plotados em um gráfico.
    
    plt.figure(figsize=(10,4))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='Prediction')
    plt.title('Original data with model')
    plt.legend()
    plt.show()
    
    # Cálculo do Erro Quadrático Médio:
    # O erro quadrático médio é calculado usando a função mean_squared_error.
    
    mse = mean_squared_error(test, forecast)
    
    return mse

# Teste de tamanhos e Resultados MSE
# São definidos tamanhos de teste e um loop é executado para cada tamanho, 
# ajustando o modelo, fazendo previsões e calculando o MSE.

test_sizes = np.arange(0.1, 1.0, 0.1)
mse_results = []
for i in test_sizes:
    mse = fit_and_forecast_state_space_model(training_data, i)
    mse_results.append(mse)
    
# Plotando o MSE Geral
# O erro quadrático médio para cada tamanho de conjunto de teste é plotado.

plt.figure(figsize=(10,4))
plt.plot(test_sizes, mse_results, marker='o')
plt.title('Overall mean square error')
plt.xlabel('Test set size')
plt.ylabel('MSE')
plt.grid(True)
plt.show()
