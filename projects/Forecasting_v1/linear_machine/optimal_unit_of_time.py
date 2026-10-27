# optimal_unit_of_time.py
#
# Este código realiza uma análise de série temporal visando encontrar o 
# intervalo de tempo ótimo para agrupar os dados e treinar um modelo de 
# regressão linear. Inicialmente, os dados de treinamento são ordenados
# em ordem cronológica. Posteriormente, é definida uma função que agrupa 
# os dados em unidades de tempo especificadas, treina um modelo de 
# regressão linear, faz previsões e calcula o erro quadrático médio(MSE).
# Um intervalo de unidades de tempo é definido para testar o desempenho
# do modelo e os resultados são plotados em um gráfico. A função retorna 
# a unidade de tempo que resulta no menor MSE.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Ordenação da série temporal. É necessário garantir que os dados estejam em
# ordem cronológica:

training_data = training_data.sort_values('time_scale')

# evaluate_model_for_unit_time(unit_time, training_data)
# Esta função realiza os seguintes processos:
# - Agrega 'sum_quant_item' em intervalos de 'unit_time' usando groupby e
# reset_index() para redefinir os índices.
# - Remodela os dados agrupados para X (features) e y (target).
# - Treina um modelo de regressão linear usando X e y.
# - Realiza previsões usando o modelo treinado e calcula o erro quadrático médio
# (MSE) entre as previsões e os valores verdadeiros

def evaluate_model_for_unit_time(unit_time, training_data):
    """
    Avalia o MSE de um modelo de regressão linear em uma determinada unidade de tempo
    """
    # Crie uma nova estrutura de dados agrupando 'sum_quant_item' em 'unit_time' intervalos de tempo
    agg_data = training_data.groupby(training_data['time_scale'] // unit_time)['sum_quant_item'].sum().reset_index()

    # Reshape data
    X = agg_data['time_scale'].values.reshape(-1, 1)
    y = agg_data['sum_quant_item'].values

    # Treine o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Preveja e calcule o erro quadrático médio
    pred_y = model.predict(X)
    mse = mean_squared_error(y, pred_y)
    return mse

# Defina o intervalo de unidades de tempo para testar. Depende dos seus dados. 
# Por padrão, testemos de 2 a 100 em passos de 2.
time_units = list(range(2, 101, 2))

performance = []

# Laço sobre as unidades de tempo e avaliar o modelo
for unit in time_units:
    mse = evaluate_model_for_unit_time(unit, training_data)
    performance.append((unit, mse))

# Seleciona a unidade de tempo com o menor MSE
optimal_unit_time, min_mse = min(performance, key=lambda x:x[1])

print(f"A unidade ótima de tempo é: {optimal_unit_time}")

# Plotar o desempenho do modelo versus a unidade de tempo
plt.plot([x[0] for x in performance], [x[1] for x in performance])
plt.xlabel('Unidade de Tempo')
plt.ylabel('MSE')
plt.show()
