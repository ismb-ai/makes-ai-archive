# takens.py
#
# Este código implementa a técnica de embedding de Takens para modelar 
# uma série-temporal. Ele cria uma série "atrasada" de dimensões especificadas 
# e calcula o erro quadrático médio entre previsões realizadas por um modelo
# KNN treinado nessa série e os valores originais, plotando os resultados.

# Importando bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# delay_embedding(series, dimensions, delay=1)
# Função para criar o embedding através de um delay.
def delay_embedding(series, dimensions, delay=1):
    # Cria uma matriz de zeros com o número de linhas determinado pela série menos
    # o produto do delay e as dimensões menos um, e o número de colunas igual
    # ao número de dimensões.
    te = np.zeros((len(series) - (dimensions - 1) * delay, dimensions))
    
    # Laço para preencher a matriz criada anteriormente com os valores da série-temporal,
    # atrasados em um número de passos igual ao produto do delay e o número de dimensões. 
    for i in range(dimensions):
        te[:, i] = series[i * delay:len(series) - (dimensions - 1) * delay + i * delay]
    
    return te

# Função para plotar o gráfico
def plot_takens_embedding(training_data, dimensions, delay):
    # Gerar o embedding
    te = delay_embedding(training_data['sum_quant_item'].values, dimensions, delay)
    
    # Criar o plot
    plt.figure(figsize=(10, 6))
    plt.plot(training_data['time_scale'].values[:len(te)], te[:, 0], label='Dados originais')
    plt.plot(training_data['time_scale'].values[:len(te)], te[:, 1], label='Embedding')
    
    plt.xlabel('time_scale')
    plt.ylabel('sum_quant_item')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Função para plotar o erro quadrático médio
def plot_mse_takens_embedding(training_data, dimensions, max_delay):
    mse_list = []
    delay_list = range(1, max_delay+1)
    
    for delay in delay_list:
        # Gerar o embedding
        te = delay_embedding(training_data['sum_quant_item'].values, dimensions, delay)
        
        # Realizar uma previsão proóxima por vizinho
        knn = KNeighborsRegressor(n_neighbors=1)
        
        # Treinar e prever em todas, exceto a última linha de 'te'
        knn.fit(te[:-1], te[1:,0])
        y_pred = knn.predict(te[:-1])
        
        # Calcular o erro quadrático médio
        mse = mean_squared_error(te[1:,0], y_pred)
        mse_list.append(mse)
        
    # Criar o plot
    plt.figure(figsize=(10, 6))
    plt.plot(delay_list, mse_list, label='MSE')
    plt.xlabel('Delay')
    plt.ylabel('MSE')
    plt.title('Erro quadrático médio para diferentes atrasos')
    plt.legend()
    plt.grid(True)
    plt.show()

# Gera e plota o embedding de Takens
dimensions = 2 # ajustar conforme necessário! 
delay = 5 # ajustar conforme necessário! 
plot_takens_embedding(training_data, dimensions, delay)

# Computa e plota o MSE do embedding de Takens
dimensions = 2 # ajustar conforme necessário! 
max_delay = 50 # ajustar conforme necessário!
plot_mse_takens_embedding(training_data, dimensions, max_delay)
