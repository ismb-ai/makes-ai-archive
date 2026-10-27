# out_of_sample_unit_of_time.py
#
# Neste código, implementa-se a função 'optimal_time_unit' para encontrar a 
# melhor unidade de tempo para o modelo da série-temporal. Inicia-se com 
# 'errors', uma lista vazia que armazenará os erros quadrados médios. 
# Verifica-se se a unidade de tempo é menor que o total de pontos na série e 
# aplica-se a janela deslizante para calcular a média dos pontos de dados. 
# Os conjuntos X e Y são divididos em treino e testes e um modelo de regressão 
# linear é ajustado, usadado para fazer previsões e calcular o erro quadrado 
# médio. O erro para cada unidade de tempo é armazenado e, ao final, retorna-se 
# a unidade com o menor erro. Um gráfico é gerado para observação dos erros 
# versus o tamanho da janela, e a função é chamada levando em consideração 
# unidades de tempo entre 1 e 60.

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# optimal_time_unit(df, min_unit, max_unit):
# Neste trecho, a função optimal_time_unit é definida para determinar a melhor
# unidade de tempo para modelar a série temporal. A função aceita um dataframe 
# (df), uma unidade de tempo mínima (min_unit) e uma unidade de tempo máxima
# (max_unit). A variável "errors" é inicializada como uma lista vazia para 
# armazenar os erros quadrados médios de cada modelo gerado.

def optimal_time_unit(df, min_unit, max_unit):
    errors = []
    
    # O seguinte laço que percorre todas as possíveis unidades de tempo do 
    # intervalo definido.
    for unit in range(min_unit, max_unit+1):
        
        # Verifica-se se a unidade de tempo é menor que a quantidade total de 
        # pontos de dados na série. Se for maior, o laço é interrompido.
        if unit < len(df):
            
            # Rolling Window 
            # Neste trecho, o código cria uma janela deslizante (rolling window)
            # para calcular a média dos pontos de dados da série temporal. 
            # X é um array da média da quantidade somada de itens e Y é um array
            # do valor da quantidade somada de itens, começando a partir da 
            # unidade definida.
            X = np.array([df['sum_quant_item'].rolling(unit).mean().iloc[unit-1:] for i in range(unit, len(df))])
            Y = df['sum_quant_item'].values[unit:]
            
            # Divisão Treino-teste
            # Aqui, os arrays X e Y são divididos em conjuntos de treinamento e
            # testes usando a função train_test_split. 80% dos dados serão
            # usados para treinamento e 20% para testes.
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

            # Criar e Treinar o Modelo 
            # Neste trecho, o código cria uma instância da classe 
            # LinearRegression e realiza o treinamento do modelo.
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Fazendo previsões e avaliação
            # Aqui, o modelo treinado é usado para fazer previsões no conjunto
            # de testes. O erro quadrado médio (MSE) é calculado comparando os
            # valores previstos com os valores reais do conjunto de teste.
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            errors.append((unit, mse))

    # Plotando Erros
    # Este bloco é usado para plotar os erros MSE em relação ao tamanho da
    # janela.
    plt.plot(*zip(*errors))
    plt.title('Model performance (MSE) versus Window Size')
    plt.xlabel('Window Size')
    plt.ylabel('Mean Squared Error')
    plt.show()

    # Unidade de Tempo Ideal
    # Este é o último passo da função que identifica a unidade de tempo com o
    # menor erro quadrado médio entre todas as unidades de tempo examinadas.
    optimal_unit = min(errors, key=lambda x:x[1])[0]
    print(f"Optimal unit of time: {optimal_unit}")

# Chame função-principal
# Neste exemplo, a função é chamada para buscar a unidade de tempo ótima nos
# dados de treinamento, considerando unidades de tempo entre 1 and 60.
optimal_time_unit(training_data, 1, 60)
