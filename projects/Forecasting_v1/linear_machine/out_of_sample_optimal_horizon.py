# out_of_sample_optimal_horizon.py
#
# Este código é usado para analisar uma série temporal, buscar o horizonte de
# previsão ideal e visualizar os resultados. O primeiro bloco de código define
# duas funções, 'linear_regression_model', que treina um modelo de regressão
# linear a partir de um dataframe, e 'find_optimal_horizon', que itera sobre 
# diferentes horizontes de previsão para encontrar aquele que produz o erro 
# quadrático médio mínimo. A terceira função, 'plot_model_performance', é então 
# usada para traçar o erro de previsão versus o horizonte de previsão. O último 
# bloco de código aplica essas funções nos dados de treinamento, em seguida, 
# imprime o horizonte ideal e plota os resultados. O tamanho da janela e o 
# intervalo do horizonte de previsão são definidos manualmente, então esses 
# devem ser ajustados de acordo com a série temporal a ser analisada.

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# linear_regression_model(df):
# A função 'linear_regression_model' aceita um dataframe como entrada. 
# Extrai duas colunas do dataframe, 'time_scale' e 'sum_quant_item', e as
# utiliza para treinar um modelo de regressão linear. O modelo treinado é então
# retornado pela função.

def linear_regression_model(df):
    X = df['time_scale'].values.reshape(-1,1)
    y = df['sum_quant_item'].values
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# find_optimal_horizon(df, window, horizon_range):
# A função 'find_optimal_horizon' começa extraindo duas colunas, 'time_scale' e
# 'sum_quant_item',  do dataframe e cria um objeto TimeSeriesSplit.
# Esse objeto é usado para fornecer índices para treinar e testar conjuntos que
# mantêm a ordem temporal dos dados.
    
def find_optimal_horizon(df, window, horizon_range):
    X = df['time_scale'].values.reshape(-1,1)
    y = df['sum_quant_item'].values
    tscv = TimeSeriesSplit(n_splits=len(df) - window)
    
    # O código então itera sobre 'horizon_range'. Para cada horizonte, iteramos 
    # com respeito aos conjuntos de treinamento e teste definidos pelo objeto 
    # TimeSeriesSplit.
    
    rmses = []
    for horizon in horizon_range:
        horizon_rmses = []
        for train_index, test_index in tscv.split(X):
            
            # Cria conjuntos de treinamento e teste para X e y, com base nos
            # índices fornecidos e no horizonte atual.
            
            X_train, X_test = X[train_index], X[test_index][:horizon]
            y_train, y_test = y[train_index], y[test_index][:horizon]
            
            # Treina um modelo de regressão linear.
            
            model = linear_regression_model(pd.DataFrame({'time_scale': X_train.flatten(), 
                                                          'sum_quant_item': y_train}))
            
            # Prevê valores para o conjunto de teste X e calcula o erro
            # quadrático médio para essas previsões. O resultado é a raiz 
            # quadrada do erro médio e é armazenado na lista 'horizon_rmses'.
            
            y_pred = model.predict(X_test)

            # Depois que todos os conjuntos de treinamento e teste foram usados,
            # calcula a média dos erros quadráticos e armazena o resultado na
            # lista 'rmses'.
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            horizon_rmses.append(rmse)
        
        rmses.append(np.mean(horizon_rmses))

    # Retorna o horizonte no qual o erro quadrático médio era mínimo e a lista
    # completa de erros quadráticos para cada horizonte.
    
    return horizon_range[np.argmin(rmses)], rmses

# plot_model_performance(horizon_range, rmses):
# A função 'plot_model_performance' aceita um 'horizon_range' e uma lista de
# erros quadráticos e traça o erro versus o horizonte.

def plot_model_performance(horizon_range, rmses):
    plt.figure(figsize=(10,6))
    plt.plot(horizon_range, rmses, marker='o', markersize=5)
    plt.title('Model Performance vs Horizon Range')
    plt.xlabel('Horizon')
    plt.ylabel('Root Mean Squared Error')
    plt.grid(True)
    plt.show()

# Definimos tamanho da janela e intervalo do horizonte de previsão.
# Ajuste conforme a série-temporal a ser analisada.

window = 100
horizon_range = range(1, 31)

# Neste último bloco de código, a função 'find_optimal_horizon' é chamada para
# determinar o horizonte ideal para as previsões. Este horizonte e a lista de
# erros quadráticos são então usados como entrada para a função
# 'plot_model_performance' para visualizar os resultados.

optimal_horizon, rmses = find_optimal_horizon(training_data, window, horizon_range)
print('Optimal horizon:', optimal_horizon)
plot_model_performance(horizon_range, rmses)
