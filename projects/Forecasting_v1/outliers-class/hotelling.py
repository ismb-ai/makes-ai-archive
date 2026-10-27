# hotelling.py
# Este código aplica a estatística Hotelling T para detectar e remover outliers
# dos dados de uma série temporal. 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2

# prepare_training_data()
# Prepara os dados de treinamento selecionando as colunas necessárias ('time_scale' e 'sum_quant_item'),
# eliminando linhas com valores faltantes e, em seguida, organizando os dados com base na coluna 'time_scale'.

def prepare_training_data():
    global training_data
    training_data = training_data[['time_scale','sum_quant_item']]
    training_data = training_data.dropna()
    training_data.sort_values(['time_scale'], axis = 0, inplace = True)  
    return training_data

# train_linear_regression(X_train, y_train)
# Treina um modelo de regressão linear usando o conjunto de treinamento das variáveis
# independentes e dependentes (X_train e y_train, respectivamente).

def train_linear_regression(X_train, y_train):
    model = LinearRegression()  
    model.fit(X_train, y_train) 
    return model

# detect_outliers_hotelling(X_train, y_train, training_data)
# Detecta outliers usando a estatística de Hotelling T. Treina um modelo de regressão linear, 
# calcula resíduos e a estatística Hotelling T ao quadrado para cada resíduo.
# Outliers são resíduos com estatística T ao quadrado superior a um limiar,
# 1-0.05 percentil da distribuição chi-quadrado com 1 grau de liberdade.
# Outliers são removidos dos dados.

def detect_outliers_hotelling(X_train, y_train, training_data):
    model = train_linear_regression(X_train, y_train)
    residuals = pd.Series(y_train.flatten()) - model.predict(X_train).flatten()

    residuals_mean = residuals.mean()
    residuals_std = residuals.std()
    hotelling_t_square = ((residuals - residuals_mean) ** 2) / (residuals_std ** 2)

    outlier_threshold = chi2.ppf((1-0.05), df=1)

    outliers = hotelling_t_square[hotelling_t_square > outlier_threshold]
    
    print('Number of outliers:', len(outliers))
    print('Outliers:\n', outliers)
    
    cleaned_data = training_data.loc[~training_data.index.isin(outliers.index)]
    return cleaned_data

# plot_data(original_data, cleaned_data)
# Gera um gráfico que mostra dados originais e limpos. Outliers são enfatizados no gráfico para comparação.

def plot_data(original_data, cleaned_data):
    plt.figure(figsize=(14, 6))
    plt.plot(original_data[['time_scale']], original_data[['sum_quant_item']], 'bo', label='Original')
    plt.plot(cleaned_data[['time_scale']], cleaned_data[['sum_quant_item']], 'ro', label='Cleaned')
    plt.legend()
    plt.show()


# main()
# Função principal que chama todas as outras funções. Prepara dados de treinamento,
# separa variáveis independentes e dependentes em conjuntos de treinamento e teste. A função de detecção
# de outliers é chamada com o conjunto de treinamento. Dados limpos são plotados e salvos em arquivo CSV.

def main():
    training_data = prepare_training_data()
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    cleaned_data = detect_outliers_hotelling(X_train, y_train, training_data)
    plot_data(training_data, cleaned_data)

    cleaned_data.to_csv('cleaned1_data.csv', index=False)
    cleaned1_data = pd.read_csv('cleaned1_data.csv')

if __name__ == "__main__":
    main()
