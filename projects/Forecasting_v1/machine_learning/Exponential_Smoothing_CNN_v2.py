# Exponential_Smoothing_CNN_v2.py
#
# O programa a seguir consiste em uma implementação de um modelo híbrido de 
# aprendizado profundo para a análise de séries temporais, combinando o método
# de suavização exponencial com uma Rede Neural de Convolução.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import seaborn as sns

# Set the theme
sns.set_theme(style='whitegrid')

# Preparando os dados

# Nesta etapa, os dados são preparados para a Rede Neural Convolucional (CNN).
# A função `preparar_dados` recebe uma série temporal e um parâmetro de janela.
# O parâmetro de janela define o intervalo de etapas de tempo a serem consideradas 
# na entrada para a modelagem. A função cria uma lista de padrões de entrada/saída 
# da série temporal. São retornados dois arrays Numpy: X como entrada e Y como saída 
# para o modelo CNN. X é redimensionado para o formato tridimensional esperado pelas
# CNNs, ou seja, [amostras, timesteps, features].

def preparar_dados(series, window):
    X, y = [], []
    for i in range(len(series)-window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # Redimensionando para o formato esperado pela CNN
    y = np.array(y)
    return X, y

# Criando a CNN

# A função `criar_cnn` é utilizada para criar um modelo de CNN Sequencial. O modelo 
# consiste em várias camadas, incluindo uma camada Convolucional (Conv1D), uma camada 
# Flatten para transformar os tensores multidimensionais em 1D e duas camadas Densas, 
# onde a última camada contém apenas um neurônio como saída, uma vez que este é um 
# problema de regressão. A função de ativação 'Relu' é utilizada para as camadas 
# ocultas e o modelo é compilado utilizando o otimizador Adam e a função de perda do 
# Erro Médio Quadrado.

def criar_cnn(window):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window, 1)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Aplicando suavização exponencial

# A função `aplicar_suavizacao_exponencial` aplica a suavização exponencial aos dados. 
# Este é um método simples de previsão para dados de séries temporais que cria um modelo
# de média móvel ponderada exponencialmente para os dados. O nível de suavização 
# (alpha) é definido como argumento com padrão 0.6. Valores de fator de suavização 
# (alpha) mais baixos dão mais peso a observações mais antigas, enquanto valores de 
# fator de suavização mais altos dão mais peso a observações recentes.

def aplicar_suavizacao_exponencial(data, alpha=0.6):
    model = SimpleExpSmoothing(data)
    model_fit = model.fit(smoothing_level=alpha, optimized=False)
    return model_fit.fittedvalues

# Carregando dados e aplicando suavização exponencial
# training_data = pd.read_csv('nome_do_arquivo.csv')

training_data_smooth = aplicar_suavizacao_exponencial(training_data['sum_quant_item'])

# Normalizando os dados

# O MinMaxScaler do módulo de pré-processamento sklearn é utilizado para normalizar 
# os dados de entrada para o intervalo [0,1]. Isso ajuda o modelo a convergir mais 
# rapidamente e evita que valores de entrada baixos e altos viciem excessivamente o 
# modelo.

scaler = MinMaxScaler(feature_range=(0, 1))
series = scaler.fit_transform(np.array(training_data_smooth).reshape(-1, 1))

# Preparando os dados para CNN

window = 10
X, y = preparar_dados(series, window)

# Criando e treinando a CNN

# O modelo CNN é criado usando a função `criar_cnn` e depois treinado nos dados 
# preparados por 200 épocas.

model = criar_cnn(window)
model.fit(X, y, epochs=200, verbose=0)

# Previsões

# O modelo faz previsões nos dados treinados e as previsões são transformadas 
# inversamente do range normalizado [0,1] para corresponder à escala dos dados originais.

predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions) # Retornando à escala original para comparar com os dados originais


# Plotando os dados originais e as previsões:
# A última parte do código é dedicada à visualização da série temporal original, junto 
# com a predição, no mesmo gráfico, utilizando matplotlib e seaborn. O gráfico consiste 
# nos dados reais da série temporal e na série temporal prevista, para que o usuário 
# possa ver o quão bem as previsões correspondem aos dados originais. O rótulo do eixo 
# x 'Time', o rótulo do eixo y 'Values' e o título 'Comparison of Original Data and 
# Model Predictions' são dados para o gráfico e uma legenda é exibida.

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(training_data['sum_quant_item'], label='Original Data', linewidth=2)
ax.plot(predictions, label='Model Predictions', linewidth=2)
plt.title('Comparison of Original Data and Model Predictions', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend()
plt.show()
