# Anomaly_Detection_Exponential_Smoothing_RNN.py
#
#     No âmbito da análise de séries temporais, o código supracitado descreve um 
# método híbrido que integra suavização exponencial e redes neurais recorrentes 
# do tipo LSTM para detecção e tratamento de anomalias. Inicialmente, a função 
# `suavizacao_exponencial` é utilizada para aplicar o método de suavização expo-
# nencial SimpleExpSmoothing aos dados da série temporal, visando suavizar flu-
# tuações e destacar tendências mais persistentes. Opta-se por fixar o parâmetro 
# de suavização (alpha) em 0.2, garantindo, desta forma, que os valores mais re-
# centes têm um peso adequado no cálculo da média móvel ponderada exponencial-
# mente.
#     A função `preparar_dados` transforma o conjunto de dados da série temporal 
# para o formato `float32` e depois os redimensiona a uma estrutura bidimensio-
# nal, para submetê-los a normalização via MinMaxScaler. Este escalonamento nor-
# maliza o intervalo dos valores, garantindo que conflitos de escala não compro-
# metam o desempenho do modelo neural. A subsequente função `separar_dados` di-
# vide o dataset normalizado em subconjuntos de treino e teste, provendo a base 
# para uma validação rigorosa do modelo.
#     A função `criar_dataset` manipula os conjuntos de treino e teste, estrutu-
# rando os dados em uma sequência em que a entrada para a LSTM consiste em 
# `look_back` períodos temporais e a saída corresponde ao período seguinte. Este 
# rearranjo é imperativo para treinar a rede a reconhecer padrões sequenciais e 
# realizar previsões futuras. O procedimento de remodelação subsequente trans-
# forma os dados em um formato de três dimensões expositoras, compatível com as 
# expectativas da rede neural recorrente.
#     O modelo LSTM, que é gerado pela função `criar_modelo`, consiste em uma 
# camada LSTM com quatro unidades seguida por uma camada densa de projeção única, 
# todo o sistema sendo compilado com a função de perda de erro quadrado médio e 
# o otimizador 'adam'. Após a construção, o modelo é treinado com base no con-
# junto de treino, usando o número de épocas e o tamanho do lote definidos como 
# parâmetros. As previsões realizadas pelo modelo nos subsets de treino e teste 
# são transformadas inversamente para a escala original, permitindo uma compara-
# ção realística com os valores efetivos da série.
#     A remoção de anomalias é realizada pela função `remover_anomalias`, que 
# estabelece um limiar estatístico baseado no desvio padrão das diferenças entre 
# os valores previstos e os reais. Os dados que excedem tal limiar são classifi-
# cados como anomalias e suprimidos do conjunto de dados. Finalmente, a função 
# `plotar_grafico` exibe uma representação gráfica abarcando os dados originais, 
# as previsões do modelo e o conjunto resultante após a depuração de anomalias, 
# fornecendo um mecanismo visual de confirmação da eficácia do processo.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model

# Suavização Exponencial:
# A função `suavizacao_exponencial` aplica a suavização exponencial a uma série 
# temporal utilizando a biblioteca `statsmodels`. Essa técnica é útil para 
# reduzir ruídos e capturar tendências, sendo implementada pela classe 
# `SimpleExpSmoothing`. O modelo é ajustado aos dados com um nível de suavização
# de 0.2. O método `fit()` é utilizado para ajustar o modelo e o método 
# `predict()` para realizar previsões.

def suavizacao_exponencial(serie):
    # Suavização exponencial da série temporal
    modelo = SimpleExpSmoothing(serie)
    modelo_ajustado = modelo.fit(smoothing_level=0.2, optimized=False)
    return modelo_ajustado.predict(len(serie), len(serie))

# Preparação de Dados:
# A função `preparar_dados` é responsável por preparar a série temporal para que 
# seja consumida pela rede neural. Isso envolve converter os dados para o tipo
# `float32`, remodelar o conjunto de dados em uma estrutura bidimensional e nor-
# malizar os valores para o intervalo [0, 1] usando a classe `MinMaxScaler` da 
# biblioteca `sklearn`. Este passo é importante para melhorar a eficiência do 
# treinamento da rede neural.

def preparar_dados(dados_treinamento):
    # Preparar os dados para a rede neural
    dataset = dados_treinamento['sum_quant_item'].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler
    
# Separação de Dados:
# A função `separar_dados` divide o conjunto de dados em partes para treino e 
# teste, definindo 67% dos dados para treino e o restante para teste. Esta di-
# visão é crucial para evitar overfitting e avaliar a performance do modelo.    

def separar_dados(dataset):
    # Divide os dados em conjunto de treinamento e teste
    tamanho_treinamento = int(len(dataset) * 0.67)
    treinamento, teste = dataset[0:tamanho_treinamento,:], dataset[tamanho_treinamento:len(dataset),:]
    return treinamento, teste
    
# Criação da Estrutura de Dados:
# A função `criar_dataset` ajusta os dados para que sejam compatíveis com a en-
# trada esperada pelo modelo LSTM, criando estruturas de `X` e `Y`. Cada elemen-
# to de `X` é um vetor com 'look_back' observações anteriores e cada correspon-
# dente elemento de `Y` é o valor subsequente que queremos prever, fornecendo o 
# formato necessário para a aprendizagem temporal.    

def criar_dataset(dataset, look_back=1):
    # Cria um novo dataset apropriado para o modelo LSTM
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
    
# Criação do Modelo:    
# Em `criar_modelo`, um modelo LSTM é formulado usando o framework Keras. A rede 
# contém uma camada LSTM e uma camada densa (`Dense`) para a saída. A função de 
# perda 'mean_squared_error' e o otimizador 'adam' são escolhidos durante a com-
# pilação do modelo.    

def criar_modelo(look_back):
    # Cria o modelo LSTM
    modelo = Sequential()
    modelo.add(LSTM(4, input_shape=(1, look_back)))
    modelo.add(Dense(1))
    modelo.compile(loss='mean_squared_error', optimizer='adam')
    return modelo

# Remoção de Anomalias:
# A função `remover_anomalias` utiliza um limiar, definido como três vezes o 
# desvio padrão das diferenças entre previsões e valores reais, para identificar 
# e remover anomalias. Se a diferença absoluta ultrapassar esse limiar, o valor 
# é considerado uma anomalia.

def remover_anomalias(predicted_diff, actual, threshold):
    # Remove anomalias dos dados
    dados_limpos = actual[np.abs(predicted_diff - actual) < threshold]
    return dados_limpos
    
# Visualização de Dados:
# A função `plotar_grafico` é responsável pela visualização que plota os dados 
# originais, as previsões do treino e teste, bem como os dados limpos.    

def plotar_grafico(dados_originais, previsao_treino, previsao_teste, dados_limpos):
    # Plota os dados originais, a previsão e os dados limpos
    plt.figure(figsize=(15, 6))
    plt.plot(dados_originais, color='blue')
    plt.plot(previsao_treino, color='orange')
    plt.plot(previsao_teste, color='green')
    plt.plot(dados_limpos, color='red')
    plt.show()

# Bloco Principal:
# O bloco final do código sequencia as funções definidas anteriormente para 
# realizar a detecção de anomalias. 
#     Primeiramente, a suavização exponencial é aplicada à série temporal. 
# Em seguida, os dados são preparados, separados em conjuntos de treino e teste, 
# e transformados para se adequarem à entrada do LSTM. O modelo então é cons-
# truído, treinado e realiza previsões sobre os dados de treino e teste. A dife-
# rença entre as previsões é calculada e usada para detectar e remover as anomalias. 
# Por fim, um gráfico é gerado para ilustrar o resultado do processo de detecção 
# e tratamento das anomalias.

# Carrega os dados
dados_treinamento = data

# Aplica a suavização exponencial
dados_treinamento['sum_quant_item_es'] = suavizacao_exponencial(dados_treinamento['sum_quant_item'])

# Prepara os dados para a rede neural
dataset, scaler = preparar_dados(dados_treinamento)

# Separa os dados em treino e teste
train, test = separar_dados(dataset)

# Cria um novo dataset para o modelo LSTM
look_back = 1
trainX, trainY = criar_dataset(train, look_back)
testX, testY = criar_dataset(test, look_back)

# Remodela para [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Cria o modelo LSTM
model = criar_modelo(look_back)

# Treina o modelo
model.fit(trainX, trainY, epochs=20, batch_size=1)

# Faz previsões com o modelo
trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Monta os dados previstos para plotagem
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Remove anomalias dos dados
predicted_diff = np.r_[0, np.diff(trainPredictPlot[:,0])]
threshold = np.std(predicted_diff) * 3
cleaned_data = remover_anomalias(predicted_diff, training_data['sum_quant_item_es'], threshold)

# Plota os dados originais, a previsão e os dados limpos
plotar_grafico(scaler.inverse_transform(dataset), trainPredictPlot, testPredictPlot, cleaned_data)
