# Anomaly_Detection_RNN.py
#
# O código-fonte a seugir implementa uma abordagem de aprendizado profundo usan-
# do Recurrent Neural Network (RNN) para detectar e substituir anomalias em uma
# série temporal, descrita pelo dataframe "training_data". Para a construção e 
# treinamento da RNN, é utilizada a função `create_and_train_RNN(data, epochs)` 
# que inicialmente normaliza os dados com a classe `StandardScaler` do pacote 
# sklearn. Os dados normalizados são então formatados em sequências de entrada 
# compostas por cinco elementos consecutivos da série temporal. O modelo de RNN 
# é montado com três camadas de memória de curto e longo prazo (LSTM) através da
# biblioteca Keras, intercaladas por camadas de regularização ou "Dropout". 
#     Em seguida, a detecção de anomalias é realizada pela função 
# `anomaly_detection(model, sc, data, threshold)`, que procede na identificação 
# de elementos nos dados originais que divergem significativamente das previsões
# do modelo treinado. Se identificada uma diferença maior do que um determinado 
# limiar, o ponto é marcado como uma anomalia. Na parte principal do código, 
# inicialmente é feita uma preparação dos dados do dataframe "training_data", 
# seguida pelo treinamento do modelo de RNN. Posteriormente, é realizada a 
# detecção de anomalias sobre esses mesmos dados. Uma vez identificadas, as 
# anomalias são removidas do dataframe original, originando um novo conjunto de 
# dados sem as anomalias. Por fim, os valores originais, valores previstos e as 
# anomalias identificadas são visualizadas por meio de um gráfico. 
#     O código, portanto, emprega técnicas modernas de aprendizado profundo para 
# detecção e tratamento de anomalias em séries temporais.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Criação e Treinamento do RNN:
# A função `create_and_train_RNN(data, epochs)` cria e treina uma RNN. A função 
# começa por normalizar os dados de entrada através da classe StandardScaler do 
# sklearn. Em seguida, os dados transformados são formatados para que cada ele-
# mento da sequência de entrada (X_train) seja um array de 5 elementos da série 
# temporal. O respectivo elemento na sequência de saída (Y_train) é o sexto 
# elemento da série temporal. 
#     O modelo de rede neural recorrente é então construído usando a biblioteca 
# Keras. O modelo consiste em três camadas LSTM (Long Short-Term Memory) e cada 
# camada LSTM é seguida por uma camada de Dropout, para evitar o sobreajuste.
# A última camada é uma camada totalmente conectada (Dense) que produz a saída 
# do modelo. O modelo é então compilado com o otimizador 'adam' e a função de 
# perda 'mean_squared_error' e treinado usando os dados de entrada por um deter-
# minado número de épocas.

def create_and_train_RNN(data, epochs):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    X_train=[]
    Y_train = []
    for i in range(5,data.shape[0]):
        X_train.append(data[i-5:i,0])
        Y_train.append(data[i,0])
    X_train,Y_train=np.array(X_train),np.array(Y_train)
    X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train,Y_train,epochs = epochs,batch_size=32)

    return model, sc

# Detecção de Anomalias:
# A função `anomaly_detection(model, sc, data, threshold)` realiza a detecção de 
# anomalias. A função começa construindo a sequência de entrada (X_test) da 
# mesma maneira que foi feita para os dados de treinamento. Em seguida, o modelo
# RNN é usado para produzir uma previsão para cada elemento em X_test. 
#     A diferença absoluta entre os dados originais e os dados previstos é cal-
# culada. Se a diferença é maior que um limiar especificado, o ponto correspon-
# dente é considerado uma anomalia.

def anomaly_detection(model, sc, data, threshold):
    X_test = []
    for i in range(5,len(data)):
        X_test.append(data[i-5:i,0])
    X_test = np.array(X_test)

    predicted = model.predict(X_test)
    predicted = sc.inverse_transform(predicted)

    diff = np.abs(data[5:] - predicted.flatten())
    anomalies_idx = np.where(diff > threshold)[0]

    return anomalies_idx, predicted

# main():
# Na parte principal do script, primeiro os dados de 'training_data' são 
# extraídos e convertidos para float32. O modelo é então treinado chamando a 
# função `create_and_train_RNN(data, epochs = 100)`. Uma vez que o modelo foi 
# treinado, a detecção de anomalias é realizada chamando a função 
# `anomaly_detection(model, sc, data, threshold = 0.05)`.
#     As anomalias identificadas são então removidas do 'training_data' para 
# criar um novo dataframe sem as anomalias.  

data = training_data['sum_quant_item'].values.astype('float32').reshape(-1,1)
model, sc = create_and_train_RNN(data, epochs = 100)
anomalies_idx, predicted = anomaly_detection(model, sc, data, threshold = 0.05)
temporary_data = training_data.drop(anomalies_idx+5, axis = 0)

# Finalmente, os dados originais, dados previstos e anomalias são plotados para 
# comparação visual. Isso permite que se avalie a eficácia do modelo em capturar 
# a estrutura dos dados e a eficácia da detecção de anomalias.

plt.figure(figsize = (14,7))
plt.plot(training_data['time_scale'], training_data['sum_quant_item'], color = 'blue', label = 'Original')
plt.plot(training_data['time_scale'][5:], predicted.flatten(), color = 'green', label = 'Predicted')
plt.scatter(training_data['time_scale'].iloc[anomalies_idx+5], training_data['sum_quant_item'].iloc[anomalies_idx+5], color = 'red', label = 'Anomalies')
plt.title('Original x Predicted x Anomalies')
plt.xlabel('Time Scale')
plt.ylabel('Sum Quant Item')
plt.legend()
plt.show()
