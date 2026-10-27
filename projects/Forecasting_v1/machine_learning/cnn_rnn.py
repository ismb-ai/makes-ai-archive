# cnn_rnn.py
#
# Este código define uma abordagem de modelo híbrido de 
# Rede Neural Convolucional (CNN) e Rede Neural Recorrente 
# (RNN) para analisar dados de séries temporais. 

from tensorflow.keras.layers import Dense, LSTM, Flatten, Conv1D, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Função shift:
# Esta função é usada para criar lags nos dados de série 
# temporal. Na análise de séries temporais, a introdução de 
# lags como features é uma técnica comum onde os valores 
# passados (valores defasados) são usados para prever o valor 
# futuro. Aqui a função 'shift' permite a criação de diferentes 
# defasagens em variáveis específicas, criando uma coluna para 
# cada desfasagem.

def shift(data, num_shifts, y_var):
    data = data.copy()
    for i in range(1, num_shifts+1):
        data[f'shifted_{y_var}_{i}'] = data[y_var].shift(i)
    data = data.dropna().reset_index(drop=True)
    return data

# Função split_sequences:
# Esta função divide a sequência dada em um conjunto de padrões 
# de entrada/saída que podem ser fornecidos ao modelo. Para cada 
# ponto na sequência, considera os n_steps anteriores como uma 
# sequência de entrada e o ponto atual como a sequência de saída. 
# Isso forma nossos 'X' (entrada) e 'y' (saída) para o modelo.

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences)-1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Função create_model:
# Esta função cria um híbrido de modelos CNN e LSTM. Neste modelo, 
# aplica-se primeiro a convolução 1D usando 'Conv1D' que desliza 
# através da sequência de entrada e descobre padrões. Em seguida, 
# a camada 'MaxPooling1D' reduz a dimensionalidade pela metade. 
# O LSTM então considera a sequência de dados para prever o próximo 
# valor. Uma camada de Dropout é adicionada para evitar o 
# sobreajuste do modelo. Finalmente, 'Dense' é a camada de saída 
# que prevê um único valor na sequência.

def create_model(n_steps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Funções plot_prediction e plot_error:
# Estas funções são usadas para visualizar os resultados. A função 
# 'plot_prediction' traça a comparação entre os valores originais e 
# previstos. A função 'plot_error' mostra o Erro Quadrático Médio 
# em cada época durante o treinamento.

def plot_prediction(y, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y, 'b', label="Original")
    plt.plot(y_pred, 'r', label="Predicted")
    plt.title('Original vs Predicted')
    plt.ylabel('sum_quant_item')
    plt.xlabel('time_step')
    plt.legend()
    plt.show()

def plot_error(error):
    plt.figure(figsize = (10,5))
    plt.plot(error)
    plt.title('MSE Trend Over Epochs')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('No. epoch')
    plt.show()

# Em seguida, o código define 3 etapas dos dados de entrada a partir dos 
# dados da série temporal, determina o número de características 
# para o modelo e define a variável alvo.

n_steps = 3
n_features = training_data.shape[1]
y_var = 'sum_quant_item'
training_data = shift(training_data, n_steps, y_var)

# O 'MinMaxScaler' normaliza os valores entre a faixa de 0 e 1. 
# 'fit_transform' ajusta o MinMaxScaler nos dados fornecidos e 
# depois transforma os dados de acordo.

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_data = scaler.fit_transform(training_data.values)

# Os dados são então divididos em sequências para treinar o modelo 
# e o modelo CNN-LSTM é criado e treinado nas sequências.
# O código também avalia o modelo nos dados de treinamento e mostra como 
# a previsão do modelo se compara aos dados reais e também mostra o 
# erro quadrático médio em cada época.

X, y = split_sequences(scaled_training_data, n_steps)

model = create_model(n_steps, n_features)

history = model.fit(X, y, epochs=200, verbose=0)

training_pred = model.predict(X)

mse = mean_squared_error(y, training_pred)

plot_prediction(y, training_pred)

plot_error(history.history['loss'])
