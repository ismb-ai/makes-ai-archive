# Transformers.py

# O código-fonte demonstra a implementação de um modelo de aprendizado 
# profundo para séries temporais usando Transformers em Python. Inicialmente, 
# normaliza-se os dados originais com MinMaxScaler para diminuir a variação.
# Após, uma nova série temporal é gerada pela função 'create_dataset', apta 
# para os Transformers. 'look_back' determina quantos tempos anteriores o 
# modelo deve considerar para prever o próximo valor.
#
# A próxima etapa é a definição da arquitetura do modelo Transformer. Compõe-se 
# de blocos encoder de transformers com uma camada de normalização, uma camada 
# de auto-atenção multi-cabeça, e duas camadas de convolução 1D. 
# A atenção multi-cabeça permite considerar diferentes aspectos simultaneamente.
#
# Definido o modelo, separam-se os dados em conjuntos de treinamento e teste.
# Estes são então reformulados para adequar a entrada esperada do Transformer.
#
# Na sequência, declaram-se os hiperparâmetros do modelo, como tamanho da cabeça,
# número de cabeças, dimensionalidade da rede "feed forward", taxa de dropout, 
# e número de blocos transformer. 
#
# Definida a arquitetura e os hiperparâmetros, compila-se o modelo com perda de 
# Erro Quadrático Médio e otimizador Adam.
#
# O modelo é então treinado e avaliado, e a perda de treino e validação é exibida.
# Finalmente, usa-se o modelo para prever no conjunto de testes. Estas previsões 
# são inversamente transformadas para a mesma escala dos dados originais, 
# e plotadas para comparação com os verdadeiros valores da série temporal.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# Normaliza os dados de treinamento usando Min-Max scaling:

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(training_data[['sum_quant_item']])

# Função para criar série temporal adequada para o Transformer:
# Essa função recebe um conjunto de dados e cria subsequências temporais para 
# serem usadas no treinamento do modelo.

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.asarray(dataX), np.asarray(dataY)

# Função para a construção do modelo Transformer:
# Essa função constrói um modelo Transformer com base nos parâmetros fornecidos.

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout_rate, mlp_dropout_rate):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for units in mlp_units:
        x = tf.keras.layers.Dense(units, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(mlp_dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs, outputs)

# Função para a construção do encoder do Transformer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate):
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout_rate
    )(x, x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    res = x + inputs

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# Utilizando a função para criar um novo conjunto de dados:
look_back = 10
X, y = create_dataset(data_scaled, look_back)

# Dividir os dados entre conjunto de treino e teste:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Expandir as dimensões
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Parâmetros para o modelo Transformer
head_size = 64
num_heads = 4
ff_dim = 32
dropout_rate = 0.1
num_transformer_blocks = 2

# Parâmetros para o modelo MLP
mlp_units = [128]
mlp_dropout_rate = 0.1

# Construir o modelo
model = build_model(X_train.shape[1:], head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout_rate, mlp_dropout_rate)

# Compilar o modelo
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

# Treinando o modelo
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, verbose=1)

# Avaliando o modelo
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Fazendo as previsões
predicted_quantity = model.predict(X_test)
predicted_quantity = scaler.inverse_transform(predicted_quantity)

# Plotando os resultados
plt.figure(figsize=(10,6))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Valor Real')
plt.plot(predicted_quantity, color='red', label='Previsões')
plt.title('Previsão do modelo Transformer')
plt.xlabel('Escala de tempo')
plt.ylabel('sum_quant_item')
plt.legend()
plt.show()
