import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Função para a criação e treinamento do modelo
def create_model(training_data):
    scaler = StandardScaler()
    training_data['scaled_sum_quant_item'] = scaler.fit_transform(training_data[['sum_quant_item']])
    
    X = []
    Y = []
    for i in range(1, len(training_data)):
        X.append(training_data.loc[i-1:i, 'scaled_sum_quant_item'].values)
        Y.append(training_data.loc[i, 'scaled_sum_quant_item'])

    X = np.array(X).reshape((-1, 2, 1))
    Y = np.array(Y)
    
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape=X[0].shape))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # TensorBoard Callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    model.fit(X, Y, batch_size=1, epochs=10, callbacks=[tensorboard_callback])
    
    return model, scaler

# Função para detecção de anomalias
def detect_anomalies(model, training_data, scaler):
    X_test = []
    Y_test = training_data.loc[1:, 'scaled_sum_quant_item'].values

    for i in range(1, len(training_data)):
        X_test.append(training_data.loc[i-1:i, 'scaled_sum_quant_item'].values)
    X_test = np.array(X_test).reshape((-1, 2, 1))

    # Fazer previsões na série temporal
    predictions = model.predict(X_test)
    residuals = Y_test - predictions.reshape(-1)

    # Classificar anomalias
    anomalies = np.where(np.abs(residuals) > 3*np.std(residuals))[0]

    # Substituir anomalias
    training_data.loc[training_data.index[1 + anomalies], 'scaled_sum_quant_item'] = training_data['scaled_sum_quant_item'].median()

    # Reverter escala 
    training_data["clean_sum_quant_item"] = scaler.inverse_transform(training_data[['scaled_sum_quant_item']])

    return training_data

# Função para plotar os dados
def plot_data(training_data, cleaned_data):
    plt.figure(figsize=(12,6))
    plt.plot(training_data['time_scale'], training_data['sum_quant_item'], 'b', label = 'Original data')
    plt.plot(cleaned_data['time_scale'], cleaned_data['clean_sum_quant_item'], 'r', label = 'Cleaned data')
    plt.legend(loc='upper left')
    plt.xlabel('Time scale')
    plt.ylabel('Sum quantity item')
    plt.title('Original vs Cleaned data')
    plt.grid(True)
    plt.show()
    
    # Execução das funções
modelo, escalador = create_model(training_data)
cleaned_data = detect_anomalies(modelo, training_data, escalador)
plot_data(training_data, cleaned_data)
