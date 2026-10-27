# exponential smoothing plus RNN

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def exponential_smoothing(series, alpha):
    model = SimpleExpSmoothing(series)
    model_fit = model.fit(smoothing_level=alpha, optimized=False)
    return model_fit.fittedvalues

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)        
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
data_ES = exponential_smoothing(training_data['sum_quant_item'], 0.3)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_ES.values.reshape(-1, 1))

X, y = create_dataset(data_scaled, data_scaled, time_steps)

test_size = int(len(X) * 0.2)
X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)

y_pred = model.predict(X_test)

y_train_inv = scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = scaler.inverse_transform(y_pred)

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Sales (sum_quant_item)')
plt.xlabel('Time Step')
plt.legend()
plt.show()
