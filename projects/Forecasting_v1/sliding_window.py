from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

def sliding_window(training_data, window, horizon):
    predictions = []  
    for i in range(window, len(training_data)-horizon):
        # train a gaussian SVR model with the current sliding window of data 
        X_train = training_data['time_scale'].iloc[i-window:i].values.reshape(-1,1)
        y_train = training_data['sum_quant_item'].iloc[i-window:i]
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        model.fit(X_train, y_train)
        
        # predicting using model
        X_test = training_data['time_scale'].iloc[i:i+horizon].values.reshape(-1,1)
        prediction =  model.predict(X_test)
        predictions.append(prediction[0])  # assume horizon=1 for simplicity
    return predictions

# if your window size is 50 and horizon = 1
window = 50
horizon = 1
predictions = sliding_window(training_data, window, horizon)

# Plot actual against predicted values
original_value = training_data['sum_quant_item'].values[window:len(training_data)-horizon]
plt.plot(original_value, label='Original Data')
plt.plot(predictions, color='red', label='Predicted Data')
plt.legend()
plt.show()
