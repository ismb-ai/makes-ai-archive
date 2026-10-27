from sklearn.metrics import mean_squared_error
import numpy as np

def perform_walk_forward_optimization(training_data, min_horizon, max_horizon, horizon_interval, window_size):
    horizons = range(min_horizon, max_horizon, horizon_interval)
    lowest_rmse = float("inf")

    for horizon in horizons:
        rmse = walk_forward_validation(training_data, horizon, window_size)
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            optimal_horizon = horizon

    print(f'The optimal horizon is: {optimal_horizon}')

def walk_forward_validation(data, horizon, window):
    X = training_data.loc[:, 'time_scale'].values.reshape(-1,1)
    Y = training_data.loc[:, 'sum_quant_item'].values

    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

    history = [x for x in X[:window]]
    testing = [x for x in X[window:]]

    predictions = list()
    
    # walk-forward validation
    for t in range(len(testing)):
        model = svr_rbf.fit(history, Y[:len(history)])
        yhat = model.predict(testing[t].reshape(1, -1))
        predictions.append(yhat)
        history.append(testing[t])
    error = mean_squared_error(Y[window:], predictions, squared=False)

    return error

perform_walk_forward_optimization(training_data, min_horizon=3, max_horizon=15, horizon_interval=1, window_size=30)
