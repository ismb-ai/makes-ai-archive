# Load training data and decide which SVR (linear or gaussian kernel) to use

import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Loading the CSV file
data = pd.read_csv("/Users/igormol/Downloads/vendas_20180102_20220826/training_data.csv")

# Printing the structure of the data
print(data.info())

# Printing the content of the data
print(data)

# Function to train and find mse of a SVR model
def train_and_evaluate_svr(kernel, x, y):
    svr = SVR(kernel=kernel)
    svr.fit(x, y)
    y_pred = svr.predict(x)
    mse = mean_squared_error(y, y_pred)
    return svr, mse

# Allowing plots to appear in Jupyter
# %matplotlib inline  # Jupyter-only display directive; plt.show() is used below.

# Getting the independent and dependent variables
X = data[['time_scale']]
y = data['sum_quant_item']

# Train and evaluate the linear SVR
svr_linear, mse_linear = train_and_evaluate_svr('linear', X, y)

# Train and evaluate the gaussian SVR
svr_gaussian, mse_gaussian = train_and_evaluate_svr('rbf', X, y)

# Select the best model
best_model, best_kernel = (svr_linear, 'linear') if mse_linear < mse_gaussian else (svr_gaussian, 'rbf')
print('Best model is with',best_kernel,'kernel')

# Plot original data
plt.scatter(X, y, color='red', label='Original data')

# Plot linear SVR
y_linear = svr_linear.predict(X)
plt.plot(X, y_linear, color='blue', label='SVR Linear')

# Plot gaussian SVR
y_gaussian = svr_gaussian.predict(X)
plt.plot(X, y_gaussian, color='green', label='SVR Gaussian')

plt.title('SVR prediction')
plt.xlabel('Time scale')
plt.ylabel('Sum of quantity items')
plt.legend()
plt.show()
