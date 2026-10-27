# optimal_horizon.py
#
# Este programa realiza a otimização de "walk-forward" em uma série-temporal
# usando o modelo de regressão linear. Através de um laço, implementamos a 
# variação do horizonte de previsão, comparando o erro médio quadrático (MSE) de
# cada horizonte para determinar o melhor valor. No entanto, se o MSE aumentar 5
# vezes consecutivas, o laço é interrompido para evitar ajustes ineficientes.
# Após a otimização, o código imprime o valor do horizonte que resultou no menor
# erro médio quadrático. Por fim, ele apresenta visualmente a performance do
# modelo através de um gráfico, onde podemos analisar o MSE em relação ao 
# horizonte.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Preparação dos dados:
# As variáveis X e y são definidas como arrays do numpy. X conta com os dados
# da coluna 'time_scale' e y com os dados da coluna 'sum_quant_item'.

X = np.array(training_data['time_scale']).reshape(-1, 1)
y = np.array(training_data['sum_quant_item'])

# Inicialização das variáveis:
# - O começo do horizonte de previsão é definido como 1.
# - O erro quadrático médio (MSE) inicial é configurado para 'infinito'.
# - A contagem de MSE aumentados é iniciada em 0.
# - Uma lista vazia chamada 'performance' é criada para armazenar o desempenho 
# do modelo para cada horizonte.

horizon_start = 1
max_mse = float('inf')  # Setting the initial maximum MSE to be infinity
mse_increase_count = 0  # Counter that increments every time the mse increases

performance = []  # Salva a performance do modelo para cada horizonte

# Iteração sobre o horizonte de previsão:
# - O script itera o horizonte de previsão começando do segundo elemento da 
# série temporal, porque o 1o elemento é considerado o começo do horizonte.
# - Para cada iteração, o modelo de regressão linear é ajustado utilizando 
# X e y e o MSE é calculado. Caso o MSE atual seja menor que o MSE máximo
# anterior, o horizonte atual é considerado como o horizonte máximo e o valor 
# do MSE máximo é atualizado.
# - Se o MSE aumentar, a contagem de aumentos do MSE será incrementada em 1.
# - A iteração é interrompida se o MSE aumentar por 5 horizontes consecutivos.

for horizon_end in range(2, len(y)):
    
    # Fit the model for each horizon, and measure its MSE
    model = LinearRegression()
    model.fit(X[horizon_start:horizon_end], y[horizon_start:horizon_end])
    y_pred = model.predict(X[horizon_start:horizon_end])
    mse = mean_squared_error(y[horizon_start:horizon_end], y_pred)
    
    # Store model performance for each step
    performance.append((horizon_end, mse))
    
    # Early stopping condition: If mse is less than the so far found mse,
    # save the horizon and update max_mse.
    if mse < max_mse:
        max_horizon = horizon_end
        max_mse = mse
        mse_increase_count = 0  # Reset counter
    else:
        mse_increase_count += 1  # Increment counter if the mse increased
    
    # Stop if the mse has increased for a certain number of consecutive horizons.
    if mse_increase_count == 5:
        break

# A função, em seguida, imprime o horizonte máximo registrado, i.e., o horizonte
# no qual o modelo apresenta o menor MSE.
print("Maximal horizon: ", max_horizon)

# Por último, o programa plota o desempenho do modelo (MSE) em relação ao 
# horizonte, permitindo uma análise visual do desempenho do modelo.
plt.figure(figsize=(10,5))
plt.plot([x[0] for x in performance], [x[1] for x in performance], label='Linear Model Performance')
plt.xlabel('Horizon')
plt.ylabel('MSE')
plt.title('Model performance vs Horizon')
plt.legend()
plt.show()
