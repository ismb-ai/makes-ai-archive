# isolation_forest.py
# Este código usa o método Isolation Forest em conjunto com Z-scores para detectar e eliminar outliers 
# de uma série temporal. 

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt

# detect_outliers_isolationforest(data, z_score_threshold, contamination) 
# Detecta outliers nos dados. Inicialmente, padroniza os dados com Z-score. 
# Em seguida, inicializa o Isolation Forest e ajusta o modelo para detectar outliers. 
# Isolation Forest é um algoritmo de aprendizado de máquina para detecção de anomalia, 
# isolando os outliers no conjunto de dados. A função adiciona uma coluna ao dataframe 
# indicando se cada registro é um outlier (1) ou não (-1).

def detect_outliers_isolationforest(data, z_score_threshold, contamination):
    # Standardize data using Z-score
    data['z_score'] = np.abs((data['sum_quant_item'] - data['sum_quant_item'].mean()) / data['sum_quant_item'].std())

    # Initialize and fit IsolationForest to detect outliers
    IF = IsolationForest(contamination=contamination)
    outliers = IF.fit_predict(data[['z_score']])
    
    # Identify and separate outliers & inliers
    data['outliers'] = outliers

    # Return the dataframe with an extra column of inlier(-1)/outlier(1) labels
    return data

# plot_data(original, cleaned)
# Gera um gráfico para visualizar os dados originais e os dados limpos (sem outliers).
# Facilita a visualização e comparação dos dados.

def plot_data(original, cleaned):
    plt.figure(figsize=(10,6))
    plt.plot(original.index, original['sum_quant_item'], 'b-', label = 'Original')
    plt.plot(cleaned.index, cleaned['sum_quant_item'], 'r-', label = 'Cleaned')
    plt.legend(loc='best')
    plt.title('Original vs Cleaned Data')
    plt.show()

# apply_outlier_detection(z_score_threshold, contamination, training_data) 
# Função principal que aplica a detecção de outliers. Detecta outliers e imprime 
# o número de outliers e as observações correspondentes. Mantém apenas os registros que 
# não são outliers (inliers) para obter os dados limpos. Em seguida, chama plot_data para 
# visualizar os dados originais e limpos e retorna os dados limpos.

def apply_outlier_detection(z_score_threshold, contamination, training_data):
    result = detect_outliers_isolationforest(training_data.copy(), z_score_threshold, contamination)
    outliers = result[result['outliers']==-1]
    print('Number of outliers detected: ', len(outliers))
    print('Outliers:\n', outliers)

    # Keeping only the inliers
    cleaned_data = result[result['outliers']!= -1]
    plot_data(training_data, cleaned_data)
    return cleaned_data[['time_scale', 'sum_quant_item']]

cleaned1_data = apply_outlier_detection(z_score_threshold=2, contamination=0.01, training_data=training_data)
