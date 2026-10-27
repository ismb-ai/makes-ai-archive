# Anomaly_Detection_ARIMA_RNN.py
#
#     O programa oferecido integra técnicas de aprendizado profundo para proces-
# sar e depurar séries temporais, com o objetivo de identificar e tratar anoma-
# lias. Utilizando o Python como linguagem de programação, o código começa pela 
# importação de pacotes indispensáveis, que fornecem ferramentas para manipula-
# ção de dados, modelagem de autoencoders, normalização de dados, construção de 
# redes neurais recorrentes (RNN), e visualização gráfica.
#     A primeira fase do processo consiste em normalizar os dados através da 
# função `standardize_data`, que aplica uma transformação MinMaxScaler para re-
# escalar os dados, garantindo que os valores de entrada da rede neural variem 
# de 0 a 1. Essa padronização é fundamental para uma convergência mais eficiente 
# durante o treinamento do modelo de autoencoder, que é realizado na sequência 
# pela função `model_autoencoder`. Neste estágio, a rede neural dimensiona e mo-
# dela a informação para identificar padrões considerados normais no conjunto de 
# dados.
#     A função `mask_anomalies` é a responsável por aplicar o modelo de autoen-
# coder treinado, identificando desvios que superam um limiar pré-estabelecido. 
# Estes são marcados como anomalias e substituídos por valores nulos, isolando-
# os da série temporal para análises futuras. A limpeza dos dados prepara o ter-
# reno para a aplicação de uma rede neural recorrente LSTM, elaborada pela fun-
# ção `lstm_model`. A LSTM é especializada na detecção de padrões em sequências 
# temporais e é adequadamente estruturada para prever valores em séries tempo-
# rais depuradas de inconsistências.
#     Após a execução das funções de limpeza, os dados são visualizados por meio 
# de um gráfico, gerado pela função `plot_data`, que exibe a comparação entre os 
# dados originais e os dados já processados e limpos. Essa representação gráfica 
# permite avaliar a eficácia da limpeza de anomalias e confirma a melhoria na 
# homogeneidade da série temporal.
#     Por fim, a função `clean_data` arquiteta a orquestração de todo o procedi-
# mento de limpeza, iniciando pelo escalonamento dos dados, seguindo pela detec-
# ção de anomalias, e concluindo pela visualização comparativa dos dados. Os da-
# dos limpos, juntamente com o modelo LSTM treinado, são devolvidos, fornecendo 
# uma base confiável e aprimorada para futuras análises e predições sobre a sé-
# rie temporal em foco.

import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, LSTM, TimeDistributed
import matplotlib.pyplot as plt

# standardize_data(data): 
# Esta função escalona os dados para o intervalo [0, 1] utilizando o `MinMaxScaler`. A normalização é importante para preparar os dados para o treinamento de redes neurais. A função retorna os dados normalizados e o objeto de escala utilizado.

def standardize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# model_autoencoder(data): 
# Define e treina um modelo de Autoencoder, que é uma rede neural usada para redução de dimensionalidade e detecção de anomalias. Os `hidden_neurons` definem a estrutura das camadas internas do Autoencoder. O modelo é treinado com os dados fornecidos e é devolvido.

def model_autoencoder(data):
    clf = AutoEncoder(hidden_neurons =[8,4,2,4,8])
    clf.fit(data)
    return clf    

# mask_anomalies(data, clf): 
# Esta função utiliza o modelo de Autoencoder treinado (clf) para calcular os escores de decisão (nível de anomalia) para cada ponto de dado. Os dados com escores acima de um limiar definido são marcados como anomalias e substituídos por `np.nan` (Not a Number), indicando que são valores ausentes ou anômalos.

def mask_anomalies(data, clf):
    pred_scores = clf.decision_function(data)
    threshold = 1 
    anomalies = (pred_scores > threshold).reshape((-1,1))
    data = np.where(anomalies, np.nan, data) 
    return data

# lstm_model(data, look_back): 
# Constrói um modelo de LSTM, que é um tipo de RNN capaz de aprender dependências temporais em séries temporais. O LSTM utiliza uma sequência de `look_back` pontos para fazer previsões. O modelo inclui camadas `RepeatVector` e `TimeDistributed` que ajudam a estruturar a entrada e saída para problemas de sequência para sequência.

def lstm_model(data, look_back):
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(None, 1)))
    model.add(RepeatVector(data.shape[1]))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mae')
    return model 

# plot_data(original_data, cleaned_data): 
# Esta função plota os dados originais e os dados limpos para comparação visual. O gráfico gerado permite a avaliação da eficácia do processo de limpeza de dados.

def plot_data(original_data, cleaned_data):
    plt.figure(figsize=(15, 10))
    plt.title('Original vs Cleaned data')
    plt.plot(original_data['time_scale'], original_data['sum_quant_item'], '-o', label='Original data')
    cleaned_data = cleaned_data.dropna()
    plt.plot(cleaned_data['time_scale'], cleaned_data['sum_quant_item'], '-o', label='Cleaned data')
    plt.legend(loc='upper right')
    plt.show()

# No restante do código:
# - A função `clean_data` é definida para orquestrar o processo de limpeza da série temporal. Nota-se que o carregamento dos dados está comentado e deve ser ajustado ao caminho de arquivo apropriado.
# - Os dados são padronizados utilizando a função `standardize_data` e anomalias são mascaradas com a função `mask_anomalies`, utilizando o modelo de Autoencoder.
# - Os dados limpos são convertidos de volta para a estrutura do DataFrame do pandas e as linhas com valores `np.nan` são removidas.
# - Os dados originais e limpos são visualizados através da função `plot_data`.
# - Um modelo LSTM é criado para os dados limpos usando a função `lstm_model`, com um parâmetro `look_back` que define quantos passos no tempo o modelo deve considerar para fazer suas previsões.
# - A função `clean_data` retorna os dados limpos e o modelo LSTM treinado.

def clean_data():
    #training_data = pd.read_csv('your_data_path.csv') # update with your data path
    scaled_data, _ = standardize_data(training_data)
    clf= model_autoencoder(scaled_data)

    # Mask anomalies in training data
    cleaned_data = mask_anomalies(scaled_data, clf)

    # Convert cleaned data back to DataFrame structure
    cleaned_data = pd.DataFrame(data=cleaned_data, columns=training_data.columns).dropna() 

    # Plot original vs cleaned data
    plot_data(training_data, cleaned_data)

    # Fit LSTM model on cleaned data
    look_back = 3
    model = lstm_model(cleaned_data, look_back)

    return cleaned_data, model

cleaned_data, model = clean_data()
print(cleaned_data)
