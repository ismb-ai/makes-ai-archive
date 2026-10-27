# ARIMA_sliding_window.py
#
# Implementamos uma abordagem de janelas-deslizantes para modelar uma série- 
# temporal pelo método ARIMA. O código inicializa duas listas: "history",
# que armazena os primeiros elementos da série temporal como determinado pelo 
# tamanho da janela, e "predictions", que armazena as previsões futuras. 
# Em seguida, iteramos sobre o restante da série temporal, construindo e
# ajustando um modelo ARIMA usando a lista "history", e fazemos uma previsão 
# n-passos adiante, onde n:='horizonte'. Por fim, adicionamos os dados reais
# após todas as previsões terem sido feitas, calculamos o erro quadrático médio 
# entre os dados reais e as previsões, imprimimos o resultado, e então confron-
# tamos os dados reais e as previsões por meio de um gráfico.


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 
# Função que implementa a abordagem de janela deslizante com ARIMA

def run_arima_with_sliding_window(data, window_size, horizon):
    
    # Primeiramente, a função cria uma lista chamada "history" que contém o número
    # de elementos definidos pelo tamanho da janela. Esses elementos são retirados do início dos dados da série temporal. A função
    # também inicializa uma lista vazia chamada "predictions", que servirá para armazenar as previsões do
    # modelo ARIMA.
    history = [x for x in data['sum_quant_item'][:window_size]]
    predictions = list()

    # Laço que percorre a série temporal a partir do tamanho da janela.
    for t in range(window_size, len(data)):
        # Cria um modelo ARIMA usando os elementos da "history" como dados de treinamento e especificando a
        # ordem do modelo ARIMA como (1, 1, 0). A ordem do modelo ARIMA define o número de lag-observations no
        # modelo (AR), o número de vezes que os valores brutos são diferenciados (I), e o tamanho de uma janela
        # de média móvel (MA).
        model = ARIMA(history, order=(1, 1, 0))
        # Ajusta o modelo aos dados históricos
        model_fit = model.fit()

        # Faz previsões para o horizonte especificado
        output = model_fit.forecast(steps=horizon)
        yhat = output[0]
        # Adiciona a previsão à lista de previsões
        predictions.append(yhat)

        # Obtém o valor real no tempo 't'
        obs = data['sum_quant_item'][t]
        # Adiciona o valor real ao histórico
        history.append(obs)
        
    # Uma vez que todas as previsões foram feitas, a função calcula o erro quadrado médio entre a porção de
    # teste dos dados reais e as previsões. Ela imprime o RMSE (Root Mean Squared Error) e depois plota
    # tanto os dados de teste reais quanto as previsões.

    # Calcula o erro médio quadrático (RMSE) para o período de teste
    mse = mean_squared_error(data['sum_quant_item'][window_size:], predictions, squared=False)
    print('Test RMSE: %.3f' % mse)

    # Gera um gráfico para comparar os valores reais com as previsões
    plt.plot(data['sum_quant_item'][window_size:].values, label='Test')
    plt.plot(predictions, color='red', label='Predicted')
    plt.legend()
    plt.show()

# Por último, a função é chamada passando a série temporal de treinamento, o tamanho da janela como 50 e o
# horizonte como 10.
run_arima_with_sliding_window(training_data, 50, 10)
