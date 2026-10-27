# exponential_smoothing.py
# Este script utiliza suavização exponencial de Holt-Winters para ajustar a série temporal.
# Resíduos (diferenças absolutas) entre valores reais e previstos são calculados.
# Z-scores são usados para identificar outliers nos resíduos, ou seja, pontos com desvios significativos das previsões.
# Estabelecemos um limite de z-score de 2 vezes o desvio padrão (prática padrão: qualquer ponto além disso em um conjunto normalmente distribuído é considerado outlier).
# Outliers são removidos dos dados e o script plota os dados originais e limpos para comparação.
# Por fim, os dados limpos são salvos no DataFrame `cleaned1_data`.

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import zscore

# plot_data(orig_data, cleaned_data) g
# Gera um gráfico comparando a série temporal original e a série temporal limpa, sem outliers. 

def plot_data(orig_data, cleaned_data):
    plt.figure(figsize=(12,6))
    plt.plot(orig_data, color='blue', label='Original Data')
    plt.plot(cleaned_data, color='red', label='Cleaned Data')
    plt.title('Original Data vs Cleaned Data')
    plt.xlabel('Time Scale')
    plt.ylabel('Sum Quant Item')
    plt.legend()
    plt.show()

# main()
# - Lê os dados de um arquivo .csv (remova '#' e substitua 'your_file.csv' pelo nome).
# - Extrai variáveis independentes (X) e dependentes (y) da série temporal.
# - Inicializa um modelo de Holt-Winters com sazonalidade aditiva e período sazonal de 4.
# - Ajusta o modelo aos dados e faz previsão para o próximo ponto na série.
# - Calcula resíduos como diferença absoluta entre valores reais e previstos.
# - Usa `zscore(residuals)` para calcular escores z.
# - Identifica outliers com escores z absolutos maiores que 1,96 (valor comum para 95% de confiança).
# - Imprime número e índices dos outliers detectados.
# - Remove outliers dos dados de treinamento.
# - Usa `plot_data(y, cleaned1_data['sum_quant_item'])` para plotar a série temporal original e a limpa.
# - Retorna os dados limpos.
    
def main():
    # load your training data into "training_data"
    # training_data = pd.read_csv('your_file.csv')

    X = training_data['time_scale'].values
    y = training_data['sum_quant_item'].values

    # Initialize and fit the model
    model = ExponentialSmoothing(y, seasonal='add',seasonal_periods=4)
    model_fit = model.fit()

    # make prediction
    yhat = model_fit.predict(len(y), len(y))

    # calculate residuals
    residuals = np.abs(y - yhat)
    
    # identify outliers using zscore method
    z_scores = zscore(residuals)
    outliers = np.where(np.abs(z_scores) > 1.96)

    print("Number of Outliers : ", len(outliers[0]))
    print("Outliers : ", outliers[0])

    
    # new data frame with outliers removed
    cleaned1_data = training_data.drop(outliers[0])
    
    # Plotting original data vs cleaned data
    plot_data(y, cleaned1_data['sum_quant_item'])

    return cleaned1_data


if __name__ == "__main__":
    cleaned1_data = main()  
