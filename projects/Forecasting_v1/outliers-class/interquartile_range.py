# interquartile_range.py
# IQR (Interquartile Range):
# Inicialmente, este script treina um modelo de Regressão Linear nos dados.
# Em seguida, ele faz previsões para a variável alvo e calcula os resíduos (atual - previsto).
# A função `calculate_outliers_iqr` é utilizada para identificar outliers nos resíduos
# usando o método do Intervalo Interquartil (IQR): qualquer ponto de dados menor que Q1 - 1,5 * IQR
# ou maior que Q3 + 1,5 * IQR é considerado um outlier.
# Os outliers são removidos do conjunto de dados, e os dados originais e limpos são plotados para comparação.
# Por fim, o DataFrame `cleaned1_data` é retornado.


from sklearn.linear_model import LinearRegression

# calculate_outliers_iqr(y) 
# Calcula o Interquartile Range (IQR) dos dados.
# Calcula os percentis 25 (Q1) e 75 (Q3), então o IQR como a diferença entre Q3 e Q1.
# O limite inferior é Q1 - 1,5 vezes o IQR, e o limite superior é Q3 + 1,5 vezes o IQR.
# Valores fora deste intervalo são outliers.

def calculate_outliers_iqr(y):
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return np.where((y < lower_bound) | (y > upper_bound))

# plot_data(orig_data, cleaned_data) 
# Gera um gráfico da série temporal original e da série limpa (sem outliers). 
# O eixo y representa as séries temporais, e o eixo x representa a escala de tempo.

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
# - Extrai variáveis independentes (X) e dependentes (y).
# - Treina um modelo de regressão linear.
# - Calcula resíduos como a diferença entre valores reais e previstos.
# - Usa `calculate_outliers_iqr(residuals)` para detectar outliers.
# - Imprime número e índices dos outliers.
# - Remove outliers dos dados de treinamento.
# - Plota dados, comparando série original com série limpa.
# - Retorna dados limpos.    
    
def main():
    # load your training data into "training_data"
    # training_data = pd.read_csv('your_file.csv')

    X = training_data.iloc[:, 0:1].values
    y = training_data.iloc[:, 1].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    residuals = np.abs(y - y_pred)
    
    outliers = calculate_outliers_iqr(residuals)

    print("Number of Outliers : ", len(outliers[0]))
    print("Outliers : ", outliers[0])

    cleaned1_data = training_data.drop(outliers[0])
    
    plot_data(y, cleaned1_data['sum_quant_item'])

    return cleaned1_data


if __name__ == "__main__":
    cleaned1_data = main()
