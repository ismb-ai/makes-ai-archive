from sklearn.svm import SVR

def optimize_time_svm(training_data, window_size = 5, horizon = 1):
    # Separate independent and dependent variables
    X = training_data[['time_scale']]
    y = training_data['sum_quant_item']

    # We will try unit times from 1 to max_unit_time
    max_unit_time = max(training_data['time_scale'])

    best_unit_time = 1
    best_score = float('-inf')

    scores = []

    for unit_time in range(1, max_unit_time+1):
        # Generate model and cross validation method
        model = SVR(kernel='rbf') # Gaussian Kernel
        tscv = TimeSeriesSplit(n_splits=int(len(X)/(window_size+horizon))) 
    
        score = 0.0
        for train_index, test_index in tscv.split(X):
            # Adjust train index with respect to horizon
            train_index_adj = train_index[:len(train_index)-horizon]
            
            # Sample unit time
            X_train, X_test = X.iloc[train_index_adj].values.reshape(-1,1)[::unit_time], X.iloc[test_index].values.reshape(-1,1)
            y_train, y_test = y[train_index_adj].values.reshape(-1,1)[::unit_time], y[test_index]
            
            # Fit and score model
            model.fit(X_train, y_train.ravel())
            score += model.score(X_test, y_test)
        
        # Average score
        score /= tscv.get_n_splits()
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_unit_time = unit_time

    # Print the optimal unit of time
    print('Optimal unit of time:', best_unit_time)

    # Plot model performance versus unit of time
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(1, max_unit_time+1)), scores, marker='o', linestyle='-', color='b')
    plt.title("Model performance per unit of time")
    plt.xlabel("Unit time")
    plt.ylabel("Model performance")
    plt.show()

# Your data:
training_data = pd.DataFrame({'time_scale':np.arange(1,100),'sum_quant_item':np.random.rand(99)})

# Call the function:
optimize_time_svm(training_data)
