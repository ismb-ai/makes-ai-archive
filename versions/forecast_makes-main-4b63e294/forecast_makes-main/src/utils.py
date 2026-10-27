import pandas as pd
import math

def relativeReturns(X):
    '''
    X = [X_i] is a vector of length n with nonzero coordinates
    '''
    n = len(X)
    returns = [math.nan]
    for i in range(1,n):
        if X[i-1] != 0:
            value = (X[i]-X[i-1])/abs(X[i-1])
            returns.append(value)
        else:
            raise ValueError(f"Coordinate {i} is zero")
    return {"returns":returns,"initial_value":X[0]}

def backtoTotals(returns,initial_value):
    n = len(returns)
    totals = [initial_value]
    for i in range(1,n):
        temp_total = (returns[i]*abs(totals[i-1]) + totals[i-1])
        totals.append(temp_total)
    return totals

def toTakens(df, dim):
    '''
    df is a dataframe with columns = 'date', 'value'. toTakens creates a new dataframe 'Y' with #dim new columns corresponding to the lags and one extra column for the shift to the future ( answer )
    usable data corresponds to Y[dim:,:].
    Output: DataFrame with Y_pred at last row = Nan
    '''
    # Y_pred corresponds to the 'Relative change over the next period'
    values = df['value']
    X = {'date':df.index}
    i = 0
    while i <= dim-1:
      X.update({'Lag '+ str(i):values.shift(i)})
      i += 1
    X = pd.DataFrame(X)
    # remove first dim-1 rows that contains NA. Then move the answer column into a separate DataFrame.
    X = X.iloc[(dim-1):,:].reset_index(drop=True)
    return X

def load_data(FILEPATH,period='15d',EXCEL=False):
    '''
    Load a CSV or XLSX file and apply .sum() depending on period.
    This function is expecting a table with columns 'date', 'item' and 'value'
    It substitutes any zero to 1 after summing for 'period'.
    '''
    if EXCEL:
        df = pd.read_excel(FILEPATH)
        print("Loading xlsx ... OK!")
    else:
        df = pd.read_csv(FILEPATH)
        print("Loading csv ... OK!")
    print("Estimating column(s) representing 'date'...")
    df['date'] = pd.to_datetime(df['date'])
    df_items = {}
    for item in df.item.unique():
        print("Choosing item ", item)
        temp = df.loc[df.item == item,:] ; temp = temp.drop(columns=["item"])
        temp.set_index('date', inplace = True) ; temp = pd.DataFrame(temp.resample(period)['value'].sum())
        for i in range(len(temp)):
            if temp.value[i]==0:
                print("Found zero for item:",item," at i:",i)
                temp.value[i] = 1
        df_items[item] = temp
    return df_items

def to_vector(df_items,dim = 5,relative_returns = True):
    '''
    df_items is a dict with df_items.keys() = list of unique items, we apply Lag Embedding for each item.
    '''
    X = {}
    if relative_returns == True:
        initial_values = {}
        for item in df_items.keys():
            temp = df_items[item]
            returns = relativeReturns(temp.value)
            df_items[item].value=returns['returns']
            initial_values[item]=returns['initial_value']
            df_items[item] = df_items[item].iloc[1:,:]
            X[item] = toTakens(df_items[item],dim)
        return [X,initial_values]
    elif relative_returns == False:
        for item in df_items.keys():
            X[item] = toTakens(df_items[item],dim)
        return X

def preprocess(FILEPATH,period,EXCEL,dim,relative_returns):
    df_items = load_data(FILEPATH,period,EXCEL)
    if relative_returns == True:
        df_lag, initial_values = to_vector(df_items,dim,relative_returns)
        print("Using relative_returns = True... Saving variable initial_values.")
        return [df_lag,initial_values]
    else:
        df_lag = to_vector(df_items,dim,relative_returns)
        return df_lag

# Sliding Window

def windowAt(df, start, final):
    '''
    iloc inclui start e vai até final - 1, ou seja, retorna um DataFrame de length
    '''
    subdf = df.iloc[start:final].reset_index(drop=True)
    return subdf

def rollWindow(df,winsize,h=None):
    n = len(df)
    out = {}
    if h:
        for i in range(n-(winsize)):
            tmp = windowAt(df,i,i+(winsize))
            tmp_output = []
            for column in tmp.columns:
                tmp_output.append(h(tmp[column]))
            out[i] = tmp_output
        out = pd.DataFrame.from_dict(out,orient='index',columns=tmp.columns)
    else:
        for i in range(n-(winsize)):
            tmp = windowAt(df,i,i+(winsize))
            out[i] = tmp
    return out

#def trainable(g):
#    adapt_g = lambda x: [g(x.iloc[:-1]), g(x.iloc[:-1])(x.iloc[-1,:])]
#    return adapt_g

def makeTSDataset(data,horizon=1): #TODO PeRMITIR HORIZON>1!!!!
    '''
    Description: Create a dataset from a timeseries after toTakens.
    horizon :: int >= 0.
    If horizon=0 then we select the last term to predict (use if the objective is to predict RL).
    If horizon=1 then we predict the last term with existing corresponding answer (last_index)
    e.g. let data =  data[0] , ... , data[i] , ... , data[n-1] and horizon=1 => X_hor = data[n-2] (the last term n-1 has no answer, we always throw away)
    Output: X_train, Y_train, X_hor, Y_hor, X_last
    '''
    n = len(data)
    last_index = n-1
    n_hor = last_index - horizon
    #X = data.loc[:,data.columns != "date"] #TODO Throw or keep date? BACKLOG!!!
    X = data
    # .shift(-1) implies that the last term is NA
    Y = data.shift(-1) # shift(-1) implies that last term is NA!!
    X_train = X.iloc[:n_hor,:].reset_index(drop=True)
    Y_train = Y.iloc[:n_hor,:].reset_index(drop=True)
    if horizon > 0:
        Y_hor = Y.iloc[n_hor:-1,[0,1]].reset_index(drop=True)
        X_hor = X.iloc[n_hor:-1,:].reset_index(drop=True) # the part that we know the answer
    elif horizon == 0:
        print("RL mode (horizon=0).")
        X_hor = X.iloc[n_hor,:].reset_index(drop=True) # the term without answer
        Y_hor = Y.iloc[n_hor,:].reset_index(drop=True) # the NA term
    return [X_train, Y_train, X_hor, Y_hor]

def modelFit(model, X, Y):
    if (len(X) != len(Y)):
        print("X and Y must have same length")
    else:
        model.fit(X,Y)
    return model

def modelPredict(model, X_pred):
    Y_pred = model.predict(X_pred)
    return Y_pred

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a TimeSeries Dataset')
    parser.add_argument('-FILEPATH', type=str)
    parser.add_argument('-period', type=str)
    parser.add_argument('-EXCEL', type=bool,default=False)
    parser.add_argument('-dim', type=int)
    parser.add_argument('-relative_returns', type=bool)
    args = parser.parse_args()
    print(args)
    preprocess(args.FILEPATH,args.period,args.EXCEL,args.dim,args.relative_returns)