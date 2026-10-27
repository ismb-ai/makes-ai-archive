import pandas as pd
from utils import *
import xgboost as xgb
import datetime
import matplotlib.pyplot as plt


models = {'xgboost':xgb.XGBRegressor(n_estimators=1000)}

def plot_predictions(predicted_dict):
    """
    Plota as previsões e os valores reais para cada estágio da janela deslizante.

    Args:
    - predicted_dict: dicionário com previsões e valores reais.
    """
    for item, df in predicted_dict.items():
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Predicted"], label="Predicted", color="blue")
        plt.plot(df["Date"], df["Real"], label="Real", color="red")
        plt.fill_between(df["Date"], df["Predicted"], df["Real"], color="grey", alpha=0.2)
        plt.title(f"Predictions vs Real for item: {item}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

def main(model_from_dict,horizon):
    W_items = {}
    allmodels = {}
    p = 15 #TODO BACKLOG: period = '15d' but p is int. Same parameter, different types (FIX THIS)
    modelpredicted = pd.DataFrame({"Date":[],"Predicted":[],"Real":[],"Difference":[]})
    predicted = {}
    for item in df_items_vec.keys():
        W_items[item] = rollWindow(df_items_vec[item],winsize=24,h=None)
        for i in range(len(W_items[item])):
            date = W_items[item][i].date
            X_train, Y_train, X_hor, Y_hor = makeTSDataset(W_items[item][i], horizon)
            model = models[model_from_dict]
            model = modelFit(model,X_train.loc[:,X_train.columns != "date"],Y_train.loc[:,"Lag 0"])
            allmodels[i] = model
            singleinput = X_hor.iloc[0,1:].astype(float).to_frame().transpose() #TODO Backlog generalize horizon to > 1
            pred = modelPredict(model, singleinput)[0] # [0] since we have array of one scalar
            # recover date for pred (decidi gerar outra data ja que se horizon=0 nao temos a proxima data)
            last_date = X_hor.date[len(X_hor.date)-1] ; next_date = last_date + datetime.timedelta(days=p)
            real_value = Y_hor.iloc[0,1]
            temp = pd.DataFrame({"Date":[next_date],"Predicted":[pred],"Real":[real_value],"Difference":[abs(pred - real_value)]})
            modelpredicted = pd.concat([modelpredicted,temp])
        predicted[item] = modelpredicted
    return predicted

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a TimeSeries Dataset')
    parser.add_argument('-FILEPATH', type=str, default = 'test.csv')
    parser.add_argument('-period', type=str, default = '15d')
    parser.add_argument('-EXCEL', type=bool,default=False)
    parser.add_argument('-dim', type=int, default = 5)
    parser.add_argument('-relative_returns', type=bool, default = True)
    parser.add_argument('-horizon', type=int, default = 1)
    parser.add_argument('-model_from_dict', type=str, default = 'xgboost')
    args = parser.parse_args()
    print(args)
    if args.relative_returns:
        df_items_vec, initial_values = preprocess(args.FILEPATH,args.period,args.EXCEL,args.dim,args.relative_returns)
    elif args.relative_returns == False:
        df_items_vec = preprocess(args.FILEPATH,args.period,args.EXCEL,args.dim,args.relative_returns)
    predicted = main(args.model_from_dict,args.horizon)
    plot_predictions(predicted) 
