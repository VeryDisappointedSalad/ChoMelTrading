import os 
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path

# Global variables

# for reading data
READING_PATH = 'Data/KibotData_New/'

# for saving data
SAVING_PATH = 'Results/ML_output'


# get available tickers from a path
def get_ticker_names(path : str = READING_PATH) -> List[str]:

    list_of_files = os.listdir(path)

    # choose only the tickers with .csv extension
    list_of_files = [elem for elem in list_of_files if elem.split('.')[-1] == 'csv']

    # get names of tickers saved in the format 'AMD_1min_03-08-2021_to_03-08-2022.csv'
    list_of_files = [elem.split('_')[0] for elem in list_of_files]
    output = list(set(list_of_files))
    print(f'Available tickers are {output} ( :')

    return output

# returns a dataframe merged from dataframes containing the ticker
def load_ticker(ticker : str, path : str = READING_PATH) -> pd.DataFrame:
    
    list_of_paths = [
        elem for elem in os.listdir(path) if elem.split('_')[0] == ticker
    ]

    # load the .csvs
    list_of_dataframes = [pd.read_csv(path + f'/{p}',
                                      names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
                                      for p in list_of_paths]
    
    df = pd.concat(list_of_dataframes)

    # make the index a DateTime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M')
    df.drop(['Date', 'Time'], axis = 1, inplace = True)
    
    df.set_index('Datetime', inplace = True)
    df.sort_index()

    #print(f'Loaded {ticker}!')
    return df

# Returns ticker ready for training on TARGET column
def feature_engineering(ticker : str, spread_range = (0.01, 0.05), volatility_factor = 0.0001, target : str = 'targetAsk', path = READING_PATH) -> List[pd.DataFrame]:

    # Read given ticker - only close and volume columns
    X = load_ticker(ticker, path)
    #print(f'Loading {ticker} with target = {target}')
    X = X[['Close', 'Volume']]

    # Generate bid and ask
    X['Bid'] = X['Close'] - np.random.uniform(*spread_range, size = len(X))
    X['Ask'] = X['Close'] + np.random.uniform(*spread_range, size = len(X))
    X['Bid'] += np.random.normal(0, volatility_factor, len(X))
    X['Ask'] += np.random.normal(0, volatility_factor, len(X))

    X['Bid'] = X[['Bid', 'Ask']].min(axis=1)
    X['Ask'] = X[['Bid', 'Ask']].max(axis=1)

    # Resample hourly
    X = X[['Bid', 'Ask', 'Volume']]
    X = X.resample('1h').mean()

    # Target variables
    X.loc[:, 'targetBid'] = X['Bid'].shift(-1)
    X.loc[:, 'targetAsk'] = X['Ask'].shift(-1)
    X = X.dropna()

    return X.drop(columns = [target]), X[[target]].rename(columns = {target : 'target'}) # X for model, y for model

# returns X_ask, y_ask, X_bid, y_bid
def get_bid_ask(ticker : str, spread_range = (0.01, 0.05), volatility_factor = 0.0001, path = READING_PATH) -> List[pd.DataFrame]:

    X_ask, y_ask = feature_engineering(ticker, spread_range, volatility_factor, target = 'targetAsk', path = path)
    X_bid, y_bid = feature_engineering(ticker, spread_range, volatility_factor, target = 'targetBid', path = path)

    return X_ask, y_ask, X_bid, y_bid

# save result y_ask, y_bid
def save_results(results_ask : pd.DataFrame, results_bid : pd.DataFrame, ticker : str, walk_forward_params : dict) -> None:

    # save csv
    list_of_params = list(walk_forward_params.values())
    name_of_file = f'AllBlocks {list_of_params[0]} Train {list_of_params[1]} Val {list_of_params[2]} Test {list_of_params[3]}'

    path_ask, path_bid = f'{SAVING_PATH}/{ticker}/Ask {name_of_file}.csv', f'{SAVING_PATH}/{ticker}/Bid {name_of_file}.csv'
    path_ask, path_bid = Path(path_ask), Path(path_bid)

    path_ask.parent.mkdir(parents=True, exist_ok=True)
    path_bid.parent.mkdir(parents=True, exist_ok=True)

    results_ask.to_csv(path_ask)
    results_bid.to_csv(path_bid)

# returns a list of predicted prices for all tickers under path
def read_results(path : str = SAVING_PATH) -> dict[List[pd.DataFrame]]:

    list_of_backtested_tickers = os.listdir(path)
    predictions = {f'{tick}' : [pd.read_csv(path + f'/{tick}/{elem}') for elem in os.listdir(path + f'/{tick}')] for tick in list_of_backtested_tickers}
    print(f'Returned predictions for {list(predictions.keys())}')
    return predictions

