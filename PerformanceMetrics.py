##### PERFORMANCE METRICS #####
import openpyxl
import numpy as np
import pandas as pd

def MaximumDrawdown(equity, text = ""):

    # Drawdown calculation
    equity['drawdowns'] = 1 - equity[text]/(equity[text].cummax())
    max_drawdown = equity['drawdowns'].max()
    start_index = equity['drawdowns'].idxmax()
    start_index = equity[text].loc[:start_index].idxmax()
    
    # Find the level of equity at the start of the drawdown
    start_equity_level = equity[text].loc[start_index]
    
    # Search for the index where equity returns to a level higher than or equal to the level at the start of the drawdown
    equity = equity.loc[start_index:]
    end_index = equity[text][equity[text] > start_equity_level].index.min()

    if str(start_index) == 'NaT':
        start_index = equity.index[0]
    if str(end_index) == 'NaT':
        end_index = equity.index[-1]


    return max_drawdown, start_index, end_index

def PerformanceMetrics(df, ticker):

    # Data
    equity = df.copy()
    fix = 'realized_pnl'
    equity[fix] = equity['capital'].iloc[0] + equity[fix]
    equity = equity[[fix]]
    equity['Daily_return'] = equity[fix].pct_change()
    equity = equity.dropna()

    # ARC
    ARC = (equity[fix].values[-1] / equity[fix].values[0])**(252/len(equity)) - 1
    

    # aSD
    aSD = (equity[f'Daily_return'].std()) * (np.sqrt(252))
    

    # Maximum Drawdown and Maximum Drawdown Duration
    MD, start_drawdown, end_drawdown = MaximumDrawdown(equity, fix)
    MLD = np.abs((end_drawdown - start_drawdown).days)/252.03

    # Information Ratio *
    IR1 = ARC/aSD

    # Information Ratio **
    IR2 = ARC**3/(aSD*ARC*MD)

    # Information Ratio ***
    IR3 = ARC**3/(aSD*MD*MLD)

    #print(f'ARC = {round(100*ARC, 3)}% aSD = {round(aSD, 5)} MD = {round(100*MD, 3)}% MLD = {round(MLD, 3)}yrs IR1 = {round(IR1, 3)} IR2 = {round(IR2, 3)} IR3 = {round(IR3, 3)}')

    metrics = {
        'name' : ticker,
        'ARC': 100 * ARC,
        'aSD': aSD,
        'MD': 100 * MD,
        'MLD': MLD,
        'IR1': IR1,
        'IR2': IR2,
        'IR3': IR3
    }
    
    return metrics

def PerformanceMetricsTable(equity_curves):

    performance_metrics_table = pd.concat(
    [pd.DataFrame(PerformanceMetrics(equity, ticker), index = list(range(1)))
     for ticker, equity in equity_curves.items()])

    performance_metrics_table.index = performance_metrics_table['name']
    performance_metrics_table = performance_metrics_table[['ARC', 'aSD', 'MD', 'MLD', 'IR1', 'IR2', 'IR3']]
    performance_metrics_table.to_excel('Results\Financial_output\Equities\PerformanceMetrics.xlsx')

    return performance_metrics_table.round(3)