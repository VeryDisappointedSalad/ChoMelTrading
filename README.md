## File structure
- Backtesting implementation of my interpretation on predicted prices from LSTM
  - ├── `Backtest_non_percentage.py`
- Notebook for running backtests and performance analysis
  - ├── `Backtest.ipynb `                
- Data loading and preprocessing utilities
  - ├── `DataLoadHelperFunctions.py`
- Notebook for LSTM model training
  - ├── `LSTM.ipynb`                     
- Performance metrics calculation
  - ├── `PerformanceMetrics.py`          
- Visualization functions
  - ├── `Plots.py`                      
- PyTorch utilities and LSTM implementation
  - └── `PytorchHelp.py`
                  
Some backtesting results on plots are included.

## Backtesting includes
- Configurable position sizing (fixed or growing)
- Stop loss and take profit levels
- Trading hour restrictions
- Transaction cost modeling

## Performance Metrics
- Annualized Return (ARC)
- Annualized Standard Deviation (aSD)
- Maximum Drawdown (MD)
- Maximum Loss Duration (MLD)
- Three variants of Information Ratios (IR1, IR2, IR3)

| Ticker |  ARC  |  aSD  |   MD  |  MLD  |  IR1  |  IR2  |  IR3  |
|------|-------|-------|-------|-------|-------|-------|-------|
| ODFL | 22.365 | 0.136 | 5.310 | 0.202 | 1.640 | 6.909 | 7.636 |
| PYPL |  1.677 | 0.051 | 7.623 | 0.425 | 0.331 | 0.073 | 0.003 |
| TSLA |  5.303 | 0.076 | 9.438 | 0.127 | 0.699 | 0.393 | 0.164 |


## Visualization
Equity curves with entry/exit markers
Position tracking
Strategy comparison plots
Price action with trading signals
