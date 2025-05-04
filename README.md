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

## Visualization
Equity curves with entry/exit markers
Position tracking
Strategy comparison plots
Price action with trading signals
