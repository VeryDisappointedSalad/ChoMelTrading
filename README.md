# File structure

.
├── Backtest_non_percentage.py      # Backtesting implementation of my interpretation on predicted prices from LSTM
├── Backtest.ipynb                  # Notebook for running backtests and performance analysis
├── DataLoadHelperFunctions.py      # Data loading and preprocessing utilities
├── LSTM.ipynb                      # Notebook for LSTM model training
├── PerformanceMetrics.py           # Performance metrics calculation
├── Plots.py                        # Visualization functions
└── PytorchHelp.py                  # PyTorch utilities and LSTM implementation

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
