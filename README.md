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

```{r}
library(knitr)

data <- data.frame(
  name = c("ODFL", "PYPL", "TSLA"),
  ARC = round(c(22.36496577, 1.67660688, 5.303478861), 3),
  aSD = round(c(0.1363416026, 0.05060823516, 0.07585461196), 3),
  MD = round(c(5.310075596, 7.623162382, 9.437718202), 3),
  MLD = round(c(0.2023568623, 0.4245526326, 0.1269690116), 3),
  IR1 = round(c(1.640362541, 0.3312913154, 0.6991636664), 3),
  IR2 = round(c(6.908875666, 0.07286284495, 0.3928915492), 3),
  IR3 = round(c(7.635855094, 0.002877437042, 0.1641102817), 3)
)

kable(data, caption = "Example performance metrics")
```

## Visualization
Equity curves with entry/exit markers
Position tracking
Strategy comparison plots
Price action with trading signals
