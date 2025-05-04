import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# signum(x - threshold)
def sign_func(x : float, threshold : float = 0.0) -> int:
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

# Having bid and ask signals
def signal_transform(results_ask : pd.DataFrame, results_bid : pd.DataFrame) -> pd.DataFrame:

    results_bid = results_bid.rename(columns = {'true_value' : 'targetBid', 'predicted_value' : 'predictedBid'})
    results_ask = results_ask.rename(columns = {'true_value' : 'targetAsk', 'predicted_value' : 'predictedAsk'})
    results = pd.concat([results_ask.drop(columns = ['timestamp']), results_bid], axis = 1)

    results['actualReturnAsk'] = results['targetAsk'].pct_change()
    results['actualReturnBid'] = results['targetBid'].pct_change()

    results['raw_signalAsk'] = results['actualReturnAsk'].apply(sign_func)
    results['raw_signalBid'] = results['actualReturnBid'].apply(sign_func)

    results['raw_signalAsk'] = results['raw_signalAsk'].replace(-1, 0)
    results['raw_signalBid'] = results['raw_signalBid'].replace(1, 0)

    results = results.dropna()
    results.index = pd.to_datetime(results['timestamp'])
    results.drop(columns = ['timestamp'])
    return results

# Most realistic backtest so far
def backtest_improved(
    data: pd.DataFrame,
    K: float,
    COSTS: float,
    S: float, # percent of entry price Stop Loss --> S=0.02 means 2% of entry price
    T: float, # Take Profit
    expected_return_threshold: float, # take trade only if predicted price is expected_return_threshold% above current
    mode: str = "fixed", # growing, fixed
    max_position_size: int = 10,
    trade_ratio: float = 0.5, # how much of the current capital can I use in one trade?
    trading_hours: tuple = (9, 16)
):
    data = data.copy()
    data['hour'] = data.index.hour
    in_trading_hours = (data['hour'] >= trading_hours[0]) & (data['hour'] < trading_hours[1])

    history = []
    starting_capital = K
    cash = K
    realized_pnl = 0.0
    position = 0.0
    entry_price = None
    total_trades = 0

    for idx, row in data.iterrows():
        is_trading_hour = in_trading_hours.loc[idx]

        predicted_ask = row['predictedAsk']
        predicted_bid = row['predictedBid']
        target_ask = row['targetAsk']
        target_bid = row['targetBid']
        signal_ask = row['raw_signalAsk']
        signal_bid = row['raw_signalBid']

        if position > 0:
            current_price = target_bid
            unrealized_pnl = (current_price - entry_price) * position
        elif position < 0:
            current_price = target_ask
            unrealized_pnl = (entry_price - current_price) * abs(position)
        else:
            current_price = None
            unrealized_pnl = 0.0

        total_value = cash + unrealized_pnl
        realized_equity = realized_pnl + cash

        if position != 0:
            gross_return_pct = unrealized_pnl / abs(position * entry_price)
            net_return_pct = gross_return_pct - (2 * COSTS / entry_price)

            if net_return_pct <= -S or net_return_pct >= T:
                cash += position * current_price
                cash -= abs(position) * COSTS
                realized_pnl += (position * (current_price - entry_price)) - abs(position) * COSTS
                total_trades += 1

                history.append({
                    'timestamp': idx,
                    'action': 'CLOSE_SL_TP',
                    'position': 0,
                    'price': current_price,
                    'size': abs(position),
                    'capital': cash,
                    'realized_pnl': realized_pnl,
                    'realized_equity': realized_equity,
                    'entry_price': entry_price,
                    'unrealized_pnl': 0.0
                })

                position = 0.0
                entry_price = None
                continue

        if not is_trading_hour:
            history.append({
                'timestamp': idx,
                'action': 'HOLD',
                'position': position,
                'price': current_price if position != 0 else None,
                'size': 0,
                'capital': cash,
                'realized_pnl': realized_pnl,
                'realized_equity': realized_equity,
                'entry_price': entry_price,
                'unrealized_pnl': unrealized_pnl
            })
            continue

        def open_position(direction, target_price, signal, predicted, condition):
            nonlocal cash, position, entry_price, total_trades, realized_pnl
            size = min(max_position_size, 1) if mode == 'growing' else max(0.01, trade_ratio * (cash / target_price))
            cost = size * target_price + size * COSTS

            if direction == "long" and cash >= cost:
                entry_price = target_price
                cash -= cost
                position = size
            elif direction == "short" and cash >= size * COSTS:
                entry_price = target_price
                cash += size * target_price
                cash -= size * COSTS
                position = -size
            else:
                return

            total_trades += 1
            history.append({
                'timestamp': idx,
                'action': f'OPEN_{direction.upper()}',
                'position': position,
                'price': entry_price,
                'size': size,
                'capital': cash,
                'realized_pnl': realized_pnl,
                'realized_equity': realized_equity,
                'entry_price': entry_price,
                'unrealized_pnl': 0.0
            })

        if position == 0:
            if signal_ask == 1 and ((predicted_ask - target_ask) / target_ask >= expected_return_threshold) and target_ask > target_bid:
                open_position("long", target_ask, signal_ask, predicted_ask, target_ask)
            elif signal_bid == -1 and ((target_bid - predicted_bid) / target_bid >= expected_return_threshold) and target_bid < target_ask:
                open_position("short", target_bid, signal_bid, predicted_bid, target_bid)

        elif mode == 'fixed':
            if position > 0 and signal_bid == -1 and ((target_bid - predicted_bid) / target_bid >= expected_return_threshold):
                cash += position * target_bid
                cash -= abs(position) * COSTS
                realized_pnl += (position * (target_bid - entry_price)) - abs(position) * COSTS
                total_trades += 1

                history.append({
                    'timestamp': idx,
                    'action': 'CLOSE_LONG_OPPOSITE',
                    'position': 0,
                    'price': target_bid,
                    'size': position,
                    'capital': cash,
                    'realized_pnl': realized_pnl,
                    'realized_equity': realized_pnl + cash,
                    'entry_price': entry_price,
                    'unrealized_pnl': 0.0
                })
                position = 0.0
                entry_price = None

            elif position < 0 and signal_ask == 1 and ((predicted_ask - target_ask) / target_ask >= expected_return_threshold):
                cash += position * target_ask
                cash -= abs(position) * COSTS
                realized_pnl += (-position * (entry_price - target_ask)) - abs(position) * COSTS
                total_trades += 1

                history.append({
                    'timestamp': idx,
                    'action': 'CLOSE_SHORT_OPPOSITE',
                    'position': 0,
                    'price': target_ask,
                    'size': abs(position),
                    'capital': cash,
                    'realized_pnl': realized_pnl,
                    'realized_equity': realized_pnl + cash,
                    'entry_price': entry_price,
                    'unrealized_pnl': 0.0
                })
                position = 0.0
                entry_price = None
        else:
            history.append({
                'timestamp': idx,
                'action': 'HOLD',
                'position': position,
                'price': current_price if position != 0 else None,
                'size': 0,
                'capital': cash,
                'realized_pnl': realized_pnl,
                'realized_equity': realized_pnl + cash,
                'entry_price': entry_price,
                'unrealized_pnl': unrealized_pnl
            })

    if position != 0:
        current_price = target_bid if position > 0 else target_ask
        cash += position * current_price
        cash -= abs(position) * COSTS
        realized_pnl += (position * (current_price - entry_price)) - abs(position) * COSTS
        total_trades += 1

        history.append({
            'timestamp': data.index[-1],
            'action': 'FORCE_CLOSE',
            'position': 0,
            'price': current_price,
            'size': abs(position),
            'capital': cash,
            'realized_pnl': realized_pnl,
            'realized_equity': realized_pnl + cash,
            'entry_price': entry_price,
            'unrealized_pnl': 0.0
        })

    history_df = pd.DataFrame(history)
    history_df.set_index('timestamp', inplace=True)

    fill_cols = ['capital', 'position', 'realized_pnl', 'realized_equity', 'entry_price', 'unrealized_pnl']
    history_df[fill_cols] = history_df[fill_cols].ffill()
    history_df['action'] = history_df['action'].fillna('HOLD')

    stats = {
        'initial_capital': starting_capital,
        'final_value': cash,
        'realized_pnl': realized_pnl,
        'total_return_pct': realized_pnl / starting_capital * 100,
        'total_trades': total_trades,
        'max_drawdown': (history_df['realized_equity'].max() - history_df['realized_equity'].min()) / history_df['realized_equity'].max() * 100
    }

    return history_df, stats


def run_backtest(predictions, backtest_params, bool_show_plot, bool_save_plot):

    def plot_all(data, history_df, title, bool_save_plot):
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
        ax_price, ax_equity, ax_position = axs

        # Plot price data
        ax_price.plot(data.index, data['targetAsk'], label='targetAsk', color='green', alpha=0.6)
        ax_price.set_ylabel("Price")
        ax_price.legend(loc='upper left')
        ax_price.set_title(title)

        # Compute total equity
        total_equity = history_df['realized_pnl'] + history_df['unrealized_pnl']

        # Plot and fill equity background: green above 0, red below
        ax_equity.fill_between(history_df.index, 0, history_df['realized_pnl'],
                            where=history_df['realized_pnl'] >= 0, facecolor='green', alpha=0.1, zorder=0)
        ax_equity.fill_between(history_df.index, 0, history_df['realized_pnl'],
                            where=history_df['realized_pnl'] < 0, facecolor='red', alpha=0.1, zorder=0)

        # Plot realized equity line
        ax_equity.plot(history_df.index, history_df['realized_pnl'], label='Realized Equity', color='blue', linewidth=1.5)

        # Overlay unrealized PnL fill
        unrealized_gain = total_equity >= history_df['realized_pnl']
        ax_equity.fill_between(history_df.index, history_df['realized_pnl'], total_equity,
                            where=unrealized_gain, facecolor='green', alpha=0.3, label='Unrealized Gain')
        ax_equity.fill_between(history_df.index, history_df['realized_pnl'], total_equity,
                            where=~unrealized_gain, facecolor='red', alpha=0.3, label='Unrealized Loss')

        ax_equity.set_ylabel("Equity ($)")
        ax_equity.legend(loc='upper left')
        ax_equity.grid(True)

        # Plot position with color fills: long (green), short (red), flat (gray)
        pos = history_df['position'].copy()
        ax_position.plot(history_df.index, pos, label='Position Size', color='black', linewidth=1)

        ax_position.fill_between(history_df.index, 0, pos,
                                where=pos > 0, color='green', alpha=0.3, label='Long')
        ax_position.fill_between(history_df.index, 0, pos,
                                where=pos < 0, color='red', alpha=0.3, label='Short')
        ax_position.fill_between(history_df.index, 0, pos,
                                where=pos == 0, color='gray', alpha=0.1, label='Flat')

        ax_position.set_ylabel("Position")
        ax_position.set_xlabel("Time")
        ax_position.grid(True)
        ax_position.legend(loc='upper left')

        # Annotate actions with styled markers
        trade_markers = {
            'OPEN_LONG':    ('^', 'lime'),
            'OPEN_SHORT':   ('v', 'red'),
            'CLOSE_SL_TP':  ('o', 'gold'),
            'CLOSE_LONG_OPPOSITE': ('s', 'red'),
            'CLOSE_SHORT_OPPOSITE': ('s', 'lime'),
            'FORCE_CLOSE':  ('s', 'black')
        }
                
        for action, (marker, color) in trade_markers.items():
            trades = history_df[history_df['action'] == action].copy()
            if not trades.empty:
                ax_price.scatter(trades.index, trades['price'], marker=marker, label=action, s=60, edgecolors=color, facecolors='none', linewidths=1.5)

        ax_price.legend(loc='upper right', fontsize='small')
        fig.tight_layout()
        plt.show()
        # Save the plot
        if bool_save_plot:
            fig.savefig(f'Results\Financial_output\Plots\{title}.jpg')

    equity_curves = {ticker : None for ticker in list(predictions.keys())}

    for ticker, dataframes in predictions.items():

        # Read predictions per ticker
        results_ask, results_bid = dataframes
        results = signal_transform(results_ask, results_bid)

        # Run the backtest
        history, _ = backtest_improved(
            data = results,
            K = backtest_params['Capital'],
            COSTS = backtest_params['Costs'],
            S = backtest_params['StopLoss'],
            T = backtest_params['TakeProfit'],
            expected_return_threshold = backtest_params['ExpectedReturnThreshold'],
            mode = backtest_params['Mode'],                 
            max_position_size = backtest_params['MaxPositionSize_for_growing'],        
            trade_ratio = backtest_params['RatioOfCapitalPerTrade_for_fixed']                
        )

        # Plot the results
        if bool_show_plot:
            plot_all(results, history, title = f'{ticker} Backtest Results', bool_save_plot = bool_save_plot)

        # Save the equity for Performance Metrics
        equity_curves[ticker] = history.copy()

        # Save the equities
        history.to_csv(f'Results\Financial_output\Equities\{ticker}.csv')


    return equity_curves