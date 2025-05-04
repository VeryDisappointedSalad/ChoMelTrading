import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bid_ask(X : pd.DataFrame):
    fig, ax1 = plt.subplots(figsize = (12,8))
    ax1.plot(X['Ask'], c = 'red', label = 'Ask')
    ax1.plot(X['Bid'], c = 'green', label = 'Bid')
    ax1.plot([], [], c = 'blue', label = 'Spread')
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(X['Ask'] - X['Bid'], color = 'blue', alpha = 0.2)
    plt.grid()

# Plot non_percentage capital curve with entries
def plot_capital_curve(capital_df, entry_times, exit_times, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(capital_df.index, capital_df['capital'], label='Capital Curve', color='blue')

    # Plot entry and exit points
    entries = capital_df.loc[capital_df.index.isin(entry_times)]
    exits = capital_df.loc[capital_df.index.isin(exit_times)]
    plt.scatter(entries.index, entries['capital'], color='green', label='Entry', marker='^', s=100, alpha = 0.5)
    plt.scatter(exits.index, exits['capital'], color='red', label='Exit', marker='v', s=100, alpha = 0.5)

    plt.title(f"Equity curve over time for {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# For 
def plot_all(data, history_df, title):
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