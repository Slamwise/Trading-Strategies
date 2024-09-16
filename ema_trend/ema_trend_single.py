import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Step 1: Get the list of S&P 500 stocks
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df['Symbol'].tolist()

tickers = get_sp500_tickers()
tickers = random.sample(tickers, 100)

# Parameters
ticker = 'TQQQ'  # Change this to any ticker you want
start_date = '2020-01-01'
end_date = '2024-09-13'

# Strategy Parameters
starting_capital = 100000  # Starting capital in dollars
leverage = 3  # Leverage factor

# Fetch historical data
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Ensure there is enough data
if len(data) < 200:
    print(f"Not enough data for {ticker}")
else:
    # Calculate EMAs
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Initialize columns with correct data types
    data['Signal'] = 0  # Integer type
    data['Position'] = 0  # Integer type

    # Initialize financial columns as floats
    data['Capital'] = float(starting_capital)
    data['Holdings'] = 0.0
    data['Cash'] = float(starting_capital)
    data['Total'] = float(starting_capital)

    in_position = False
    buy_signals = []
    sell_signals = []
    num_shares = 0

    for i in range(1, len(data)):
        date = data.index[i]
        prev_date = data.index[i - 1]

        # Carry forward previous day's Cash and Holdings
        data.loc[date, 'Cash'] = data.loc[prev_date, 'Cash']
        data.loc[date, 'Holdings'] = data.loc[prev_date, 'Holdings']

        if not in_position:
            # Buy condition
            if (
                (data.loc[date, 'Close'] > data.loc[date, 'EMA_21'] > data.loc[date, 'EMA_50'] > data.loc[date, 'EMA_200']) and
                (data.loc[date, 'Close'] > data.loc[date, 'EMA_200']) and
                (data.loc[prev_date, 'Close'] <= data.loc[prev_date, 'EMA_21']) and
                (data.loc[date, 'Close'] >= data.loc[date, 'EMA_200'])
            ):
                data.loc[date, 'Signal'] = 1  # Buy
                in_position = True
                buy_signals.append((date, data.loc[date, 'Close']))

                # Calculate the amount to invest with leverage
                amount_to_invest = data.loc[date, 'Cash'] * leverage
                num_shares = amount_to_invest // data.loc[date, 'Close']
                total_investment = num_shares * data.loc[date, 'Close']

                # Update cash and holdings
                data.loc[date, 'Cash'] -= total_investment
                data.loc[date, 'Holdings'] += total_investment
        else:
            # Update holdings value based on current price
            data.loc[date, 'Holdings'] = num_shares * data.loc[date, 'Close']

            # Sell condition
            if (
                (data.loc[date, 'Close'] < data.loc[date, 'EMA_21']) or
                (data.loc[date, 'EMA_21'] < data.loc[date, 'EMA_50']) or
                (data.loc[date, 'Close'] < data.loc[date, 'EMA_200'])
            ):
                data.loc[date, 'Signal'] = -1  # Sell
                in_position = False
                sell_signals.append((date, data.loc[date, 'Close']))

                # Calculate proceeds from selling
                proceeds = num_shares * data.loc[date, 'Close']

                # Update cash and holdings
                data.loc[date, 'Cash'] += proceeds
                data.loc[date, 'Holdings'] = 0.0
                num_shares = 0

        # Update the position
        data.loc[date, 'Position'] = 1 if in_position else 0

        # Update total portfolio value
        data.loc[date, 'Total'] = data.loc[date, 'Cash'] + data.loc[date, 'Holdings']
        data.loc[date, 'Capital'] = data.loc[date, 'Total']

        # Optionally, print the total to check if it's updating
        # print(f"Date: {date}, Total: {data.loc[date, 'Total']}")

    # After the loop, you can check the final capital
    print(f"Final Total Capital: {data.loc[data.index[-1], 'Total']}")

    # Identify buy and sell points for plotting
    buy_dates = [point[0] for point in buy_signals]
    buy_prices = [point[1] for point in buy_signals]
    sell_dates = [point[0] for point in sell_signals]
    sell_prices = [point[1] for point in sell_signals]

    # Create an interactive plot
    fig = go.Figure()

    # Add Close Price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Close Price: %{y:.2f}<extra></extra>'
    ))

    # Add EMAs
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_21'],
        mode='lines',
        name='EMA 21',
        line=dict(color='green'),
        hovertemplate='Date: %{x}<br>EMA 21: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_50'],
        mode='lines',
        name='EMA 50',
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>EMA 50: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_200'],
        mode='lines',
        name='EMA 200',
        line=dict(color='purple'),
        hovertemplate='Date: %{x}<br>EMA 200: %{y:.2f}<extra></extra>'
    ))

    # Add Buy Signals
    if buy_signals:
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=12),
            hovertemplate='Date: %{x}<br>Buy Price: %{y:.2f}<extra></extra>'
        ))

    # Add Sell Signals
    if sell_signals:
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=12),
            hovertemplate='Date: %{x}<br>Sell Price: %{y:.2f}<extra></extra>'
        ))

    # Customize layout
    fig.update_layout(
        title=f'{ticker} Price Chart with Buy and Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show the interactive chart
    fig.show()

    # Calculate Trading Metrics

    # Create a DataFrame of trades
    trades = []
    num_trades = min(len(buy_signals), len(sell_signals))

    for i in range(num_trades):
        entry_date = buy_dates[i]
        entry_price = buy_prices[i]
        exit_date = sell_dates[i]
        exit_price = sell_prices[i]
        num_shares = (starting_capital * leverage) // entry_price
        profit_loss = (exit_price - entry_price) * num_shares
        return_pct = (exit_price / entry_price - 1) * leverage * 100
        trades.append({
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Exit Date': exit_date,
            'Exit Price': exit_price,
            'Shares': num_shares,
            'Profit/Loss': profit_loss,
            'Return (%)': return_pct
        })

    # Handle open positions
    if len(buy_signals) > len(sell_signals):
        entry_date = buy_dates[-1]
        entry_price = buy_prices[-1]
        exit_date = data.index[-1]
        exit_price = data['Close'].iloc[-1]
        num_shares = (starting_capital * leverage) // entry_price
        profit_loss = (exit_price - entry_price) * num_shares
        return_pct = (exit_price / entry_price - 1) * leverage * 100
        trades.append({
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Exit Date': exit_date,
            'Exit Price': exit_price,
            'Shares': num_shares,
            'Profit/Loss': profit_loss,
            'Return (%)': return_pct
        })

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['Profit/Loss'] > 0]
    num_winning_trades = len(winning_trades)
    num_losing_trades = total_trades - num_winning_trades
    win_rate = (num_winning_trades / total_trades) * 100
    total_profit = trades_df['Profit/Loss'].sum()
    avg_profit_per_trade = trades_df['Profit/Loss'].mean()

    # Calculate strategy returns
    data['Strategy_Return'] = data['Total'].pct_change()
    data['Strategy_Return'].fillna(0, inplace=True)
    data['Cumulative_Strategy_Return'] = data['Total'] / starting_capital

    # Calculate market returns
    data['Market_Return'] = data['Close'].pct_change()
    data['Market_Return'].fillna(0, inplace=True)
    data['Cumulative_Market_Return'] = (1 + data['Market_Return']).cumprod()

    # Calculate drawdown
    data['Cumulative_High'] = data['Total'].cummax()
    data['Drawdown'] = data['Total'] / data['Cumulative_High'] - 1
    max_drawdown = data['Drawdown'].min()

    # Calculate Sharpe Ratio
    average_daily_return = data['Strategy_Return'].mean()
    std_daily_return = data['Strategy_Return'].std()
    sharpe_ratio = (average_daily_return / std_daily_return) * np.sqrt(252)  # Annualized Sharpe Ratio

    # Strategy and Market Cumulative Returns
    strategy_cum_return = data['Cumulative_Strategy_Return'].iloc[-1] - 1
    market_cum_return = data['Cumulative_Market_Return'].iloc[-1] - 1

    # Display Metrics
    print(f"Total Trades: {total_trades}")
    print(f"Number of Winning Trades: {num_winning_trades}")
    print(f"Number of Losing Trades: {num_losing_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: ${total_profit:.2f}")
    print(f"Average Profit per Trade: ${avg_profit_per_trade:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Strategy Cumulative Return: {strategy_cum_return:.2%}")
    print(f"Market Cumulative Return: {market_cum_return:.2%}")

    # Plot cumulative returns
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Scatter(
        x=data.index,
        y=data['Cumulative_Strategy_Return'],
        mode='lines',
        name='Strategy Cumulative Return',
        line=dict(color='green'),
    ))
    fig_returns.add_trace(go.Scatter(
        x=data.index,
        y=data['Cumulative_Market_Return'],
        mode='lines',
        name='Market Cumulative Return',
        line=dict(color='blue'),
    ))
    fig_returns.update_layout(
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig_returns.show()
