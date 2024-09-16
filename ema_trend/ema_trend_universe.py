import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import os
import random

def backtest_strategy(
    tickers,
    start_date='2020-01-01',
    end_date='2024-09-13',
    starting_capital=100000,
    leverage=3,
    rank_by='Total Profit',
    visualization=True
):
    results = []

    # Ensure output directories exist for charts
    strategy_chart_dir = 'strategy_charts'
    returns_chart_dir = 'returns_charts'
    if visualization:
        for directory in [strategy_chart_dir, returns_chart_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    for ticker in tickers:
        try:
            metrics, strategy_fig, returns_fig = backtest_single_ticker(
                ticker,
                start_date,
                end_date,
                starting_capital,
                leverage,
                visualization
            )
            if visualization and strategy_fig is not None and returns_fig is not None:
                # Save the strategy figure as an HTML file
                strategy_filename = f"{strategy_chart_dir}/{ticker}_strategy_chart.html"
                strategy_fig.write_html(strategy_filename)

                # Save the cumulative returns figure as an HTML file
                returns_filename = f"{returns_chart_dir}/{ticker}_returns_chart.html"
                returns_fig.write_html(returns_filename)

                # Create clickable links
                # metrics['Strategy Chart'] = f'<a href="{strategy_filename}" target="_blank">View Strategy Chart</a>'
                # metrics['Returns Chart'] = f'<a href="{returns_filename}" target="_blank">View Returns Chart</a>'
            # else:
            #     metrics['Strategy Chart'] = 'No Chart'
            #     metrics['Returns Chart'] = 'No Chart'
            if metrics is not None:  # Only append non-None results
                results.append(metrics)
            else:
                print(f"No metrics for {ticker}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Rank by the specified metric
    results_df.sort_values(by=rank_by, ascending=False, inplace=True)

    # Reset index
    results_df.reset_index(drop=True, inplace=True)

    # Display the DataFrame with clickable links if in Jupyter
    if visualization:
        pd.set_option('display.max_colwidth', None)
        print(results_df.to_string())
    else:
        print(results_df)

    return results_df

def backtest_single_ticker(
    ticker,
    start_date,
    end_date,
    starting_capital,
    leverage,
    visualization
):
    # Fetch historical data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Ensure there is enough data
    if len(data) < 200:
        print(f"Not enough data for {ticker}")
        return None, None, None

    # Calculate EMAs
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Initialize columns
    data['Signal'] = 0
    data['Position'] = 0
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

    # Handle open positions at the end
    if in_position:
        sell_signals.append((data.index[-1], data['Close'].iloc[-1]))
        proceeds = num_shares * data['Close'].iloc[-1]
        data.loc[data.index[-1], 'Cash'] += proceeds
        data.loc[data.index[-1], 'Holdings'] = 0.0
        data.loc[data.index[-1], 'Total'] = data.loc[data.index[-1], 'Cash']

    # Identify buy and sell points for plotting
    buy_dates = [point[0] for point in buy_signals]
    buy_prices = [point[1] for point in buy_signals]
    sell_dates = [point[0] for point in sell_signals]
    sell_prices = [point[1] for point in sell_signals]

    # Calculate Trading Metrics
    trades = []
    num_trades = min(len(buy_signals), len(sell_signals))

    for i in range(num_trades):
        entry_date = buy_dates[i]
        entry_price = buy_prices[i]
        exit_date = sell_dates[i]
        exit_price = sell_prices[i]
        shares = (starting_capital * leverage) // entry_price
        profit_loss = (exit_price - entry_price) * shares
        return_pct = (exit_price / entry_price - 1) * leverage * 100
        trades.append({
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Exit Date': exit_date,
            'Exit Price': exit_price,
            'Shares': shares,
            'Profit/Loss': profit_loss,
            'Return (%)': return_pct
        })

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print(f"No trades executed for {ticker}")
        return None, None, None

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
    if std_daily_return != 0:
        sharpe_ratio = (average_daily_return / std_daily_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Strategy and Market Cumulative Returns
    strategy_cum_return = data['Cumulative_Strategy_Return'].iloc[-1] - 1
    market_cum_return = data['Cumulative_Market_Return'].iloc[-1] - 1

    # Prepare metrics dictionary
    metrics = {
        'Ticker': ticker,
        'Total Trades': total_trades,
        'Winning Trades': num_winning_trades,
        'Losing Trades': num_losing_trades,
        'Win Rate (%)': win_rate,
        'Total Profit': total_profit,
        'Avg Profit per Trade': avg_profit_per_trade,
        'Max Drawdown (%)': max_drawdown * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Strategy Return (%)': strategy_cum_return * 100,
        'Market Return (%)': market_cum_return * 100
    }

    # Visualization
    if visualization:
        strategy_fig = create_strategy_chart(data, buy_signals, sell_signals, ticker)
        returns_fig = create_returns_chart(data, ticker)
    else:
        strategy_fig = None
        returns_fig = None

    return metrics, strategy_fig, returns_fig

def create_strategy_chart(data, buy_signals, sell_signals, ticker):
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

    return fig

def create_returns_chart(data, ticker):
    # Create cumulative returns chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Cumulative_Strategy_Return'],
        mode='lines',
        name='Strategy Cumulative Return',
        line=dict(color='green'),
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Cumulative_Market_Return'],
        mode='lines',
        name='Market Cumulative Return',
        line=dict(color='blue'),
    ))
    fig.update_layout(
        title=f'{ticker} Cumulative Returns Comparison',
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
    return fig



# Step 1: Get the list of S&P 500 stocks
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df['Symbol'].tolist()

tickers = get_sp500_tickers()
# tickers = random.sample(tickers, 100)

backtest_strategy(tickers, start_date='2023-01-01', visualization=True)
