"""
Overnight (close-to-open) backtest for MU and a basket of other high beta names.

The strategy is deliberately simple: every trading day, buy at the official
close and sell at the next official open. Nothing is held through the regular
session, so the strategy only ever earns the overnight gap.

For each name the nightly return is

    overnight_return = Open[t] / Close[t-1] - 1

and it is booked on the date of the open, i.e. the day the position is sold.
Prices are split and dividend adjusted, so a split does not show up as a fake
overnight gap and a dividend is credited to the holder the way it would be in a
real account.

Each night the same fixed notional is deployed (see CAPITAL below), so the P&L
curves are additive and directly comparable across names. Compounded returns
are reported in the metrics table as well.

Two things worth remembering before reading anything into the results: filling
at the official close and the official open needs market-on-close and
market-on-open orders, and roughly 250 round trips a year makes the strategy
very sensitive to costs. The cost sensitivity table at the bottom of the run
shows how quickly the edge decays.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf

# Parameters
TICKERS = ['MU', 'NVDA', 'AMD', 'TSLA', 'MRVL', 'ON', 'SMCI', 'COIN']
BENCHMARK = 'SPY'          # used for the realized beta column
START_DATE = '2015-01-01'
END_DATE = None            # None = up to the latest available bar
CAPITAL = 100_000          # notional deployed each night, per name
COST_BPS_PER_SIDE = 0.0    # commission + slippage per side, in basis points
CHART_DIR = 'overnight_charts'
TRADING_DAYS = 252

# Categorical series colors, used in this fixed order (never cycled).
SERIES_COLORS = [
    '#2a78d6',  # blue
    '#eb6834',  # orange
    '#1baf7a',  # aqua
    '#eda100',  # yellow
    '#e87ba4',  # magenta
    '#008300',  # green
    '#4a3aa7',  # violet
    '#e34948',  # red
]
INK = '#0b0b0b'         # primary text, also the portfolio line
INK_MUTED = '#52514e'   # axis and secondary text
SURFACE = '#fcfcfb'
GRID = '#e8e7e3'


def make_session():
    """Requests session for yfinance.

    yfinance defaults to curl_cffi with a browser TLS fingerprint, which some
    corporate proxies reset. A plain requests session goes through cleanly.
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0 Safari/537.36'
        )
    })
    return session


def download_prices(tickers, start_date, end_date):
    """Download adjusted daily bars and return {ticker: DataFrame}."""
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        group_by='ticker',
        threads=False,
        session=make_session(),
    )

    prices = {}
    for ticker in tickers:
        if raw.columns.nlevels == 2:
            if ticker not in raw.columns.get_level_values(0):
                print(f"No data returned for {ticker}, skipping")
                continue
            data = raw[ticker]
        else:
            data = raw

        data = data.dropna(subset=['Open', 'Close'])
        if len(data) < 2:
            print(f"Not enough data for {ticker}, skipping")
            continue
        prices[ticker] = data

    return prices


def session_returns(data):
    """Split one ticker's bars into overnight, intraday and buy and hold returns.

    All three are indexed by the same dates so they can be compared night for
    night. The first bar is dropped because it has no prior close.
    """
    overnight = data['Open'] / data['Close'].shift(1) - 1
    overnight = overnight.dropna()

    intraday = (data['Close'] / data['Open'] - 1).reindex(overnight.index)
    buy_hold = data['Close'].pct_change().reindex(overnight.index)

    return pd.DataFrame({
        'Overnight': overnight,
        'Intraday': intraday,
        'BuyHold': buy_hold,
    })


def apply_costs(returns, cost_bps_per_side):
    """Subtract a round trip of costs from every nightly return."""
    if not cost_bps_per_side:
        return returns
    return returns - 2 * cost_bps_per_side / 10_000


def cumulative_pnl(returns, capital=CAPITAL):
    """Cumulative dollar P&L from deploying a fixed notional every night."""
    return (returns.fillna(0) * capital).cumsum()


def compounded_equity(returns, capital=CAPITAL):
    """Equity curve if the whole book is reinvested every night."""
    return capital * (1 + returns.fillna(0)).cumprod()


def max_drawdown(equity):
    """Worst peak to trough decline of the compounded equity curve."""
    return (equity / equity.cummax() - 1).min()


def realized_beta(returns, benchmark_returns):
    """Beta of daily close-to-close returns against the benchmark."""
    if benchmark_returns is None:
        return np.nan

    joined = pd.concat([returns, benchmark_returns], axis=1, sort=False).dropna()
    if len(joined) < 2:
        return np.nan

    asset, market = joined.iloc[:, 0], joined.iloc[:, 1]
    market_var = market.var()
    if market_var == 0:
        return np.nan
    return asset.cov(market) / market_var


def compute_metrics(name, frame, benchmark_returns, cost_bps_per_side):
    """Summary statistics for one name's overnight strategy."""
    overnight = apply_costs(frame['Overnight'], cost_bps_per_side)
    equity = compounded_equity(overnight)
    pnl = cumulative_pnl(overnight)

    years = (frame.index[-1] - frame.index[0]).days / 365.25
    total_return = equity.iloc[-1] / CAPITAL - 1
    cagr = (equity.iloc[-1] / CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
    std = overnight.std()
    sharpe = overnight.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan

    return {
        'Ticker': name,
        'Start': frame.index[0].date(),
        'End': frame.index[-1].date(),
        'Nights': len(overnight),
        'Beta vs ' + BENCHMARK: realized_beta(frame['BuyHold'], benchmark_returns),
        'Total P&L': pnl.iloc[-1],
        'Compounded Return': total_return,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Max Drawdown': max_drawdown(equity),
        'Win Rate': (overnight > 0).mean(),
        'Avg bps/Night': overnight.mean() * 10_000,
        'Best Night': overnight.max(),
        'Worst Night': overnight.min(),
        'Intraday Return': (1 + frame['Intraday'].fillna(0)).prod() - 1,
        'Buy & Hold Return': (1 + frame['BuyHold'].fillna(0)).prod() - 1,
    }


def build_portfolio(returns_by_ticker):
    """Equal weight the available names each night, rebalanced daily.

    Names with a later listing date simply join the average once they have
    data, so the early part of the curve is the basket that actually existed.
    """
    matrix = pd.DataFrame({t: f['Overnight'] for t, f in returns_by_ticker.items()})
    intraday = pd.DataFrame({t: f['Intraday'] for t, f in returns_by_ticker.items()})
    buy_hold = pd.DataFrame({t: f['BuyHold'] for t, f in returns_by_ticker.items()})

    return pd.DataFrame({
        'Overnight': matrix.mean(axis=1, skipna=True),
        'Intraday': intraday.mean(axis=1, skipna=True),
        'BuyHold': buy_hold.mean(axis=1, skipna=True),
    }).dropna(how='all')


def style_layout(fig, title, subtitle, yaxis_title, x_range=None):
    """Shared chart chrome: recessive grid, one axis, legend below the plot.

    The legend sits under the x axis so it never crowds the subtitle, and the x
    range is pinned to the data so a right hand direct label does not stretch
    the axis into empty years.
    """
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:13px;color:{INK_MUTED}'>{subtitle}</span>",
            font=dict(color=INK, size=20),
            x=0.02,
            xanchor='left',
        ),
        xaxis_title=None,
        yaxis_title=yaxis_title,
        hovermode='x unified',
        template='plotly_white',
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_MUTED, size=13),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.08,
            xanchor='center',
            x=0.5,
            title=None,
        ),
        margin=dict(l=80, r=150, t=90, b=80),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks='outside', tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickprefix='$', tickformat=',.0f')
    fig.add_hline(y=0, line_width=1, line_color=GRID)
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    return fig


def create_pnl_chart(pnl_by_ticker, portfolio_pnl, cost_bps_per_side):
    """The headline chart: cumulative P&L over time, one line per name."""
    fig = go.Figure()

    for i, (ticker, pnl) in enumerate(pnl_by_ticker.items()):
        fig.add_trace(go.Scatter(
            x=pnl.index,
            y=pnl.values,
            mode='lines',
            name=ticker,
            line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
            hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
        ))

    fig.add_trace(go.Scatter(
        x=portfolio_pnl.index,
        y=portfolio_pnl.values,
        mode='lines',
        name='Equal weight basket',
        line=dict(color=INK, width=3),
        hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
    ))

    # Direct label on the emphasis series so it reads without the legend.
    fig.add_annotation(
        x=portfolio_pnl.index[-1],
        y=portfolio_pnl.iloc[-1],
        text='  Equal weight basket',
        showarrow=False,
        xanchor='left',
        font=dict(color=INK, size=12),
    )

    cost_note = (
        'gross of costs' if not cost_bps_per_side
        else f'net of {cost_bps_per_side:g} bps per side'
    )
    style_layout(
        fig,
        'Buy the close, sell the open',
        f'Cumulative P&amp;L on ${CAPITAL:,.0f} deployed every night, {cost_note}',
        'Cumulative P&L',
        x_range=(portfolio_pnl.index[0], portfolio_pnl.index[-1]),
    )
    return fig


def create_session_chart(portfolio, cost_bps_per_side):
    """Where the return actually comes from: overnight vs intraday vs holding."""
    fig = go.Figure()

    series = [
        ('Overnight (close to open)', apply_costs(portfolio['Overnight'], cost_bps_per_side)),
        ('Intraday (open to close)', portfolio['Intraday']),
        ('Buy and hold', portfolio['BuyHold']),
    ]

    for i, (label, returns) in enumerate(series):
        pnl = cumulative_pnl(returns)
        fig.add_trace(go.Scatter(
            x=pnl.index,
            y=pnl.values,
            mode='lines',
            name=label,
            line=dict(color=SERIES_COLORS[i], width=2),
            hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
        ))

    style_layout(
        fig,
        'Where the return shows up',
        f'Equal weight basket, cumulative P&amp;L on ${CAPITAL:,.0f} per session',
        'Cumulative P&L',
        x_range=(portfolio.index[0], portfolio.index[-1]),
    )
    return fig


def cost_sensitivity(portfolio_overnight):
    """How the basket holds up as costs rise, in bps per side."""
    rows = []
    for cost in [0.0, 1.0, 2.0, 5.0, 10.0]:
        net = apply_costs(portfolio_overnight, cost)
        equity = compounded_equity(net)
        years = (net.index[-1] - net.index[0]).days / 365.25
        rows.append({
            'Cost bps/side': cost,
            'Total P&L': cumulative_pnl(net).iloc[-1],
            'Compounded Return': equity.iloc[-1] / CAPITAL - 1,
            'CAGR': (equity.iloc[-1] / CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def format_metrics(results_df):
    """Percentages as percentages and dollars as dollars, for printing."""
    out = results_df.copy()
    for column in ['Compounded Return', 'CAGR', 'Max Drawdown', 'Win Rate',
                   'Best Night', 'Worst Night', 'Intraday Return', 'Buy & Hold Return']:
        out[column] = out[column].map(lambda v: f"{v:.2%}")
    out['Total P&L'] = out['Total P&L'].map(lambda v: f"${v:,.0f}")
    for column in ['Sharpe', 'Avg bps/Night', 'Beta vs ' + BENCHMARK]:
        out[column] = out[column].map(lambda v: f"{v:.2f}")
    return out


def show_figures(*figures):
    """Open the charts inline, but do not fail a headless run if there is no renderer."""
    for figure in figures:
        try:
            figure.show()
        except Exception as error:
            print(f"Could not display chart inline ({error}); open the saved HTML instead")
            return


def main():
    tickers = list(TICKERS)
    prices = download_prices(tickers + [BENCHMARK], START_DATE, END_DATE)

    benchmark_returns = None
    if BENCHMARK in prices:
        benchmark_returns = prices[BENCHMARK]['Close'].pct_change().rename(BENCHMARK)

    returns_by_ticker = {t: session_returns(prices[t]) for t in tickers if t in prices}
    if not returns_by_ticker:
        print('No usable price data, nothing to backtest')
        return

    results = [
        compute_metrics(ticker, frame, benchmark_returns, COST_BPS_PER_SIDE)
        for ticker, frame in returns_by_ticker.items()
    ]

    portfolio = build_portfolio(returns_by_ticker)
    results.append(
        compute_metrics('BASKET (EW)', portfolio, benchmark_returns, COST_BPS_PER_SIDE)
    )

    results_df = pd.DataFrame(results)

    print('\nBuy at the close, sell at the next open')
    print(f"Window: {START_DATE} to {END_DATE or 'latest'} | "
          f"${CAPITAL:,.0f} per name per night | "
          f"costs: {COST_BPS_PER_SIDE:g} bps per side\n")
    print(format_metrics(results_df).to_string(index=False))

    print('\nCost sensitivity, equal weight basket')
    sensitivity = cost_sensitivity(portfolio['Overnight'])
    print(sensitivity.assign(
        **{
            'Total P&L': sensitivity['Total P&L'].map(lambda v: f"${v:,.0f}"),
            'Compounded Return': sensitivity['Compounded Return'].map(lambda v: f"{v:.2%}"),
            'CAGR': sensitivity['CAGR'].map(lambda v: f"{v:.2%}"),
        }
    ).to_string(index=False))

    pnl_by_ticker = {
        ticker: cumulative_pnl(apply_costs(frame['Overnight'], COST_BPS_PER_SIDE))
        for ticker, frame in returns_by_ticker.items()
    }
    portfolio_pnl = cumulative_pnl(apply_costs(portfolio['Overnight'], COST_BPS_PER_SIDE))

    pnl_fig = create_pnl_chart(pnl_by_ticker, portfolio_pnl, COST_BPS_PER_SIDE)
    session_fig = create_session_chart(portfolio, COST_BPS_PER_SIDE)

    os.makedirs(CHART_DIR, exist_ok=True)
    pnl_path = os.path.join(CHART_DIR, 'overnight_pnl.html')
    session_path = os.path.join(CHART_DIR, 'overnight_vs_intraday.html')
    metrics_path = os.path.join(CHART_DIR, 'overnight_metrics.csv')

    pnl_fig.write_html(pnl_path)
    session_fig.write_html(session_path)
    results_df.to_csv(metrics_path, index=False)

    print(f"\nSaved {pnl_path}")
    print(f"Saved {session_path}")
    print(f"Saved {metrics_path}")

    show_figures(pnl_fig, session_fig)


if __name__ == '__main__':
    main()
