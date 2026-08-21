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

Two presentation choices matter more than anything else here, because the
overnight effect is a compounding story:

  * The window runs over each name's full available history by default. MU has
    bars back to 1984, and starting in, say, 2015 throws away thirty years of
    compounding and badly understates the effect.
  * The headline chart compounds. Overnight returns carry roughly half the
    volatility of the full close-to-close move at a similar average return, so
    the gap between them only opens up once returns are reinvested. A fixed
    notional chart is also produced, but by construction it makes buy and hold
    look best whenever the intraday leg is positive, because the two legs
    simply add up.

Two things worth remembering before reading anything into the results: filling
at the official close and the official open needs market-on-close and
market-on-open orders, and roughly 250 round trips a year makes the strategy
very sensitive to costs. The run prints the cost per side at which the
overnight strategy stops beating buy and hold, which is the number that decides
whether any of this survives contact with a broker.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf

# Parameters
TICKERS = ['MU', 'NVDA', 'AMD', 'TSLA', 'MRVL', 'ON', 'SMCI', 'COIN']
FOCUS_TICKER = 'MU'        # the name broken out in the session decomposition chart
BENCHMARK = 'SPY'          # used for the realized beta column
START_DATE = None          # None = each name's full available history
END_DATE = None            # None = up to the latest available bar
CAPITAL = 100_000          # stake per name: compounded, and deployed per night
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
INK = '#0b0b0b'         # primary text, also the basket line
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
    request = dict(
        progress=False,
        auto_adjust=True,
        group_by='ticker',
        threads=False,
        session=make_session(),
    )
    if start_date is None and end_date is None:
        request['period'] = 'max'
    else:
        request['start'] = start_date
        request['end'] = end_date

    raw = yf.download(tickers, **request)

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


def compounded_equity(returns, capital=CAPITAL):
    """Equity curve if the whole stake is reinvested every night."""
    return capital * (1 + returns.fillna(0)).cumprod()


def cumulative_pnl(returns, capital=CAPITAL):
    """Cumulative dollar P&L from deploying a fixed notional every night."""
    return (returns.fillna(0) * capital).cumsum()


def total_return(returns):
    return (1 + returns.fillna(0)).prod() - 1


def annualized(returns):
    """CAGR of the compounded curve, from the calendar span of the returns."""
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (1 + total_return(returns)) ** (1 / years) - 1


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


def breakeven_cost_vs_buy_hold(overnight, buy_hold):
    """Highest cost per side at which overnight still beats buy and hold.

    Searched on a 0.1 bp grid. Returns 0.0 if the strategy never wins and the
    top of the grid if costs never bite hard enough to matter.
    """
    target = total_return(buy_hold)
    best = 0.0
    for cost in np.arange(0.0, 20.01, 0.1):
        if total_return(apply_costs(overnight, cost)) > target:
            best = cost
        else:
            break
    return best


def compute_metrics(name, frame, benchmark_returns, cost_bps_per_side):
    """Summary statistics for one name's overnight strategy."""
    overnight = apply_costs(frame['Overnight'], cost_bps_per_side)
    equity = compounded_equity(overnight)
    std = overnight.std()

    return {
        'Ticker': name,
        'Start': frame.index[0].date(),
        'Nights': len(overnight),
        'Beta vs ' + BENCHMARK: realized_beta(frame['BuyHold'], benchmark_returns),
        'Overnight Return': total_return(overnight),
        'Overnight CAGR': annualized(overnight),
        'Overnight Vol': std * np.sqrt(TRADING_DAYS),
        'Intraday Return': total_return(frame['Intraday']),
        'Buy & Hold Return': total_return(frame['BuyHold']),
        'Buy & Hold CAGR': annualized(frame['BuyHold']),
        'Buy & Hold Vol': frame['BuyHold'].std() * np.sqrt(TRADING_DAYS),
        'Sharpe': overnight.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        'Max Drawdown': max_drawdown(equity),
        'Win Rate': (overnight > 0).mean(),
        'Avg bps/Night': overnight.mean() * 10_000,
        'Breakeven bps/side': breakeven_cost_vs_buy_hold(frame['Overnight'], frame['BuyHold']),
        'Fixed-Notional P&L': cumulative_pnl(overnight).iloc[-1],
    }


def build_basket(returns_by_ticker):
    """Equal weight the available names each night, rebalanced daily.

    Names with a later listing date join the average once they have data, so
    the early part of the curve is the basket that actually existed. Over full
    history that means the basket starts as one or two names and widens.
    """
    frames = {
        column: pd.DataFrame({t: f[column] for t, f in returns_by_ticker.items()})
        for column in ['Overnight', 'Intraday', 'BuyHold']
    }
    return pd.DataFrame({
        column: frame.mean(axis=1, skipna=True) for column, frame in frames.items()
    }).dropna(how='all')


def style_layout(fig, title, subtitle, yaxis_title, x_range=None, log=False):
    """Shared chart chrome: recessive grid, one axis, legend below the plot."""
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
        margin=dict(l=90, r=150, t=90, b=80),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks='outside', tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID,
                     tickprefix='$', tickformat=',.0f')
    if log:
        # Equity spans several orders of magnitude over a multi-decade run, so a
        # linear axis would flatten everything before the last few years.
        fig.update_yaxes(type='log', dtick=1)
    else:
        fig.add_hline(y=0, line_width=1, line_color=GRID)
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    return fig


def add_end_label(fig, series, text, color):
    """Direct label at the right end of a line, drawn into the right margin."""
    fig.add_annotation(
        x=series.index[-1],
        y=np.log10(series.iloc[-1]) if fig.layout.yaxis.type == 'log' else series.iloc[-1],
        text=f"  {text}",
        showarrow=False,
        xanchor='left',
        font=dict(color=color, size=12),
    )


def create_growth_chart(returns_by_ticker, basket, cost_bps_per_side):
    """The headline chart: compounded growth of the stake, log scale."""
    fig = go.Figure()

    for i, (ticker, frame) in enumerate(returns_by_ticker.items()):
        equity = compounded_equity(apply_costs(frame['Overnight'], cost_bps_per_side))
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name=ticker,
            line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
            hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
        ))

    basket_equity = compounded_equity(apply_costs(basket['Overnight'], cost_bps_per_side))
    fig.add_trace(go.Scatter(
        x=basket_equity.index,
        y=basket_equity.values,
        mode='lines',
        name='Equal weight basket',
        line=dict(color=INK, width=3),
        hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
    ))

    cost_note = (
        'gross of costs' if not cost_bps_per_side
        else f'net of {cost_bps_per_side:g} bps per side'
    )
    style_layout(
        fig,
        'Buy the close, sell the open',
        f'Growth of ${CAPITAL:,.0f} reinvested every night, log scale, {cost_note}',
        f'Value of ${CAPITAL:,.0f} stake',
        x_range=(basket_equity.index[0], basket_equity.index[-1]),
        log=True,
    )
    add_end_label(fig, basket_equity, 'Equal weight basket', INK)
    return fig


def create_pnl_chart(returns_by_ticker, basket, cost_bps_per_side):
    """Cumulative P&L from a fixed stake each night, without reinvestment.

    This answers 'what did an average night pay' rather than 'what would the
    account be worth', so the vertical scale stays in plain dollars.
    """
    fig = go.Figure()

    for i, (ticker, frame) in enumerate(returns_by_ticker.items()):
        pnl = cumulative_pnl(apply_costs(frame['Overnight'], cost_bps_per_side))
        fig.add_trace(go.Scatter(
            x=pnl.index,
            y=pnl.values,
            mode='lines',
            name=ticker,
            line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
            hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
        ))

    basket_pnl = cumulative_pnl(apply_costs(basket['Overnight'], cost_bps_per_side))
    fig.add_trace(go.Scatter(
        x=basket_pnl.index,
        y=basket_pnl.values,
        mode='lines',
        name='Equal weight basket',
        line=dict(color=INK, width=3),
        hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
    ))

    style_layout(
        fig,
        'Buy the close, sell the open, no reinvestment',
        f'Cumulative P&amp;L on a flat ${CAPITAL:,.0f} deployed every night',
        'Cumulative P&L',
        x_range=(basket_pnl.index[0], basket_pnl.index[-1]),
    )
    add_end_label(fig, basket_pnl, 'Equal weight basket', INK)
    return fig


def create_session_chart(frame, label, cost_bps_per_side):
    """Where the return actually comes from: overnight vs intraday vs holding."""
    fig = go.Figure()

    series = [
        ('Overnight (close to open)', apply_costs(frame['Overnight'], cost_bps_per_side)),
        ('Intraday (open to close)', frame['Intraday']),
        ('Buy and hold', frame['BuyHold']),
    ]

    for i, (name, returns) in enumerate(series):
        # Floor the curve at $1 so a leg that compounds to nothing stays on a
        # log axis instead of running off to negative infinity.
        equity = compounded_equity(returns).clip(lower=1.0)
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name=name,
            line=dict(color=SERIES_COLORS[i], width=2),
            hovertemplate='%{fullData.name}: $%{y:,.0f}<extra></extra>',
        ))

    style_layout(
        fig,
        f'{label}: where the return shows up',
        f'Growth of ${CAPITAL:,.0f} by session, log scale',
        f'Value of ${CAPITAL:,.0f} stake',
        x_range=(frame.index[0], frame.index[-1]),
        log=True,
    )
    return fig


def cost_sensitivity(frame, label):
    """How the strategy holds up as costs rise, against buy and hold."""
    buy_hold = total_return(frame['BuyHold'])
    rows = []
    for cost in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
        net = apply_costs(frame['Overnight'], cost)
        rows.append({
            'Cost bps/side': cost,
            'Overnight Return': total_return(net),
            'Overnight CAGR': annualized(net),
            'Beats Buy & Hold': 'yes' if total_return(net) > buy_hold else 'no',
        })
    frame_out = pd.DataFrame(rows)
    print(f"\nCost sensitivity, {label} (buy and hold = {buy_hold:,.1%})")
    print(frame_out.assign(**{
        'Overnight Return': frame_out['Overnight Return'].map(lambda v: f"{v:,.1%}"),
        'Overnight CAGR': frame_out['Overnight CAGR'].map(lambda v: f"{v:.2%}"),
    }).to_string(index=False))


def format_metrics(results_df):
    """Percentages as percentages and dollars as dollars, for printing."""
    out = results_df.copy()
    for column in ['Overnight Return', 'Intraday Return', 'Buy & Hold Return']:
        out[column] = out[column].map(lambda v: f"{v:,.1%}")
    for column in ['Overnight CAGR', 'Buy & Hold CAGR', 'Overnight Vol',
                   'Buy & Hold Vol', 'Max Drawdown', 'Win Rate']:
        out[column] = out[column].map(lambda v: f"{v:.2%}")
    out['Fixed-Notional P&L'] = out['Fixed-Notional P&L'].map(lambda v: f"${v:,.0f}")
    for column in ['Sharpe', 'Avg bps/Night', 'Beta vs ' + BENCHMARK, 'Breakeven bps/side']:
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

    basket = build_basket(returns_by_ticker)

    results = [
        compute_metrics(ticker, frame, benchmark_returns, COST_BPS_PER_SIDE)
        for ticker, frame in returns_by_ticker.items()
    ]
    results.append(compute_metrics('BASKET (EW)', basket, benchmark_returns, COST_BPS_PER_SIDE))
    results_df = pd.DataFrame(results)

    print('\nBuy at the close, sell at the next open')
    print(f"Window: {START_DATE or 'full history'} to {END_DATE or 'latest'} | "
          f"${CAPITAL:,.0f} stake | costs: {COST_BPS_PER_SIDE:g} bps per side")
    print('Returns are compounded; "Breakeven bps/side" is the cost at which '
          'overnight stops beating buy and hold.\n')
    print(format_metrics(results_df).to_string(index=False))

    cost_sensitivity(basket, 'equal weight basket')
    if FOCUS_TICKER in returns_by_ticker:
        cost_sensitivity(returns_by_ticker[FOCUS_TICKER], FOCUS_TICKER)

    growth_fig = create_growth_chart(returns_by_ticker, basket, COST_BPS_PER_SIDE)
    pnl_fig = create_pnl_chart(returns_by_ticker, basket, COST_BPS_PER_SIDE)
    focus_frame = returns_by_ticker.get(FOCUS_TICKER, basket)
    focus_label = FOCUS_TICKER if FOCUS_TICKER in returns_by_ticker else 'Equal weight basket'
    session_fig = create_session_chart(focus_frame, focus_label, COST_BPS_PER_SIDE)

    os.makedirs(CHART_DIR, exist_ok=True)
    outputs = [
        ('overnight_growth.html', growth_fig),
        ('overnight_pnl.html', pnl_fig),
        ('overnight_vs_intraday.html', session_fig),
    ]
    for filename, figure in outputs:
        figure.write_html(os.path.join(CHART_DIR, filename))
        print(f"Saved {os.path.join(CHART_DIR, filename)}")

    metrics_path = os.path.join(CHART_DIR, 'overnight_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    show_figures(growth_fig, pnl_fig, session_fig)


if __name__ == '__main__':
    main()
