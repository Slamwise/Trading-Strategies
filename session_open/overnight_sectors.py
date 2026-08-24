"""
The overnight (close-to-open) analysis run across robotics, metals and energy.

Nothing about the strategy changes here: this is the same return definition,
the same equal weight basket construction and the same statistics as
overnight_close_to_open.py, pointed at different universes. Everything is
imported from the existing modules so the two stay in step.

The semiconductor basket is carried along as the comparison, because the point
of the exercise is whether the other groups have started to behave the way the
semis did.

The chart covers RECENT_YEAR only, so the question it answers is narrow: within
this year, when did each group's overnight leg start working? Slope is the
thing to read, not the endpoint. A single year is 160 nights, which is not
enough to establish an edge on its own; the all-time table printed above it is
there to say whether the year is continuing something or breaking from it.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overnight_close_to_open import (  # noqa: E402
    BENCHMARK,
    CHART_DIR,
    COST_BPS_PER_SIDE,
    SERIES_COLORS,
    TICKERS as SEMI_TICKERS,
    add_end_label,
    build_basket,
    compute_metrics,
    download_prices,
    format_metrics,
    session_returns,
    show_figures,
    style_layout,
)
from overnight_day_of_week import (  # noqa: E402
    format_stats,
    label_entry_days,
    weekday_stats,
)

# Parameters
RECENT_YEAR = 2026
SECTORS = {
    'Semis': list(SEMI_TICKERS),
    'Robotics': ['ISRG', 'ROK', 'TER', 'SYM', 'PATH', 'SERV'],
    'Metals': ['FCX', 'NEM', 'AA', 'CLF', 'MP', 'SCCO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'OXY', 'DVN'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'PFE', 'MRK', 'MRNA'],
}


def sector_frames(prices, tickers):
    """Per-name session returns for the names in one sector that have data."""
    return {t: session_returns(prices[t]) for t in tickers if t in prices}


def slice_year(frame, year):
    return frame[frame.index.year == year]


def cumulative_percent(overnight):
    """Compounded overnight return, in percent, from the start of the window."""
    return ((1 + overnight.fillna(0)).cumprod() - 1) * 100


def create_recent_year_chart(baskets, year):
    """Cumulative overnight return within a single year, one line per sector."""
    fig = go.Figure()

    curves = {}
    for i, (sector, basket) in enumerate(baskets.items()):
        overnight = slice_year(basket, year)['Overnight']
        if overnight.empty:
            continue
        curve = cumulative_percent(overnight)
        curves[sector] = (curve, SERIES_COLORS[i % len(SERIES_COLORS)])
        fig.add_trace(go.Scatter(
            x=curve.index,
            y=curve.values,
            mode='lines',
            name=sector,
            line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
            hovertemplate='%{fullData.name}: %{y:.1f}%<extra></extra>',
        ))

    first = next(iter(curves.values()))[0]
    style_layout(
        fig,
        f'Overnight hold within {year}',
        'Cumulative close-to-open return of each equal weight basket, '
        'reset to zero at the start of the year. Read the slope, not the endpoint.',
        'Cumulative overnight return',
        x_range=(first.index[0], first.index[-1]),
    )
    # style_layout formats the axis in dollars; this chart is in percent.
    fig.update_yaxes(tickprefix=None, ticksuffix='%', tickformat='.0f')
    for sector, (curve, color) in curves.items():
        add_end_label(fig, curve, sector, color)
    return fig


def main():
    tickers = sorted({t for names in SECTORS.values() for t in names})
    prices = download_prices(tickers + [BENCHMARK], None, None)
    if not prices:
        print('No usable price data, nothing to analyse')
        return

    benchmark_returns = None
    if BENCHMARK in prices:
        benchmark_returns = prices[BENCHMARK]['Close'].pct_change().rename(BENCHMARK)

    baskets = {}
    per_name_rows = []
    for sector, names in SECTORS.items():
        frames = sector_frames(prices, names)
        missing = [t for t in names if t not in frames]
        if missing:
            print(f"{sector}: no data for {', '.join(missing)}")
        if not frames:
            continue

        baskets[sector] = build_basket(frames)
        for ticker, frame in frames.items():
            row = compute_metrics(ticker, frame, benchmark_returns, COST_BPS_PER_SIDE)
            row['Sector'] = sector
            per_name_rows.append(row)

    columns = ['Sector', 'Ticker', 'Start', 'Nights', 'Beta vs ' + BENCHMARK,
               'Overnight Return', 'Overnight CAGR', 'Overnight Vol',
               'Intraday Return', 'Buy & Hold Return', 'Sharpe', 'Win Rate',
               'Avg bps/Night', 'Breakeven bps/side']

    print('\nOvernight hold by sector, full history')
    per_name = pd.DataFrame(per_name_rows)
    print(format_metrics(per_name).sort_values(['Sector', 'Ticker'])[columns].to_string(index=False))

    basket_rows = [
        {**compute_metrics(sector, basket, benchmark_returns, COST_BPS_PER_SIDE),
         'Sector': sector}
        for sector, basket in baskets.items()
    ]
    print('\nSector baskets, full history')
    print(format_metrics(pd.DataFrame(basket_rows))[columns[1:]].to_string(index=False))

    recent_rows = []
    for sector, basket in baskets.items():
        recent = slice_year(basket, RECENT_YEAR)
        if len(recent) < 2:
            continue
        recent_rows.append(compute_metrics(sector, recent, benchmark_returns, COST_BPS_PER_SIDE))

    if not recent_rows:
        print(f"\nNo {RECENT_YEAR} data available")
        return

    print(f'\nSector baskets, {RECENT_YEAR} year to date')
    print(format_metrics(pd.DataFrame(recent_rows))[columns[1:]].to_string(index=False))

    print(f'\nEntry weekday, {RECENT_YEAR} year to date')
    for sector, basket in baskets.items():
        recent = slice_year(basket, RECENT_YEAR)
        if len(recent) < 2:
            continue
        labelled = label_entry_days(recent['Overnight'], recent.index)
        print(f"\n  {sector}")
        print(format_stats(weekday_stats(labelled)).to_string(index=False))

    fig = create_recent_year_chart(baskets, RECENT_YEAR)
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, f'overnight_sectors_{RECENT_YEAR}.html')
    fig.write_html(path)
    print(f"\nSaved {path}")

    csv_path = os.path.join(CHART_DIR, 'overnight_sectors.csv')
    per_name.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    show_figures(fig)


if __name__ == '__main__':
    main()
