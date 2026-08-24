"""
Which names inside a basket are actually driving its overnight return.

A basket average hides whether an edge is broad or one lucky gap, and the two
call for very different conclusions. This splits an equal weight basket into
per-name contributions and then re-runs the comparison with chosen names
removed, so the question "is this real without X" gets a number.

Contribution is exact for the arithmetic mean. The basket's mean nightly return
is the sum of every name's mean(r_it / N_t), where N_t counts the names with
data that night, so the contribution column adds up to the basket mean and the
share column adds to 100%. It is only approximate for the compounded cumulative
figure, which is why the table reports both.

The outlier columns are the point of the exercise. A name whose mean collapses
once its single best night is removed did not have an edge, it had an event.
Median beside mean says the same thing more quietly.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overnight_close_to_open import (  # noqa: E402
    CHART_DIR,
    build_basket,
    download_prices,
    session_returns,
    show_figures,
)
from overnight_sectors import SECTORS, create_recent_year_chart, slice_year  # noqa: E402

# Parameters
YEAR = 2026
ATTRIBUTE = 'Healthcare'              # the basket to break into per-name contributions
EXCLUDE = ['MRNA']                    # names to drop for the "without" comparison
COMPARE = ['Semis', 'Metals', 'Healthcare']
WINDOWS = [60, 40, 20]                # trailing night counts, alongside the full year


def contribution_table(frames, year):
    """Per-name contribution to the equal weight basket's mean nightly return."""
    overnight = pd.DataFrame({t: f['Overnight'] for t, f in frames.items()})
    overnight = overnight[overnight.index.year == year]
    available = overnight.notna().sum(axis=1)
    weighted = overnight.div(available, axis=0)

    rows = []
    for ticker in overnight.columns:
        name_returns = overnight[ticker].dropna()
        if name_returns.empty:
            continue
        best = name_returns.idxmax()
        rows.append({
            'Ticker': ticker,
            'Mean bps': name_returns.mean() * 10_000,
            'Median bps': name_returns.median() * 10_000,
            'Contribution bps': weighted[ticker].sum() / len(overnight) * 10_000,
            'Cumulative': (1 + name_returns).prod() - 1,
            'Best Night': best.date(),
            'Best %': name_returns.max(),
            'Mean ex-Best bps': name_returns.drop(best).mean() * 10_000,
        })

    table = pd.DataFrame(rows)
    total = table['Contribution bps'].sum()
    table['Share'] = table['Contribution bps'] / total if total else np.nan
    return table.sort_values('Contribution bps', ascending=False), total


def trailing_table(baskets, year, windows):
    """Mean bps and cumulative return over the full year and trailing windows."""
    rows = []
    for label, basket in baskets.items():
        overnight = slice_year(basket, year)['Overnight']
        if overnight.empty:
            continue
        row = {'Basket': label}
        for name, series in [(f'{year} YTD', overnight)] + [
            (f'last {w}n', overnight.tail(w)) for w in windows
        ]:
            row[f'{name} bps'] = series.mean() * 10_000
            row[f'{name} cum'] = (1 + series).prod() - 1
        rows.append(row)
    return pd.DataFrame(rows)


def risk_table(baskets, year):
    """Return, dispersion and drawdown of each basket's overnight leg."""
    rows = []
    for label, basket in baskets.items():
        overnight = slice_year(basket, year)['Overnight']
        if overnight.empty:
            continue
        equity = (1 + overnight).cumprod()
        std = overnight.std()
        rows.append({
            'Basket': label,
            'Cumulative': (1 + overnight).prod() - 1,
            'Mean bps': overnight.mean() * 10_000,
            'Median bps': overnight.median() * 10_000,
            'Vol': std * np.sqrt(252),
            'Sharpe': overnight.mean() / std * np.sqrt(252) if std > 0 else np.nan,
            'Win Rate': (overnight > 0).mean(),
            'Max Drawdown': (equity / equity.cummax() - 1).min(),
        })
    return pd.DataFrame(rows)


def show(frame, percent_columns=(), bps_columns=()):
    out = frame.copy()
    for column in percent_columns:
        out[column] = out[column].map(lambda v: f"{v:,.1%}" if pd.notna(v) else '')
    for column in bps_columns:
        out[column] = out[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else '')
    return out.to_string(index=False)


def main():
    sectors = {s: SECTORS[s] for s in dict.fromkeys(COMPARE + [ATTRIBUTE]) if s in SECTORS}
    tickers = sorted({t for names in sectors.values() for t in names})
    prices = download_prices(tickers, None, None)
    if not prices:
        print('No usable price data, nothing to analyse')
        return

    frames = {
        sector: {t: session_returns(prices[t]) for t in names if t in prices}
        for sector, names in sectors.items()
    }

    table, total = contribution_table(frames[ATTRIBUTE], YEAR)
    print(f"\n{ATTRIBUTE} {YEAR}: per-name contribution to a basket mean of {total:.2f} bps per night")
    print(show(table[['Ticker', 'Mean bps', 'Median bps', 'Contribution bps', 'Share',
                      'Cumulative', 'Best Night', 'Best %', 'Mean ex-Best bps']],
               percent_columns=['Share', 'Cumulative', 'Best %'],
               bps_columns=['Mean bps', 'Median bps', 'Contribution bps', 'Mean ex-Best bps']))

    baskets = {sector: build_basket(f) for sector, f in frames.items()}
    excluded = [t for t in EXCLUDE if t in frames[ATTRIBUTE]]
    if excluded:
        label = f"{ATTRIBUTE} ex-{'/'.join(excluded)}"
        baskets[label] = build_basket(
            {t: f for t, f in frames[ATTRIBUTE].items() if t not in excluded}
        )

    trailing = trailing_table(baskets, YEAR, WINDOWS)
    print("\nOvernight leg, trailing windows")
    print(show(trailing,
               percent_columns=[c for c in trailing.columns if c.endswith('cum')],
               bps_columns=[c for c in trailing.columns if c.endswith('bps')]))

    risk = risk_table(baskets, YEAR)
    print(f"\n{YEAR} risk profile of the overnight leg")
    print(show(risk,
               percent_columns=['Cumulative', 'Vol', 'Win Rate', 'Max Drawdown'],
               bps_columns=['Mean bps', 'Median bps', 'Sharpe']))

    fig = create_recent_year_chart(baskets, YEAR)
    fig.update_layout(title=dict(text=(
        f"Overnight hold within {YEAR}, with and without {'/'.join(excluded) or 'exclusions'}"
        f"<br><span style='font-size:13px;color:#52514e'>Cumulative close-to-open return of each "
        f"equal weight basket, reset to zero at the start of the year.</span>"
    )))

    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, f'overnight_attribution_{YEAR}.html')
    fig.write_html(path)
    print(f"\nSaved {path}")

    csv_path = os.path.join(CHART_DIR, f'overnight_attribution_{YEAR}.csv')
    table.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    show_figures(fig)


if __name__ == '__main__':
    main()
