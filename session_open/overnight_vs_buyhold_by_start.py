"""
Would you have been better off just holding? Answered for every start date.

For each possible start date the two strategies are run from that date through
the end of the data, and their terminal wealth compared. That turns "is buy and
hold better now" into a curve rather than an opinion: the ratio of the two
terminal multiples, start date by start date.

Reading the output:

  * "B&H wins from" is the earliest start date such that buy and hold finished
    ahead for that start and for every later one. It is the honest answer to
    "if I had started any time recently, would holding have beaten this?"
  * Start dates within MIN_REMAINING nights of the end are dropped. A window of
    three nights is not a comparison of strategies, and without the cut the
    statistic is decided by whatever happened last week.
  * Costs are reported at zero and at one basis point per side, because the
    answer moves a long way between them. Overnight trades roughly 250 round
    trips a year and buy and hold trades once, so any friction at all is paid
    by one side only.

Terminal wealth is a growth multiple, not a return, so it stays positive and
can be drawn on a log axis; 1.0 means the stake came back unchanged.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overnight_close_to_open import (  # noqa: E402
    CHART_DIR,
    INK,
    INK_MUTED,
    SERIES_COLORS,
    apply_costs,
    build_basket,
    download_prices,
    session_returns,
    show_figures,
    style_layout,
)
from overnight_day_of_week import style_panels  # noqa: E402
from overnight_sectors import SECTORS  # noqa: E402

# Parameters
FOCUS = ['MU', 'Semis']        # the two broken out in the levels chart
COMPARE = ['MU', 'Semis', 'Metals', 'Healthcare', 'Energy']
COSTS = [0.0, 1.0]             # bps per side
MIN_REMAINING = 252            # drop start dates with under a year of nights left
WINDOWS = [1, 2, 3, 5, 10, 15, 20]


def terminal_multiple(returns):
    """Growth multiple from each start date through the end of the series."""
    growth = (1 + returns.fillna(0)).cumprod()
    return growth.iloc[-1] / growth.shift(1).fillna(1.0)


def trim(series, minimum=MIN_REMAINING):
    return series.iloc[:-minimum] if minimum and len(series) > minimum else series


def flip_dates(frame, cost):
    """When buy and hold starts winning, and how often it wins overall."""
    overnight = trim(terminal_multiple(apply_costs(frame['Overnight'], cost)))
    buy_hold = trim(terminal_multiple(frame['BuyHold']))
    wins = (overnight / buy_hold) < 1

    if not wins.any():
        return {'Buy & Hold Win Rate': 0.0, 'First Win': None, 'Wins From': None}

    # True where buy and hold wins for this start and for every later one.
    always = wins[::-1].cummin()[::-1]
    return {
        'Buy & Hold Win Rate': wins.mean(),
        'First Win': wins[wins].index[0].date(),
        'Wins From': always[always].index[0].date() if always.any() else None,
    }


def window_table(frame, label):
    """Head to head over the usual trailing windows."""
    rows = []
    end = frame.index[-1]
    for years in WINDOWS:
        window = frame[frame.index >= end - pd.DateOffset(years=years)]
        if len(window) < 20:
            continue
        overnight = (1 + window['Overnight'].fillna(0)).prod() - 1
        buy_hold = (1 + window['BuyHold'].fillna(0)).prod() - 1
        rows.append({
            'Basket': label,
            'Window': f'{years}y',
            'Start': window.index[0].date(),
            'Overnight': overnight,
            'Buy & Hold': buy_hold,
            'Winner': 'overnight' if overnight > buy_hold else 'buy & hold',
        })
    return pd.DataFrame(rows)


def create_levels_chart(frames, cost):
    """Terminal wealth by start date, overnight beside buy and hold."""
    labels = list(frames)
    fig = make_subplots(rows=1, cols=len(labels), subplot_titles=labels,
                        horizontal_spacing=0.10)

    for column, label in enumerate(labels, start=1):
        frame = frames[label]
        overnight = trim(terminal_multiple(apply_costs(frame['Overnight'], cost)))
        buy_hold = trim(terminal_multiple(frame['BuyHold']))
        for i, (name, series) in enumerate([('Overnight', overnight),
                                            ('Buy and hold', buy_hold)]):
            fig.add_trace(
                go.Scatter(
                    x=series.index, y=series.values, mode='lines', name=name,
                    line=dict(color=SERIES_COLORS[i], width=2),
                    legendgroup=name, showlegend=(column == 1),
                    hovertemplate='%{fullData.name}: %{y:,.1f}x<extra></extra>',
                ),
                row=1, col=column,
            )

        crossover = flip_dates(frame, cost)['Wins From']
        if crossover is not None:
            fig.add_vline(
                x=pd.Timestamp(crossover), line_dash='dot', line_color=INK_MUTED,
                row=1, col=column,
                annotation_text=f'  buy and hold wins<br>  from {crossover}',
                annotation_position='top left',
                annotation_font=dict(color=INK_MUTED, size=11),
            )

    style_panels(
        fig,
        'Start here, and which would have won?',
        f'Terminal wealth per $1 from each start date through the end of the data, '
        f'log scale, {"gross of costs" if not cost else f"net of {cost:g} bps per side"}.',
    )
    fig.update_layout(showlegend=True, legend=dict(
        orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5, title=None))
    fig.update_yaxes(type='log', dtick=1, tickprefix=None, ticksuffix='x', tickformat=',')
    fig.update_yaxes(title_text='Terminal wealth per $1', row=1, col=1)
    for annotation in fig.layout.annotations[:len(labels)]:
        annotation.font.update(color=INK, size=14)
    return fig


def create_ratio_chart(frames, cost):
    """Overnight divided by buy and hold, by start date. Below 1 means holding won."""
    fig = go.Figure()

    curves = {}
    for i, (label, frame) in enumerate(frames.items()):
        overnight = trim(terminal_multiple(apply_costs(frame['Overnight'], cost)))
        buy_hold = trim(terminal_multiple(frame['BuyHold']))
        ratio = overnight / buy_hold
        curves[label] = (ratio, SERIES_COLORS[i % len(SERIES_COLORS)])
        fig.add_trace(go.Scatter(
            x=ratio.index, y=ratio.values, mode='lines', name=label,
            line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
            hovertemplate='%{fullData.name}: %{y:,.2f}x<extra></extra>',
        ))

    first = next(iter(curves.values()))[0]
    style_layout(
        fig,
        'Overnight versus buy and hold, by start date',
        f'Ratio of terminal wealth, log scale, '
        f'{"gross of costs" if not cost else f"net of {cost:g} bps per side"}. '
        f'Below the line, holding won.',
        'Overnight / buy and hold',
        x_range=(first.index[0], first.index[-1]),
        log=True,
    )
    fig.update_yaxes(tickprefix=None, ticksuffix='x', tickformat=',')
    fig.add_hline(y=1, line_width=1.5, line_color=INK, line_dash='dash')
    return fig


def main():
    sectors = {s: SECTORS[s] for s in SECTORS if s in COMPARE}
    tickers = sorted({t for names in sectors.values() for t in names} | {'MU'})
    prices = download_prices(tickers, None, None)
    if not prices:
        print('No usable price data, nothing to analyse')
        return

    frames = {}
    for sector, names in sectors.items():
        available = {t: session_returns(prices[t]) for t in names if t in prices}
        if available:
            frames[sector] = build_basket(available)
    if 'MU' in prices:
        frames['MU'] = session_returns(prices['MU'])

    ordered = {label: frames[label] for label in COMPARE if label in frames}

    print('\nHead to head over trailing windows (gross of costs)')
    windows = pd.concat([window_table(f, l) for l, f in ordered.items()])
    print(windows.assign(**{
        'Overnight': windows['Overnight'].map(lambda v: f"{v:,.1%}"),
        'Buy & Hold': windows['Buy & Hold'].map(lambda v: f"{v:,.1%}"),
    }).to_string(index=False))

    print(f'\nWhen does buy and hold take over? '
          f'(start dates with at least {MIN_REMAINING} nights remaining)')
    rows = []
    for label, frame in ordered.items():
        for cost in COSTS:
            rows.append({'Basket': label, 'Cost bps/side': cost, **flip_dates(frame, cost)})
    flips = pd.DataFrame(rows)
    print(flips.assign(**{
        'Buy & Hold Win Rate': flips['Buy & Hold Win Rate'].map(lambda v: f"{v:.1%}"),
    }).to_string(index=False))

    focus = {label: ordered[label] for label in FOCUS if label in ordered}
    levels_fig = create_levels_chart(focus, COSTS[0])
    ratio_fig = create_ratio_chart(ordered, COSTS[0])

    os.makedirs(CHART_DIR, exist_ok=True)
    outputs = [
        ('overnight_vs_buyhold_levels.html', levels_fig),
        ('overnight_vs_buyhold_ratio.html', ratio_fig),
    ]
    for filename, figure in outputs:
        figure.write_html(os.path.join(CHART_DIR, filename))
        print(f"Saved {os.path.join(CHART_DIR, filename)}")

    csv_path = os.path.join(CHART_DIR, 'overnight_vs_buyhold.csv')
    flips.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    show_figures(levels_fig, ratio_fig)


if __name__ == '__main__':
    main()
