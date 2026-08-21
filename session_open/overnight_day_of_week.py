"""
Day-of-week breakdown of the overnight (close-to-open) hold.

Reuses the data plumbing and the return definition from
overnight_close_to_open.py, then splits every night by the weekday of the
close you buy.

Labelling is by ENTRY day, not by the morning you sell, because the entry is
the decision: "should I put this trade on tonight?" So a Friday row is the
weekend hold, bought Friday's close and sold Monday's open, and its average
hold is about three calendar days. Every other row is a one-day hold. The
HoldDays column makes that explicit, and picks up holiday-shortened weeks too.

The run prints an all-time table and a table for RECENT_YEAR on its own. Read
the second one with care: a partial year leaves roughly thirty nights per
weekday, so the confidence intervals are wide enough to swallow most of the
apparent spread. Both tables carry n, a t-statistic and a 95% interval for
exactly that reason, and five weekdays tested at once means one of them
clearing t=2 is close to what you would expect from noise alone.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overnight_close_to_open import (  # noqa: E402
    CHART_DIR,
    COST_BPS_PER_SIDE,
    GRID,
    INK,
    INK_MUTED,
    SERIES_COLORS,
    SURFACE,
    TICKERS,
    apply_costs,
    download_prices,
    session_returns,
    show_figures,
)

# Parameters
RECENT_YEAR = 2026         # the single year broken out beside the all-time view
BASKET_LABEL = 'BASKET (EW)'
WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# Diverging scale for the heatmap: red for negative, gray at zero, blue for
# positive. Kept as light tints so the value printed in each cell stays legible.
DIVERGING = [[0.0, '#f4a6a5'], [0.5, '#f0efec'], [1.0, '#9ec5f4']]


def label_entry_days(overnight, bar_index):
    """Attach the entry (buy) date and weekday to each overnight return.

    Returns come in indexed by the date of the open, i.e. the morning the
    position is sold. The trade was put on at the previous session's close, so
    the entry is the preceding bar. The first return of a series is dropped
    when its entry date is not in the index.
    """
    positions = bar_index.get_indexer(overnight.index)
    keep = positions > 0
    overnight = overnight[keep]
    entry = bar_index[positions[keep] - 1]

    return pd.DataFrame({
        'Return': overnight.to_numpy(),
        'Entry': entry,
        'EntryWeekday': entry.dayofweek,
        'HoldDays': (overnight.index - entry).days,
    }, index=overnight.index)


def weekday_stats(frame):
    """Mean, spread and significance of the overnight return by entry weekday."""
    rows = []
    for day, label in enumerate(WEEKDAYS):
        group = frame[frame['EntryWeekday'] == day]['Return']
        if group.empty:
            continue

        mean = group.mean()
        stderr = group.std() / np.sqrt(len(group)) if len(group) > 1 else np.nan
        rows.append({
            'Entry Day': label,
            'Nights': len(group),
            'Mean bps': mean * 10_000,
            'CI Low bps': (mean - 1.96 * stderr) * 10_000,
            'CI High bps': (mean + 1.96 * stderr) * 10_000,
            't-stat': mean / stderr if stderr and stderr > 0 else np.nan,
            'Win Rate': (group > 0).mean(),
            'Cumulative': (1 + group).prod() - 1,
            'Avg Hold Days': frame[frame['EntryWeekday'] == day]['HoldDays'].mean(),
        })
    return pd.DataFrame(rows)


def welch_test(frame, day_a, day_b):
    """Welch t-statistic for the difference between two entry weekdays."""
    a = frame[frame['EntryWeekday'] == WEEKDAYS.index(day_a)]['Return']
    b = frame[frame['EntryWeekday'] == WEEKDAYS.index(day_b)]['Return']
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def format_stats(stats):
    out = stats.copy()
    for column in ['Mean bps', 'CI Low bps', 'CI High bps', 't-stat', 'Avg Hold Days']:
        out[column] = out[column].map(lambda v: f"{v:.2f}")
    out['Win Rate'] = out['Win Rate'].map(lambda v: f"{v:.1%}")
    out['Cumulative'] = out['Cumulative'].map(lambda v: f"{v:,.1%}")
    return out


def weekday_matrix(frames_by_name):
    """Mean bps per name per entry weekday, as a name x weekday matrix."""
    return pd.DataFrame(
        {
            name: [
                frame[frame['EntryWeekday'] == day]['Return'].mean() * 10_000
                for day in range(len(WEEKDAYS))
            ]
            for name, frame in frames_by_name.items()
        },
        index=WEEKDAYS,
    ).T


def style_panels(fig, title, subtitle, yaxis_title=None):
    """Shared chrome for the two-panel figures."""
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:13px;color:{INK_MUTED}'>{subtitle}</span>",
            font=dict(color=INK, size=20),
            x=0.02,
            xanchor='left',
        ),
        template='plotly_white',
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_MUTED, size=13),
        showlegend=False,
        margin=dict(l=90, r=60, t=110, b=60),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks='outside', tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID,
                     title_text=yaxis_title)
    return fig


def create_weekday_chart(all_time, recent, recent_label):
    """Mean overnight return by entry weekday, with 95% confidence intervals.

    Two panels rather than one axis: the recent year runs several times hotter
    than the full history, and forcing both onto a shared scale would flatten
    the all-time bars into nothing. Separate panels keep each readable, and the
    error bars carry the point the numbers alone would hide.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('All time', recent_label),
        horizontal_spacing=0.12,
    )

    for column, stats in enumerate([all_time, recent], start=1):
        errors = stats['CI High bps'] - stats['Mean bps']
        fig.add_trace(
            go.Bar(
                x=stats['Entry Day'],
                y=stats['Mean bps'],
                marker=dict(color=SERIES_COLORS[column - 1], line=dict(width=0)),
                error_y=dict(type='data', array=errors, color=INK_MUTED, thickness=1.5, width=5),
                hovertemplate='%{x} entry: %{y:.1f} bps<extra></extra>',
            ),
            row=1, col=column,
        )
        fig.add_hline(y=0, line_width=1, line_color=INK_MUTED, row=1, col=column)

    style_panels(
        fig,
        'Overnight return by day of week',
        'Equal weight basket, mean basis points per night with 95% confidence intervals. '
        'Friday entry is the weekend hold.',
    )
    fig.update_yaxes(title_text='Mean bps per night', row=1, col=1)
    for annotation in fig.layout.annotations[:2]:
        annotation.font.update(color=INK, size=14)
    return fig


def create_heatmap(all_time_matrix, recent_matrix, recent_label):
    """Mean bps per name per entry weekday, all time beside the recent year."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('All time', recent_label),
        horizontal_spacing=0.12,
    )

    for column, matrix in enumerate([all_time_matrix, recent_matrix], start=1):
        limit = np.nanmax(np.abs(matrix.to_numpy())) or 1.0
        fig.add_trace(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale=DIVERGING,
                zmid=0,
                zmin=-limit,
                zmax=limit,
                showscale=False,
                xgap=2,
                ygap=2,
                text=[[f"{v:.0f}" for v in row] for row in matrix.to_numpy()],
                texttemplate='%{text}',
                textfont=dict(color=INK, size=11),
                hovertemplate='%{y}, %{x} entry: %{z:.1f} bps<extra></extra>',
            ),
            row=1, col=column,
        )

    style_panels(
        fig,
        'Overnight return by name and day of week',
        'Mean basis points per night. Blue is positive, red negative; every cell is labelled.',
    )
    fig.update_yaxes(autorange='reversed', showgrid=False)
    fig.update_xaxes(showgrid=False)
    for annotation in fig.layout.annotations[:2]:
        annotation.font.update(color=INK, size=14)
    return fig


def main():
    prices = download_prices(list(TICKERS), None, None)
    if not prices:
        print('No usable price data, nothing to analyse')
        return

    frames = {}
    overnight_by_ticker = {}
    for ticker, data in prices.items():
        overnight = apply_costs(session_returns(data)['Overnight'], COST_BPS_PER_SIDE)
        overnight_by_ticker[ticker] = overnight
        frames[ticker] = label_entry_days(overnight, data.index)

    basket = pd.DataFrame(overnight_by_ticker).mean(axis=1, skipna=True).dropna()
    frames[BASKET_LABEL] = label_entry_days(basket, basket.index)

    basket_frame = frames[BASKET_LABEL]
    recent_frame = basket_frame[basket_frame.index.year == RECENT_YEAR]
    recent_label = f'{RECENT_YEAR} year to date'

    all_time_stats = weekday_stats(basket_frame)
    recent_stats = weekday_stats(recent_frame)

    print('\nOvernight hold by day of week, equal weight basket')
    print('Rows are labelled by the day you BUY the close. Friday entry sells '
          'at Monday\'s open, so it carries the weekend.\n')
    print(f"All time ({basket_frame.index[0].date()} to {basket_frame.index[-1].date()}, "
          f"{len(basket_frame):,} nights)")
    print(format_stats(all_time_stats).to_string(index=False))

    best, worst = 'Mon', 'Thu'
    print(f"\n  {best} minus {worst} entry, Welch t = {welch_test(basket_frame, best, worst):.2f} "
          f"(the spread across weekdays is far weaker evidence than the fact that every day is positive)")

    if recent_frame.empty:
        print(f"\nNo {RECENT_YEAR} data available")
    else:
        print(f"\n{recent_label} ({recent_frame.index[0].date()} to "
              f"{recent_frame.index[-1].date()}, {len(recent_frame):,} nights)")
        print(format_stats(recent_stats).to_string(index=False))
        print(f"\n  {RECENT_YEAR} runs hot overall: {recent_frame['Return'].mean() * 10_000:.2f} bps per night "
              f"vs {basket_frame['Return'].mean() * 10_000:.2f} all time, at "
              f"{recent_frame['Return'].std() * np.sqrt(252):.1%} vol vs "
              f"{basket_frame['Return'].std() * np.sqrt(252):.1%}.")
        print('  Compare weekdays within a panel, not across them.')

    ordered = [t for t in TICKERS if t in frames] + [BASKET_LABEL]
    all_time_matrix = weekday_matrix({name: frames[name] for name in ordered})
    recent_matrix = weekday_matrix({
        name: frames[name][frames[name].index.year == RECENT_YEAR] for name in ordered
    })

    print('\nMean bps by entry day, per name (all time)')
    print(all_time_matrix.round(2).to_string())
    print(f'\nMean bps by entry day, per name ({recent_label})')
    print(recent_matrix.round(2).to_string())

    weekday_fig = create_weekday_chart(all_time_stats, recent_stats, recent_label)
    heatmap_fig = create_heatmap(all_time_matrix, recent_matrix, recent_label)

    os.makedirs(CHART_DIR, exist_ok=True)
    outputs = [
        ('overnight_day_of_week.html', weekday_fig),
        ('overnight_day_of_week_heatmap.html', heatmap_fig),
    ]
    for filename, figure in outputs:
        figure.write_html(os.path.join(CHART_DIR, filename))
        print(f"Saved {os.path.join(CHART_DIR, filename)}")

    stats_path = os.path.join(CHART_DIR, 'overnight_day_of_week.csv')
    pd.concat([
        all_time_stats.assign(Window='All time'),
        recent_stats.assign(Window=recent_label),
    ]).to_csv(stats_path, index=False)
    print(f"Saved {stats_path}")

    show_figures(weekday_fig, heatmap_fig)


if __name__ == '__main__':
    main()
