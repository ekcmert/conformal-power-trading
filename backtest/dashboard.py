from __future__ import annotations

from pathlib import Path

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from backtest.config import SCENARIO_ORDER


SCENARIO_STYLES = {
    "strategy_close_id": {"color": "#1f77b4", "dash": "solid"},
    "strategy_close_id3": {"color": "#2ca02c", "dash": "solid"},
    "strategy_close_id1": {"color": "#ff7f0e", "dash": "solid"},
    "strategy_close_imb": {"color": "#d62728", "dash": "solid"},
    "strategy_perfect_close": {"color": "#111827", "dash": "solid"},
    "perfect_direction_close_id": {"color": "#1f77b4", "dash": "dash"},
    "perfect_direction_close_id3": {"color": "#2ca02c", "dash": "dash"},
    "perfect_direction_close_id1": {"color": "#ff7f0e", "dash": "dash"},
    "perfect_direction_close_imb": {"color": "#d62728", "dash": "dash"},
    "perfect_direction_perfect_close": {"color": "#111827", "dash": "dash"},
}


def _format_dashboard_title(title: str) -> str:
    parts = [part.strip() for part in title.split("|")]
    if len(parts) >= 3:
        return f"{' | '.join(parts[:2])}<br>{' | '.join(parts[2:])}"
    return title


def write_backtest_dashboard(
    hourly_results: pd.DataFrame,
    summary_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    dashboard_path = Path(output_path).expanduser().resolve()
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.72, 0.28],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.08,
        subplot_titles=("Cumulative PnL", "Scenario Summary"),
    )

    ordered_summary = summary_frame.set_index("scenario_name").reindex(SCENARIO_ORDER).reset_index()
    for _, row in ordered_summary.dropna(subset=["scenario_name"]).iterrows():
        scenario_name = str(row["scenario_name"])
        cumulative_column = f"{scenario_name}_cum_pnl"
        if cumulative_column not in hourly_results.columns:
            continue

        style = SCENARIO_STYLES.get(scenario_name, {"color": "#4b5563", "dash": "solid"})
        figure.add_trace(
            go.Scatter(
                x=hourly_results["date"],
                y=hourly_results[cumulative_column],
                mode="lines",
                name=str(row["display_name"]),
                line={"color": style["color"], "dash": style["dash"], "width": 2},
                hovertemplate="Date=%{x}<br>Cumulative PnL=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    table_columns = [
        "display_name",
        "avg_hourly_pnl",
        "avg_unit_pnl",
        "cumulative_pnl",
        "hit_ratio",
        "pnl_volatility",
        "pnl_mean_over_std",
        "max_drawdown",
        "avg_abs_position_size",
    ]
    rounded_summary = ordered_summary.loc[:, table_columns].copy()
    rounded_summary["avg_hourly_pnl"] = rounded_summary["avg_hourly_pnl"].round(3)
    rounded_summary["avg_unit_pnl"] = rounded_summary["avg_unit_pnl"].round(3)
    rounded_summary["cumulative_pnl"] = rounded_summary["cumulative_pnl"].round(3)
    rounded_summary["hit_ratio"] = rounded_summary["hit_ratio"].round(3)
    rounded_summary["pnl_volatility"] = rounded_summary["pnl_volatility"].round(3)
    rounded_summary["pnl_mean_over_std"] = rounded_summary["pnl_mean_over_std"].round(3)
    rounded_summary["max_drawdown"] = rounded_summary["max_drawdown"].round(3)
    rounded_summary["avg_abs_position_size"] = rounded_summary["avg_abs_position_size"].round(3)

    figure.add_trace(
        go.Table(
            header={
                "values": [
                    "Scenario",
                    "Avg Hourly PnL",
                    "Unit PnL (EUR/MWh)",
                    "Cumulative PnL",
                    "Hit Ratio",
                    "PnL Volatility",
                    "Mean / Std",
                    "Max Drawdown",
                    "Avg |Position|",
                ],
                "fill_color": "#e5e7eb",
                "align": "left",
                "font": {"color": "#111827", "size": 12},
            },
            cells={
                "values": [rounded_summary[column] for column in table_columns],
                "fill_color": "#ffffff",
                "align": "left",
                "font": {"color": "#111827", "size": 11},
                "height": 26,
            },
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        title={
            "text": _format_dashboard_title(title),
            "x": 0.02,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 22, "color": "#1f3b73"},
        },
        template="plotly_white",
        height=1020,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "#d1d5db",
            "borderwidth": 1,
            "font": {"size": 12},
        },
        margin={"l": 60, "r": 330, "t": 125, "b": 40},
    )
    figure.update_xaxes(title_text="Date", row=1, col=1)
    figure.update_yaxes(title_text="Cumulative PnL", row=1, col=1)

    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(dashboard_path, include_plotlyjs="cdn")
    return dashboard_path
