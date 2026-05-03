from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.config import (
    BACKTEST_RESULTS_ROOT,
    CALIBRATED_LOWER_COLUMN,
    CALIBRATED_UPPER_COLUMN,
    CLOSE_PRICE_COLUMNS,
    DA_PRICE_COLUMN,
    DATE_COLUMN,
    DEFAULT_ENTRY_BAND_FRACTION,
    DEFAULT_EXIT_BAND_FRACTION,
    DEFAULT_POSITION_CAP,
    DEFAULT_POSITION_METHOD,
    DEFAULT_SIGNAL_FAMILY,
    DEFAULT_SIGNAL_POWER,
    DEFAULT_SIGNAL_SCALE,
    DEFAULT_SIZE_MAP,
    INTERVAL_WIDTH_COLUMN,
    REFERENCE_PREDICTION_COLUMN,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    Y_PATH,
)
from backtest.dashboard import write_backtest_dashboard
from backtest.io import infer_output_dir, load_prediction_frame, load_price_frame, resolve_prediction_csv


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    display_name: str
    group: str
    close_market: str
    pnl_column: str
    position_column: str
    best_market_column: str | None = None


@dataclass
class BacktestConfig:
    pred_path: Path | str
    conf_factor: float = 1.0
    size_map: dict[float, float] = field(default_factory=lambda: dict(DEFAULT_SIZE_MAP))
    mapping_mode: str = "step"
    position_method: str = DEFAULT_POSITION_METHOD
    entry_band_fraction: float = DEFAULT_ENTRY_BAND_FRACTION
    exit_band_fraction: float = DEFAULT_EXIT_BAND_FRACTION
    signal_family: str = DEFAULT_SIGNAL_FAMILY
    signal_scale: float = DEFAULT_SIGNAL_SCALE
    signal_power: float = DEFAULT_SIGNAL_POWER
    position_cap: float = DEFAULT_POSITION_CAP
    min_interval_width: float = 1e-6
    y_path: Path = Y_PATH
    output_dir: Path | None = None
    results_root: Path = BACKTEST_RESULTS_ROOT

    def __post_init__(self) -> None:
        self.pred_path = Path(self.pred_path).expanduser().resolve()
        self.y_path = Path(self.y_path).expanduser().resolve()
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.results_root = Path(self.results_root).expanduser().resolve()
        self.size_map = dict(sorted({float(key): float(value) for key, value in self.size_map.items()}.items()))
        self.mapping_mode = self.mapping_mode.lower()
        self.position_method = self.position_method.lower()
        self.signal_family = self.signal_family.lower()
        if self.position_method == "partial_interval_band":
            self.position_method = "interval_band"

        if self.conf_factor <= 0:
            raise ValueError(f"conf_factor must be positive, got {self.conf_factor}.")
        if self.min_interval_width <= 0:
            raise ValueError(f"min_interval_width must be positive, got {self.min_interval_width}.")
        if self.mapping_mode not in {"step", "linear"}:
            raise ValueError("mapping_mode must be 'step' or 'linear'.")
        if self.position_method not in {"legacy_map", "interval_band"}:
            raise ValueError("position_method must be 'legacy_map' or 'interval_band'.")
        if self.signal_family not in {"linear", "tanh", "softsign", "arctan"}:
            raise ValueError("signal_family must be 'linear', 'tanh', 'softsign', or 'arctan'.")
        if not 0 <= self.entry_band_fraction <= 1:
            raise ValueError(f"entry_band_fraction must be between 0 and 1, got {self.entry_band_fraction}.")
        if not is_openable_interval_band(self.entry_band_fraction, self.exit_band_fraction):
            raise ValueError(
                "exit_band_fraction must be greater than entry_band_fraction so the active trading corridor exists. "
                "Configs with exit_band_fraction <= entry_band_fraction cannot open a position."
            )
        if self.signal_scale <= 0:
            raise ValueError(f"signal_scale must be positive, got {self.signal_scale}.")
        if self.signal_power <= 0:
            raise ValueError(f"signal_power must be positive, got {self.signal_power}.")
        if self.position_cap <= 0:
            raise ValueError(f"position_cap must be positive, got {self.position_cap}.")
        if not self.size_map:
            raise ValueError("size_map cannot be empty.")


def is_openable_interval_band(entry_band_fraction: float, exit_band_fraction: float) -> bool:
    return float(exit_band_fraction) > float(entry_band_fraction)


@dataclass
class BacktestArtifacts:
    output_dir: Path
    prediction_csv: Path
    hourly_results_path: Path
    summary_path: Path
    dashboard_path: Path
    metadata_path: Path
    hourly_results: pd.DataFrame
    summary: pd.DataFrame


def apply_size_mapping(
    margins: pd.Series | Iterable[float],
    *,
    size_map: dict[float, float],
    mode: str = "step",
) -> pd.Series:
    margin_series = pd.Series(margins, copy=False, dtype="float64")
    sorted_items = sorted((float(key), float(value)) for key, value in size_map.items())
    thresholds = np.asarray([item[0] for item in sorted_items], dtype="float64")
    sizes = np.asarray([item[1] for item in sorted_items], dtype="float64")

    if mode == "linear":
        interpolated = np.interp(margin_series.to_numpy(dtype="float64"), thresholds, sizes, left=sizes[0], right=sizes[-1])
        return pd.Series(interpolated, index=margin_series.index, dtype="float64")

    zero_size = float(size_map.get(0.0, 0.0))
    negative_thresholds = thresholds[thresholds < 0]
    positive_thresholds = thresholds[thresholds > 0]
    threshold_to_size = {threshold: size for threshold, size in sorted_items}

    def map_single_margin(margin: float) -> float:
        if pd.isna(margin):
            return np.nan
        if margin < 0:
            crossed = negative_thresholds[negative_thresholds >= margin]
            if crossed.size == 0:
                return zero_size
            return threshold_to_size[float(crossed.min())]
        if margin > 0:
            crossed = positive_thresholds[positive_thresholds <= margin]
            if crossed.size == 0:
                return zero_size
            return threshold_to_size[float(crossed.max())]
        return zero_size

    return margin_series.apply(map_single_margin).astype("float64")


def compute_max_drawdown(cumulative_pnl: pd.Series) -> float:
    if cumulative_pnl.empty:
        return float("nan")
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    return float(drawdown.max())


def compute_unit_pnl_series(pnl: pd.Series, position: pd.Series) -> pd.Series:
    pnl_series = pnl.astype("float64")
    position_abs = position.astype("float64").abs()
    unit_pnl = pd.Series(np.nan, index=pnl_series.index, dtype="float64")
    active_mask = position_abs > 0
    unit_pnl.loc[active_mask] = pnl_series.loc[active_mask] / position_abs.loc[active_mask]
    return unit_pnl


def compute_total_unit_pnl(pnl: pd.Series, position: pd.Series) -> float:
    pnl_series = pnl.astype("float64")
    position_abs = position.astype("float64").abs()
    total_position = float(position_abs.sum())
    if total_position <= 0:
        return float("nan")
    return float(pnl_series.sum() / total_position)


def compute_scenario_metrics(pnl: pd.Series, position: pd.Series) -> dict[str, float]:
    valid_mask = pnl.notna() & position.notna()
    pnl_valid = pnl.loc[valid_mask].astype("float64")
    position_valid = position.loc[valid_mask].astype("float64")
    active_mask = position_valid.abs() > 0
    active_pnl_valid = pnl_valid.loc[active_mask]
    unit_pnl = compute_total_unit_pnl(pnl_valid, position_valid)

    if pnl_valid.empty:
        return {
            "hours": 0,
            "avg_hourly_pnl": float("nan"),
            "avg_unit_pnl": float("nan"),
            "cumulative_pnl": float("nan"),
            "hit_ratio": float("nan"),
            "pnl_volatility": float("nan"),
            "pnl_mean_over_std": float("nan"),
            "max_drawdown": float("nan"),
            "avg_abs_position_size": float("nan"),
        }

    cumulative_pnl = pnl_valid.cumsum()
    pnl_volatility = float(pnl_valid.std(ddof=0))
    pnl_mean_over_std = float(pnl_valid.mean() / pnl_volatility) if pnl_volatility > 0 else float("nan")

    return {
        "hours": int(len(pnl_valid)),
        "avg_hourly_pnl": float(pnl_valid.mean()),
        "avg_unit_pnl": unit_pnl,
        "cumulative_pnl": float(cumulative_pnl.iloc[-1]),
        "hit_ratio": float((active_pnl_valid > 0).mean()) if not active_pnl_valid.empty else float("nan"),
        "pnl_volatility": pnl_volatility,
        "pnl_mean_over_std": pnl_mean_over_std,
        "max_drawdown": compute_max_drawdown(cumulative_pnl),
        "avg_abs_position_size": float(position_valid.abs().mean()),
    }


def apply_signal_family(
    base_signal: pd.Series | Iterable[float],
    *,
    family: str,
    scale: float,
    position_cap: float,
) -> pd.Series:
    signal_series = pd.Series(base_signal, copy=False, dtype="float64")
    scaled_signal = scale * signal_series

    if family == "tanh":
        transformed = np.tanh(scaled_signal)
    elif family == "linear":
        transformed = scaled_signal.clip(lower=-1.0, upper=1.0)
    elif family == "softsign":
        transformed = scaled_signal / (1.0 + scaled_signal.abs())
    elif family == "arctan":
        transformed = (2.0 / np.pi) * np.arctan(scaled_signal)
    else:
        raise ValueError(f"Unsupported signal family: {family}")

    return (position_cap * transformed).astype("float64")


def _best_market(metric_frame: pd.DataFrame) -> pd.Series:
    valid_rows = metric_frame.notna().any(axis=1)
    best_markets = pd.Series(pd.NA, index=metric_frame.index, dtype="object")
    if not valid_rows.any():
        return best_markets

    best_indices = metric_frame.fillna(-np.inf).to_numpy().argmax(axis=1)
    market_names = np.asarray(metric_frame.columns, dtype=object)
    valid_array = valid_rows.to_numpy()
    best_markets.loc[valid_rows] = market_names[best_indices[valid_array]]
    return best_markets


class IntervalBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self) -> BacktestArtifacts:
        prediction_csv = resolve_prediction_csv(self.config.pred_path)
        output_dir = self.config.output_dir or infer_output_dir(prediction_csv, results_root=self.config.results_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        prediction_frame = load_prediction_frame(prediction_csv)
        price_frame = load_price_frame(self.config.y_path)
        hourly_results, scenario_specs, merge_stats = self._build_hourly_results(prediction_frame, price_frame)
        summary_frame = self._build_summary_frame(hourly_results, scenario_specs)

        hourly_results_path = output_dir / "backtest_hourly_results.csv"
        summary_path = output_dir / "backtest_summary.csv"
        dashboard_path = output_dir / "backtest_dashboard.html"
        metadata_path = output_dir / "run_metadata.json"

        hourly_results.to_csv(hourly_results_path, index=False)
        summary_frame.to_csv(summary_path, index=False)

        if self.config.position_method == "interval_band":
            title = (
                f"Backtest Dashboard | {prediction_csv.parent.name} | "
                f"Method=interval_band/{self.config.signal_family} | "
                f"Entry={self.config.entry_band_fraction:g} | "
                f"Exit={self.config.exit_band_fraction:g} | "
                f"Scale={self.config.signal_scale:g} | MaxPos={self.config.position_cap:g}"
            )
        else:
            title = (
                f"Backtest Dashboard | {prediction_csv.parent.name} | "
                f"Method=legacy_map | Confidence={self.config.conf_factor:g} | "
                f"Mapping={self.config.mapping_mode}"
            )
        write_backtest_dashboard(hourly_results, summary_frame, dashboard_path, title=title)

        metadata = {
            "prediction_csv": str(prediction_csv),
            "prediction_dir": str(prediction_csv.parent),
            "output_dir": str(output_dir),
            "y_path": str(self.config.y_path),
            "position_method": self.config.position_method,
            "min_interval_width": self.config.min_interval_width,
            **merge_stats,
        }
        if self.config.position_method == "interval_band":
            metadata.update(
                {
                    "entry_band_fraction": self.config.entry_band_fraction,
                    "exit_band_fraction": self.config.exit_band_fraction,
                    "signal_family": self.config.signal_family,
                    "signal_scale": self.config.signal_scale,
                    "signal_power": self.config.signal_power,
                    "position_cap": self.config.position_cap,
                    "signal_families_supported": ["tanh", "linear", "softsign", "arctan"],
                }
            )
        else:
            metadata.update(
                {
                    "conf_factor": self.config.conf_factor,
                    "mapping_mode": self.config.mapping_mode,
                    "position_cap": self.config.position_cap,
                    "size_map": {str(key): value for key, value in self.config.size_map.items()},
                }
            )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return BacktestArtifacts(
            output_dir=output_dir,
            prediction_csv=prediction_csv,
            hourly_results_path=hourly_results_path,
            summary_path=summary_path,
            dashboard_path=dashboard_path,
            metadata_path=metadata_path,
            hourly_results=hourly_results,
            summary=summary_frame,
        )

    def _build_hourly_results(
        self,
        prediction_frame: pd.DataFrame,
        price_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[ScenarioSpec], dict[str, int]]:
        merged = prediction_frame.merge(price_frame, on=DATE_COLUMN, how="inner", validate="one_to_one")
        required_core_columns = [
            DATE_COLUMN,
            REFERENCE_PREDICTION_COLUMN,
            CALIBRATED_LOWER_COLUMN,
            CALIBRATED_UPPER_COLUMN,
            INTERVAL_WIDTH_COLUMN,
            DA_PRICE_COLUMN,
        ]
        valid_core_mask = merged[required_core_columns].notna().all(axis=1)
        frame = merged.loc[valid_core_mask].copy().sort_values(DATE_COLUMN).reset_index(drop=True)

        frame = self._build_strategy_positions(frame)
        frame["strategy_abs_position"] = frame["strategy_position"].abs()
        frame["strategy_direction"] = np.sign(frame["strategy_position"])

        strategy_market_pnls = pd.DataFrame(index=frame.index)
        perfect_direction_market_pnls = pd.DataFrame(index=frame.index)
        scenario_specs: list[ScenarioSpec] = []

        for market_name, close_column in CLOSE_PRICE_COLUMNS.items():
            market_slug = market_name.lower()
            delta_column = f"{market_slug}_close_minus_da"
            strategy_pnl_column = f"strategy_close_{market_slug}_pnl"
            strategy_unit_pnl_column = f"strategy_close_{market_slug}_unit_pnl"
            perfect_position_column = f"perfect_direction_close_{market_slug}_position"
            perfect_pnl_column = f"perfect_direction_close_{market_slug}_pnl"
            perfect_unit_pnl_column = f"perfect_direction_close_{market_slug}_unit_pnl"
            perfect_sign_column = f"perfect_direction_close_{market_slug}_sign"

            frame[delta_column] = frame[close_column] - frame[DA_PRICE_COLUMN]
            frame[strategy_pnl_column] = frame["strategy_position"] * frame[delta_column]
            frame[perfect_sign_column] = np.where(
                frame[delta_column] > 0,
                1.0,
                np.where(frame[delta_column] < 0, -1.0, 0.0),
            )
            frame[perfect_position_column] = frame["strategy_abs_position"] * frame[perfect_sign_column]
            frame[perfect_pnl_column] = frame["strategy_abs_position"] * frame[delta_column].abs()
            frame[strategy_unit_pnl_column] = compute_unit_pnl_series(
                frame[strategy_pnl_column],
                frame["strategy_position"],
            )
            frame[perfect_unit_pnl_column] = compute_unit_pnl_series(
                frame[perfect_pnl_column],
                frame[perfect_position_column],
            )

            strategy_market_pnls[market_name] = frame[strategy_pnl_column]
            perfect_direction_market_pnls[market_name] = frame[perfect_pnl_column]

            scenario_specs.append(
                ScenarioSpec(
                    name=f"strategy_close_{market_slug}",
                    display_name=SCENARIO_LABELS[f"strategy_close_{market_slug}"],
                    group="strategy",
                    close_market=market_name,
                    pnl_column=strategy_pnl_column,
                    position_column="strategy_position",
                )
            )
            scenario_specs.append(
                ScenarioSpec(
                    name=f"perfect_direction_close_{market_slug}",
                    display_name=SCENARIO_LABELS[f"perfect_direction_close_{market_slug}"],
                    group="perfect_direction",
                    close_market=market_name,
                    pnl_column=perfect_pnl_column,
                    position_column=perfect_position_column,
                )
            )

        frame["strategy_perfect_close_pnl"] = strategy_market_pnls.max(axis=1, skipna=True)
        frame["strategy_perfect_close_best_market"] = _best_market(strategy_market_pnls)
        frame["perfect_direction_perfect_close_pnl"] = perfect_direction_market_pnls.max(axis=1, skipna=True)
        frame["perfect_direction_perfect_close_best_market"] = _best_market(perfect_direction_market_pnls)
        frame["strategy_perfect_close_unit_pnl"] = compute_unit_pnl_series(
            frame["strategy_perfect_close_pnl"],
            frame["strategy_position"],
        )
        frame["perfect_direction_perfect_close_unit_pnl"] = compute_unit_pnl_series(
            frame["perfect_direction_perfect_close_pnl"],
            frame["strategy_abs_position"],
        )

        scenario_specs.append(
            ScenarioSpec(
                name="strategy_perfect_close",
                display_name=SCENARIO_LABELS["strategy_perfect_close"],
                group="strategy",
                close_market="perfect_close",
                pnl_column="strategy_perfect_close_pnl",
                position_column="strategy_position",
                best_market_column="strategy_perfect_close_best_market",
            )
        )
        scenario_specs.append(
            ScenarioSpec(
                name="perfect_direction_perfect_close",
                display_name=SCENARIO_LABELS["perfect_direction_perfect_close"],
                group="perfect_direction",
                close_market="perfect_close",
                pnl_column="perfect_direction_perfect_close_pnl",
                position_column="strategy_abs_position",
                best_market_column="perfect_direction_perfect_close_best_market",
            )
        )

        ordered_scenarios = {scenario.name: scenario for scenario in scenario_specs}
        scenario_specs = [ordered_scenarios[name] for name in SCENARIO_ORDER]
        for scenario in scenario_specs:
            frame[f"{scenario.name}_cum_pnl"] = frame[scenario.pnl_column].fillna(0.0).cumsum()

        merge_stats = {
            "prediction_rows": int(len(prediction_frame)),
            "price_rows": int(len(price_frame)),
            "merged_rows": int(len(merged)),
            "backtest_rows": int(len(frame)),
            "dropped_rows_after_merge": int(len(merged) - len(frame)),
        }
        return frame, scenario_specs, merge_stats

    def _build_strategy_positions(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working["interval_width_clipped"] = working[INTERVAL_WIDTH_COLUMN].clip(lower=self.config.min_interval_width)
        # Positive values mean DA is above the reference forecast (overpriced -> short).
        # Negative values mean DA is below the reference forecast (underpriced -> long).
        working["margin_to_reference"] = working[DA_PRICE_COLUMN] - working[REFERENCE_PREDICTION_COLUMN]

        if self.config.position_method == "legacy_map":
            working["confidence"] = self.config.conf_factor / working["interval_width_clipped"]
            working["lower_side_width"] = (
                working[REFERENCE_PREDICTION_COLUMN] - working[CALIBRATED_LOWER_COLUMN]
            ).clip(lower=self.config.min_interval_width)
            working["upper_side_width"] = (
                working[CALIBRATED_UPPER_COLUMN] - working[REFERENCE_PREDICTION_COLUMN]
            ).clip(lower=self.config.min_interval_width)
            working["lower_entry_price"] = working[REFERENCE_PREDICTION_COLUMN]
            working["upper_entry_price"] = working[REFERENCE_PREDICTION_COLUMN]
            working["underpricing_excess"] = np.maximum(
                working[REFERENCE_PREDICTION_COLUMN] - working[DA_PRICE_COLUMN],
                0.0,
            )
            working["overpricing_excess"] = np.maximum(
                working[DA_PRICE_COLUMN] - working[REFERENCE_PREDICTION_COLUMN],
                0.0,
            )
            working["signed_signal"] = (
                working["underpricing_excess"] / working["lower_side_width"]
                - working["overpricing_excess"] / working["upper_side_width"]
            )
            working["width_multiplier"] = working["confidence"]
            working["base_position"] = apply_size_mapping(
                working["margin_to_reference"],
                size_map=self.config.size_map,
                mode=self.config.mapping_mode,
            )
            working["raw_strategy_position"] = working["base_position"] * working["width_multiplier"]
            working["strategy_position"] = working["raw_strategy_position"].clip(
                lower=-self.config.position_cap,
                upper=self.config.position_cap,
            )
            return working

        working["lower_side_width"] = (
            working[REFERENCE_PREDICTION_COLUMN] - working[CALIBRATED_LOWER_COLUMN]
        ).clip(lower=self.config.min_interval_width)
        working["upper_side_width"] = (
            working[CALIBRATED_UPPER_COLUMN] - working[REFERENCE_PREDICTION_COLUMN]
        ).clip(lower=self.config.min_interval_width)
        working["lower_entry_price"] = (
            working[REFERENCE_PREDICTION_COLUMN]
            - self.config.entry_band_fraction * working["lower_side_width"]
        )
        working["upper_entry_price"] = (
            working[REFERENCE_PREDICTION_COLUMN]
            + self.config.entry_band_fraction * working["upper_side_width"]
        )
        working["lower_exit_price"] = (
            working[REFERENCE_PREDICTION_COLUMN]
            - self.config.exit_band_fraction * working["lower_side_width"]
        )
        working["upper_exit_price"] = (
            working[REFERENCE_PREDICTION_COLUMN]
            + self.config.exit_band_fraction * working["upper_side_width"]
        )
        working["lower_active_band_width"] = (
            working["lower_entry_price"] - working["lower_exit_price"]
        ).clip(lower=self.config.min_interval_width)
        working["upper_active_band_width"] = (
            working["upper_exit_price"] - working["upper_entry_price"]
        ).clip(lower=self.config.min_interval_width)

        long_active = (
            (working[DA_PRICE_COLUMN] < working["lower_entry_price"])
            & (working[DA_PRICE_COLUMN] >= working["lower_exit_price"])
        )
        short_active = (
            (working[DA_PRICE_COLUMN] > working["upper_entry_price"])
            & (working[DA_PRICE_COLUMN] <= working["upper_exit_price"])
        )

        working["underpricing_excess"] = np.where(
            long_active,
            working["lower_entry_price"] - working[DA_PRICE_COLUMN],
            0.0,
        )
        working["overpricing_excess"] = np.where(
            short_active,
            working[DA_PRICE_COLUMN] - working["upper_entry_price"],
            0.0,
        )
        working["long_band_progress"] = np.where(
            long_active,
            working["underpricing_excess"] / working["lower_active_band_width"],
            0.0,
        )
        working["short_band_progress"] = np.where(
            short_active,
            working["overpricing_excess"] / working["upper_active_band_width"],
            0.0,
        )
        working["signal_progress"] = working["long_band_progress"] + working["short_band_progress"]
        working["confidence"] = working["signal_progress"]
        working["signed_signal"] = working["long_band_progress"] - working["short_band_progress"]
        working["base_position"] = (
            np.sign(working["signed_signal"])
            * np.power(np.abs(working["signed_signal"]), self.config.signal_power)
        )
        working["raw_strategy_position"] = apply_signal_family(
            working["base_position"],
            family=self.config.signal_family,
            scale=self.config.signal_scale,
            position_cap=self.config.position_cap,
        )
        working["strategy_position"] = working["raw_strategy_position"]
        return working

    def _build_summary_frame(
        self,
        hourly_results: pd.DataFrame,
        scenario_specs: list[ScenarioSpec],
    ) -> pd.DataFrame:
        summary_rows: list[dict[str, object]] = []
        for rank, scenario in enumerate(scenario_specs, start=1):
            metrics = compute_scenario_metrics(
                hourly_results[scenario.pnl_column],
                hourly_results[scenario.position_column],
            )
            summary_rows.append(
                {
                    "scenario_rank": rank,
                    "scenario_name": scenario.name,
                    "display_name": scenario.display_name,
                    "scenario_group": scenario.group,
                    "close_market": scenario.close_market,
                    **metrics,
                }
            )
        return pd.DataFrame(summary_rows)
