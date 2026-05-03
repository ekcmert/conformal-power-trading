from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_models.point_estimation.common import REPO_ROOT
from exp2 import regime_free_experiment as rfe
from exp3 import adaptive_conformal_inference as aci
from exp3 import ensemble_batch_prediction_intervals as enbpi
from exp3 import local_adaptive_conformal_prediction as lacp
from exp3 import mondrian_conformal_prediction as mcp


RESULTS_ROOT = REPO_ROOT / "results" / "regime_aware"
SUMMARY_FILENAME = "all_method_results.csv"
PIPELINE_LIST: tuple[str, ...] = ("mcp", "enbpi", "lacp", "aci")
IMPLEMENTED_PIPELINES: tuple[str, ...] = ("mcp", "enbpi", "lacp", "aci")


@dataclass(frozen=True)
class RegimeAwareExperimentResult:
    ranked_results: pd.DataFrame
    pipeline_results: dict[str, pd.DataFrame]
    pipeline_failures: dict[str, pd.DataFrame]
    skipped_pipelines: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and/or combine regime-aware experiment summaries across MCP, LACP, "
            "ACI, and ENBPI."
        ),
    )
    parser.add_argument(
        "--run-pipelines",
        action="store_true",
        help="Run the selected regime-aware pipelines before combining summaries.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(PIPELINE_LIST),
        help="Pipelines to run/combine. Available: mcp, enbpi, lacp, aci.",
    )
    parser.add_argument("--calibration-range", type=int, default=mcp.CALIBRATION_RANGE)
    parser.add_argument("--calibration-frequency", type=int, default=mcp.CALIBRATION_FREQUENCY)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=mcp.TARGET_LIST,
        help="Subset of targets to run. Available: DA ID ID3 ID1 IMB",
    )
    parser.add_argument(
        "--method-spec",
        action="append",
        default=[],
        help=(
            "Optional method selection using regime-free folder naming, for example "
            "'cp_symmetric_lightgbm/regression' or 'cqr_lightgbm/quantile'. "
            "Repeat to run multiple combinations."
        ),
    )
    parser.add_argument(
        "--regime-groups",
        nargs="+",
        default=[],
        help="Optional subset of regime subfolders under data/regimes for MCP.",
    )
    parser.add_argument(
        "--scale-names",
        nargs="+",
        default=[],
        help="Optional subset of scale subfolders under data/scales for LACP.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=None,
        help="Learning rates used for ACI.",
    )
    parser.add_argument(
        "--enbpi-num-batches",
        type=int,
        default=enbpi.NUM_BATCHES,
        help="Number of bootstrap base learners used by EnbPI.",
    )
    parser.add_argument(
        "--enbpi-model-name",
        default=enbpi.ENBPI_MODEL_NAME,
        help="Single point base learner model name used by EnbPI.",
    )
    parser.add_argument(
        "--enbpi-objective-name",
        default=enbpi.ENBPI_OBJECTIVE_NAME,
        help="Single point base learner objective/loss name used by EnbPI.",
    )
    parser.add_argument(
        "--enbpi-symmetry",
        nargs="+",
        choices=("symmetric", "asymmetric", "both"),
        default=["symmetric", "asymmetric"],
        help="EnbPI residual calibration variant(s) to run.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory for regime-aware method folders.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove each selected pipeline output directory before running it.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Run selected pipelines without writing the combined root summary.",
    )
    return parser.parse_args()


def _normalize_pipeline_name(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "mondrian": "mcp",
        "mondrian_conformal_prediction": "mcp",
        "macp": "mcp",
        "local_adaptive": "lacp",
        "local_adaptive_conformal_prediction": "lacp",
        "adaptive": "aci",
        "adaptive_conformal_inference": "aci",
        "ensemble_batch_prediction_intervals": "enbpi",
        "ensemble_batch_prediction_interval": "enbpi",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PIPELINE_LIST:
        raise ValueError(
            f"Unsupported pipeline {value!r}. Expected one of {list(PIPELINE_LIST)}."
        )
    return normalized


def _normalize_pipeline_names(pipeline_names: Iterable[object] | None) -> list[str]:
    if pipeline_names is None:
        pipeline_names = PIPELINE_LIST

    normalized: list[str] = []
    seen: set[str] = set()
    for value in pipeline_names:
        pipeline_name = _normalize_pipeline_name(value)
        if pipeline_name in seen:
            continue
        normalized.append(pipeline_name)
        seen.add(pipeline_name)

    if not normalized:
        raise ValueError("At least one regime-aware pipeline must be selected.")
    return normalized


def _normalize_context_names(values: Iterable[object] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    return normalized or None


def _learning_rate_folder_name(learning_rate: float) -> str:
    return format(float(learning_rate), "g")


def _learning_rate_context_names(
    learning_rates: Iterable[float] | None,
) -> tuple[str, ...] | None:
    if learning_rates is None:
        return None
    normalized = tuple(_learning_rate_folder_name(value) for value in learning_rates)
    return normalized or None


def _normalize_enbpi_symmetry(symmetry_values: object) -> bool | None:
    if symmetry_values is None:
        return None
    if isinstance(symmetry_values, bool):
        return symmetry_values
    if isinstance(symmetry_values, str):
        values = [symmetry_values]
    else:
        values = list(symmetry_values)

    normalized = {
        str(value).strip().lower().replace("-", "_")
        for value in values
        if str(value).strip()
    }
    if not normalized or "both" in normalized:
        return None
    if normalized == {"symmetric"}:
        return True
    if normalized == {"asymmetric"}:
        return False
    if normalized == {"symmetric", "asymmetric"}:
        return None
    raise ValueError(
        "Unsupported EnbPI symmetry values. Expected symmetric, asymmetric, or both; "
        f"got {sorted(normalized)}."
    )


def _load_summary_frame(
    *,
    summary_path: Path,
    pipeline_name: str,
    context_name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    frame["pipeline"] = pipeline_name
    frame["pipeline_variant"] = context_name
    frame["source_summary_path"] = str(summary_path)

    if pipeline_name == "mcp" and "regime_group_name" not in frame.columns:
        frame["regime_group_name"] = context_name
    if pipeline_name == "lacp" and "scale_name" not in frame.columns:
        frame["scale_name"] = context_name
    if pipeline_name == "aci" and "learning_rate" not in frame.columns:
        try:
            frame["learning_rate"] = float(context_name)
        except ValueError:
            frame["learning_rate"] = context_name
    if pipeline_name == "enbpi":
        frame["pipeline_variant"] = "enbpi"

    return frame


def _discover_context_summaries(
    method_root: Path,
    *,
    context_names: Iterable[str] | None = None,
) -> list[tuple[str, Path]]:
    if not method_root.exists():
        return []

    requested = set(_normalize_context_names(context_names) or ())
    discovered: list[tuple[str, Path]] = []
    for context_dir in sorted(path for path in method_root.iterdir() if path.is_dir()):
        if requested and context_dir.name not in requested:
            continue
        summary_path = context_dir / SUMMARY_FILENAME
        if summary_path.exists():
            discovered.append((context_dir.name, summary_path))
    return discovered


def combine_regime_aware_summaries(
    *,
    results_root: Path = RESULTS_ROOT,
    pipeline_names: Iterable[object] | None = PIPELINE_LIST,
    regime_group_names: Iterable[str] | None = None,
    scale_names: Iterable[str] | None = None,
    learning_rates: Iterable[float] | None = None,
    include_enbpi: bool = True,
) -> pd.DataFrame:
    results_root = results_root.resolve()
    normalized_pipelines = _normalize_pipeline_names(pipeline_names)
    combined_frames: list[pd.DataFrame] = []

    context_filters = {
        "mcp": _normalize_context_names(regime_group_names),
        "lacp": _normalize_context_names(scale_names),
        "aci": _learning_rate_context_names(learning_rates),
    }

    for pipeline_name in ("mcp", "lacp", "aci"):
        if pipeline_name not in normalized_pipelines:
            continue
        method_root = results_root / pipeline_name
        for context_name, summary_path in _discover_context_summaries(
            method_root,
            context_names=context_filters[pipeline_name],
        ):
            combined_frames.append(
                _load_summary_frame(
                    summary_path=summary_path,
                    pipeline_name=pipeline_name,
                    context_name=context_name,
                )
            )

    if include_enbpi and "enbpi" in normalized_pipelines:
        enbpi_summary_path = results_root / "enbpi" / SUMMARY_FILENAME
        if enbpi_summary_path.exists():
            combined_frames.append(
                _load_summary_frame(
                    summary_path=enbpi_summary_path,
                    pipeline_name="enbpi",
                    context_name="enbpi",
                )
            )

    if not combined_frames:
        selected_roots = ", ".join(
            str(results_root / pipeline_name) for pipeline_name in normalized_pipelines
        )
        raise FileNotFoundError(
            f"No {SUMMARY_FILENAME!r} files were found under the selected roots: "
            f"{selected_roots}."
        )

    combined = pd.concat(combined_frames, ignore_index=True)
    ranked = rfe._rank_results(combined)
    ranked.to_csv(results_root / SUMMARY_FILENAME, index=False)
    return ranked


def _run_mcp(
    *,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    target_list: list[str] | tuple[str, ...],
    results_root: Path,
    method_model_loss_list: Iterable[object] | None,
    regime_group_names: Iterable[str] | None,
    clean_output: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return mcp.run_mondrian_conformal_prediction(
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
        target_list=target_list,
        results_root=results_root / "mcp",
        method_model_loss_list=method_model_loss_list,
        regime_group_names=regime_group_names,
        clean_output=clean_output,
    )


def _run_lacp(
    *,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    target_list: list[str] | tuple[str, ...],
    results_root: Path,
    method_model_loss_list: Iterable[object] | None,
    scale_names: Iterable[str] | None,
    clean_output: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return lacp.run_local_adaptive_conformal_prediction(
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
        target_list=target_list,
        results_root=results_root / "lacp",
        method_model_loss_list=method_model_loss_list,
        scale_names=scale_names,
        clean_output=clean_output,
    )


def _run_aci(
    *,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    target_list: list[str] | tuple[str, ...],
    results_root: Path,
    method_model_loss_list: Iterable[object] | None,
    learning_rates: Iterable[float] | None,
    clean_output: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return aci.run_adaptive_conformal_inference(
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
        target_list=target_list,
        results_root=results_root / "aci",
        learning_rates=learning_rates,
        method_model_loss_list=method_model_loss_list,
        clean_output=clean_output,
    )


def _run_enbpi(
    *,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    target_list: list[str] | tuple[str, ...],
    results_root: Path,
    model_name: str,
    objective_name: str,
    symmetry: object,
    num_batches: int,
    clean_output: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return enbpi.run_ensemble_batch_prediction_intervals(
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
        target_list=target_list,
        results_root=results_root / "enbpi",
        model_name=model_name,
        objective_name=objective_name,
        num_batches=num_batches,
        symmetric=_normalize_enbpi_symmetry(symmetry),
        clean_output=clean_output,
    )


def run_regime_aware_experiment(
    *,
    pipeline_names: Iterable[object] | None = PIPELINE_LIST,
    calibration_range_weeks: int = mcp.CALIBRATION_RANGE,
    calibration_frequency_weeks: int = mcp.CALIBRATION_FREQUENCY,
    target_list: list[str] | tuple[str, ...] = mcp.TARGET_LIST,
    results_root: Path = RESULTS_ROOT,
    method_model_loss_list: Iterable[object] | None = mcp.METHOD_MODEL_LOSS_LIST,
    regime_group_names: Iterable[str] | None = mcp.REGIME_GROUP_LIST,
    scale_names: Iterable[str] | None = lacp.SCALE_NAME_LIST,
    learning_rates: Iterable[float] | None = aci.LEARNING_RATES,
    enbpi_model_name: str = enbpi.ENBPI_MODEL_NAME,
    enbpi_objective_name: str = enbpi.ENBPI_OBJECTIVE_NAME,
    enbpi_symmetry: Iterable[str] | str | None = ("symmetric", "asymmetric"),
    enbpi_num_batches: int = enbpi.NUM_BATCHES,
    clean_output: bool = False,
    combine_results: bool = True,
) -> RegimeAwareExperimentResult:
    results_root = results_root.resolve()
    selected_pipelines = _normalize_pipeline_names(pipeline_names)
    pipeline_results: dict[str, pd.DataFrame] = {}
    pipeline_failures: dict[str, pd.DataFrame] = {}
    skipped_pipelines: list[str] = []

    for pipeline_name in selected_pipelines:
        if pipeline_name not in IMPLEMENTED_PIPELINES:
            skipped_pipelines.append(pipeline_name)
            rfe._log(
                f"Skipping {pipeline_name.upper()}: no regime-aware "
                f"{pipeline_name.upper()} pipeline is implemented yet."
            )
            continue

        if pipeline_name == "mcp":
            ranked_results, failures_df = _run_mcp(
                calibration_range_weeks=calibration_range_weeks,
                calibration_frequency_weeks=calibration_frequency_weeks,
                target_list=target_list,
                results_root=results_root,
                method_model_loss_list=method_model_loss_list,
                regime_group_names=regime_group_names,
                clean_output=clean_output,
            )
        elif pipeline_name == "lacp":
            ranked_results, failures_df = _run_lacp(
                calibration_range_weeks=calibration_range_weeks,
                calibration_frequency_weeks=calibration_frequency_weeks,
                target_list=target_list,
                results_root=results_root,
                method_model_loss_list=method_model_loss_list,
                scale_names=scale_names,
                clean_output=clean_output,
            )
        elif pipeline_name == "aci":
            ranked_results, failures_df = _run_aci(
                calibration_range_weeks=calibration_range_weeks,
                calibration_frequency_weeks=calibration_frequency_weeks,
                target_list=target_list,
                results_root=results_root,
                method_model_loss_list=method_model_loss_list,
                learning_rates=learning_rates,
                clean_output=clean_output,
            )
        elif pipeline_name == "enbpi":
            ranked_results, failures_df = _run_enbpi(
                calibration_range_weeks=calibration_range_weeks,
                calibration_frequency_weeks=calibration_frequency_weeks,
                target_list=target_list,
                results_root=results_root,
                model_name=enbpi_model_name,
                objective_name=enbpi_objective_name,
                symmetry=enbpi_symmetry,
                num_batches=enbpi_num_batches,
                clean_output=clean_output,
            )
        else:
            raise AssertionError(f"Unhandled pipeline {pipeline_name!r}.")

        pipeline_results[pipeline_name] = ranked_results
        pipeline_failures[pipeline_name] = failures_df

    pipelines_to_combine = [
        pipeline_name
        for pipeline_name in selected_pipelines
        if pipeline_name not in skipped_pipelines
    ]
    if combine_results and pipelines_to_combine:
        ranked = combine_regime_aware_summaries(
            results_root=results_root,
            pipeline_names=pipelines_to_combine,
            regime_group_names=regime_group_names,
            scale_names=scale_names,
            learning_rates=learning_rates,
            include_enbpi=True,
        )
    else:
        ranked = pd.DataFrame()

    return RegimeAwareExperimentResult(
        ranked_results=ranked,
        pipeline_results=pipeline_results,
        pipeline_failures=pipeline_failures,
        skipped_pipelines=tuple(skipped_pipelines),
    )


def main() -> int:
    args = _parse_args()
    method_model_loss_list = args.method_spec or mcp.METHOD_MODEL_LOSS_LIST
    regime_group_names = args.regime_groups or None
    scale_names = args.scale_names or None
    learning_rates = args.learning_rates
    if args.run_pipelines and learning_rates is None:
        learning_rates = aci.LEARNING_RATES

    if args.run_pipelines:
        result = run_regime_aware_experiment(
            pipeline_names=args.pipelines,
            calibration_range_weeks=args.calibration_range,
            calibration_frequency_weeks=args.calibration_frequency,
            target_list=args.targets,
            results_root=args.results_root,
            method_model_loss_list=method_model_loss_list,
            regime_group_names=regime_group_names,
            scale_names=scale_names,
            learning_rates=learning_rates,
            enbpi_model_name=args.enbpi_model_name,
            enbpi_objective_name=args.enbpi_objective_name,
            enbpi_symmetry=args.enbpi_symmetry,
            enbpi_num_batches=args.enbpi_num_batches,
            clean_output=bool(args.clean_output),
            combine_results=not bool(args.skip_combine),
        )
        ranked = result.ranked_results
        failure_count = sum(len(frame) for frame in result.pipeline_failures.values())
        rfe._log(f"Combined rows : {len(ranked):,}")
        rfe._log(f"Pipeline failures : {failure_count:,}")
        if result.skipped_pipelines:
            rfe._log(f"Skipped pipelines : {list(result.skipped_pipelines)}")
        return 0 if failure_count == 0 else 1

    ranked = combine_regime_aware_summaries(
        results_root=args.results_root,
        pipeline_names=args.pipelines,
        regime_group_names=regime_group_names,
        scale_names=scale_names,
        learning_rates=learning_rates,
    )
    rfe._log(f"Combined rows : {len(ranked):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
