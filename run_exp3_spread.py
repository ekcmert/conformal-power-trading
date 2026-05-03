from __future__ import annotations

from base_models.point_estimation.common import REPO_ROOT
from exp3 import mondrian_conformal_prediction as mcp
from exp3.regime_aware_experiment import PIPELINE_LIST, run_regime_aware_experiment
from exp3.regime_eval import run_regime_evaluation


RUN_REGIME_EVALUATION = False
RUN_REGIME_AWARE_EXPERIMENT = True
RUN_COMBINE_RESULTS = True
CLEAN_OUTPUT = False

RESULTS_ROOT = REPO_ROOT / "results" / "regime_aware_spreads"

PIPELINES = tuple(pipeline for pipeline in PIPELINE_LIST if pipeline != "enbpi")
# TARGET_LIST = mcp.TARGET_LIST
TARGET_LIST = ["DA", "ID", "ID3", "ID1", "IMB"]
CALIBRATION_RANGE = mcp.CALIBRATION_RANGE
CALIBRATION_FREQUENCY = mcp.CALIBRATION_FREQUENCY

METHOD_MODEL_LOSS_LIST = mcp.METHOD_MODEL_LOSS_LIST
# REGIME_GROUP_LIST = ("kmeans_pca_6", "kmedoids_6", "hmm_8", "kmeans_6", "kmedoids_pca_6", "agglomerative_pca_6", "divisive_6")
# SCALE_NAME_LIST = ("wind_fc", "res_load_fc", "cons_cli", "wind_cli", "res_load_cli", "cons_temp_fc")
# LEARNING_RATES = aci.LEARNING_RATES

REGIME_GROUP_LIST = ("divisive_6",)
SCALE_NAME_LIST = ("cons_cli",)
LEARNING_RATES = (0.1,)

PLOT_COLUMN_PAIRS = [
    (
        "DE Residual Load MWh/h 15min Forecast__mean",
        "DE Price Spot EUR/MWh H Forecast__mean",
    ),
    (
        "DE Wind Power Production MWh/h 15min Forecast__mean",
        "DE Solar Photovoltaic Production MWh/h 15min Forecast__mean",
    ),
    (
        "DE Consumption MWh/h 15min Forecast__mean",
        "DE Residual Load MWh/h 15min Forecast__mean",
    ),
]
REFERENCE_RESIDUAL_MODEL_PATH = (
    r"C:\Users\mert.ekici\Desktop\CPT\data\residuals\point_estimation\lightgbm\regression\da_res.csv"
)
TARGET_PRICE_COLUMN_NAME = "DE Price Spot EUR/MWh EPEX H Actual"
REGIME_GLOB = "*"


def main() -> int:
    if RUN_REGIME_EVALUATION:
        result = run_regime_evaluation(
            plot_column_pairs=PLOT_COLUMN_PAIRS,
            reference_residual_model_path=REFERENCE_RESIDUAL_MODEL_PATH,
            target_price_column_name=TARGET_PRICE_COLUMN_NAME,
            regime_glob=REGIME_GLOB,
            skip_plots=False,
            show_progress=True,
        )

        print(f"Regime groups evaluated  : {result.regime_groups_evaluated}")
        print(f"Residual series evaluated: {result.residual_series_evaluated}")
        print(f"Results root             : {result.results_root}")
        print(f"Summary CSV              : {result.summary_path}")

    if RUN_REGIME_AWARE_EXPERIMENT:
        experiment_result = run_regime_aware_experiment(
            pipeline_names=PIPELINES,
            calibration_range_weeks=CALIBRATION_RANGE,
            calibration_frequency_weeks=CALIBRATION_FREQUENCY,
            target_list=TARGET_LIST,
            results_root=RESULTS_ROOT,
            method_model_loss_list=METHOD_MODEL_LOSS_LIST,
            regime_group_names=REGIME_GROUP_LIST,
            scale_names=SCALE_NAME_LIST,
            learning_rates=LEARNING_RATES,
            clean_output=CLEAN_OUTPUT,
            combine_results=RUN_COMBINE_RESULTS,
        )

        failure_count = sum(len(frame) for frame in experiment_result.pipeline_failures.values())
        print(f"Combined regime-aware spread rows: {len(experiment_result.ranked_results):,}")
        print(f"Pipeline failures                : {failure_count:,}")
        print(f"Results root                     : {RESULTS_ROOT}")
        if experiment_result.skipped_pipelines:
            print(f"Skipped pipelines                : {list(experiment_result.skipped_pipelines)}")
        return 0 if failure_count == 0 else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
