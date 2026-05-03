from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from xgboost import XGBRegressor

from base_models.tuning_grids import INTERVAL_TUNING_PARAM_GRIDS, expand_param_grid

from .common import validate_quantile_pair
from .estimators import MultiQuantileEstimator, SeparateQuantileEstimator, TreeEnsembleQuantileIntervalEstimator


RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentSpec:
    model_name: str
    objective_name: str | None
    display_name: str
    builder: Callable[[], object]
    params: dict[str, object]
    params_summary: str


@dataclass(frozen=True)
class ExperimentGroup:
    model_name: str
    objective_name: str | None
    display_name: str
    candidates: tuple[ExperimentSpec, ...]

    @property
    def default_spec(self) -> ExperimentSpec:
        return self.candidates[0]


@dataclass(frozen=True)
class ExperimentDefinition:
    model_name: str
    objective_name: str | None
    display_name: str
    default_params: dict[str, object]
    builder_factory: Callable[[dict[str, object]], object]
    summary_builder: Callable[[dict[str, object]], dict[str, object]]


def _format_params_summary(params: dict[str, object]) -> str:
    return repr(dict(sorted(params.items(), key=lambda item: item[0])))


def _candidate_params_for_definition(
    definition: ExperimentDefinition,
    *,
    tuning_active: bool,
) -> list[dict[str, object]]:
    candidate_params = [dict(definition.default_params)]
    if tuning_active:
        candidate_params = expand_param_grid(
            definition.default_params,
            INTERVAL_TUNING_PARAM_GRIDS.get((definition.model_name, definition.objective_name)),
        )

    deduplicated_candidates: list[dict[str, object]] = []
    seen_summaries: set[str] = set()
    for params in candidate_params:
        summary = _format_params_summary(params)
        if summary in seen_summaries:
            continue
        seen_summaries.add(summary)
        deduplicated_candidates.append(params)
    return deduplicated_candidates


def _build_group(
    definition: ExperimentDefinition,
    *,
    tuning_active: bool,
) -> ExperimentGroup:
    candidates = []
    for params in _candidate_params_for_definition(definition, tuning_active=tuning_active):
        summary_params = definition.summary_builder(dict(params))
        candidates.append(
            ExperimentSpec(
                model_name=definition.model_name,
                objective_name=definition.objective_name,
                display_name=definition.display_name,
                builder=lambda p=dict(params), factory=definition.builder_factory: factory(dict(p)),
                params=dict(params),
                params_summary=_format_params_summary(summary_params),
            )
        )

    return ExperimentGroup(
        model_name=definition.model_name,
        objective_name=definition.objective_name,
        display_name=definition.display_name,
        candidates=tuple(candidates),
    )


def get_experiment_spec_groups(
    lower_quantile: float,
    upper_quantile: float,
    *,
    tuning_active: bool = False,
) -> list[ExperimentGroup]:
    validate_quantile_pair(lower_quantile, upper_quantile)

    def add_quantiles(params: dict[str, object]) -> dict[str, object]:
        return {**params, "quantiles": [lower_quantile, upper_quantile]}

    quantile_regressor_base = {
        "alpha": 1e-4,
        "fit_intercept": True,
        "solver": "highs",
    }
    definitions: list[ExperimentDefinition] = [
        ExperimentDefinition(
            model_name="quantile_regressor",
            objective_name="pinball_loss",
            display_name="QuantileRegressor | pinball loss",
            default_params=quantile_regressor_base,
            builder_factory=lambda params: SeparateQuantileEstimator(
                estimator_builder=lambda quantile, p=dict(params): QuantileRegressor(
                    **p,
                    quantile=quantile,
                ),
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            ),
            summary_builder=lambda params: {**add_quantiles(params), "standardization": "global"},
        )
    ]

    histgb_base = {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "random_state": RANDOM_STATE,
        "loss": "quantile",
    }
    definitions.append(
        ExperimentDefinition(
            model_name="hist_gradient_boosting",
            objective_name="quantile",
            display_name="HistGradientBoosting | quantile",
            default_params=histgb_base,
            builder_factory=lambda params: SeparateQuantileEstimator(
                estimator_builder=lambda quantile, p=dict(params): HistGradientBoostingRegressor(
                    **p,
                    quantile=quantile,
                ),
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            ),
            summary_builder=add_quantiles,
        )
    )

    lgbm_base = {
        "importance_type": "gain",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "objective": "quantile",
    }
    definitions.append(
        ExperimentDefinition(
            model_name="lightgbm",
            objective_name="quantile",
            display_name="LightGBM | quantile",
            default_params=lgbm_base,
            builder_factory=lambda params: SeparateQuantileEstimator(
                estimator_builder=lambda quantile, p=dict(params): LGBMRegressor(
                    **p,
                    alpha=quantile,
                ),
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            ),
            summary_builder=add_quantiles,
        )
    )

    catboost_base = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": RANDOM_STATE,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
    }
    definitions.append(
        ExperimentDefinition(
            model_name="catboost",
            objective_name="quantile",
            display_name="CatBoost | Quantile",
            default_params={**catboost_base, "loss_function": "Quantile"},
            builder_factory=lambda params: SeparateQuantileEstimator(
                estimator_builder=lambda quantile, p=dict(params): CatBoostRegressor(
                    **{key: value for key, value in p.items() if key != "loss_function"},
                    loss_function=f"{p['loss_function']}:alpha={quantile}",
                ),
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            ),
            summary_builder=add_quantiles,
        )
    )
    definitions.append(
        ExperimentDefinition(
            model_name="catboost",
            objective_name="multiquantile",
            display_name="CatBoost | MultiQuantile",
            default_params={**catboost_base, "loss_function": "MultiQuantile"},
            builder_factory=lambda params: MultiQuantileEstimator(
                estimator_builder=lambda quantiles, p=dict(params): CatBoostRegressor(
                    **{key: value for key, value in p.items() if key != "loss_function"},
                    loss_function=f"{p['loss_function']}:alpha={quantiles[0]},{quantiles[-1]}",
                ),
                quantiles=(lower_quantile, upper_quantile),
            ),
            summary_builder=add_quantiles,
        )
    )

    xgboost_base = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
        "objective": "reg:quantileerror",
    }
    definitions.append(
        ExperimentDefinition(
            model_name="xgboost",
            objective_name="reg_quantileerror",
            display_name="XGBoost | reg:quantileerror",
            default_params=xgboost_base,
            builder_factory=lambda params: MultiQuantileEstimator(
                estimator_builder=lambda quantiles, p=dict(params): XGBRegressor(
                    **p,
                    quantile_alpha=list(quantiles),
                ),
                quantiles=(lower_quantile, upper_quantile),
            ),
            summary_builder=add_quantiles,
        )
    )

    random_forest_base = {
        "n_estimators": 120,
        "min_samples_leaf": 2,
        "criterion": "squared_error",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    definitions.append(
        ExperimentDefinition(
            model_name="quantile_random_forest",
            objective_name="squared_error",
            display_name="Quantile Random Forest | squared_error",
            default_params=random_forest_base,
            builder_factory=lambda params: TreeEnsembleQuantileIntervalEstimator(
                ensemble_builder=lambda p=dict(params): RandomForestRegressor(**p),
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            ),
            summary_builder=lambda params: {
                **add_quantiles(params),
                "interval_construction": "tree-wise empirical quantiles",
            },
        )
    )

    extra_trees_base = {
        "n_estimators": 160,
        "min_samples_leaf": 2,
        "criterion": "squared_error",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    definitions.append(
        ExperimentDefinition(
            model_name="quantile_extra_trees",
            objective_name="squared_error",
            display_name="Quantile ExtraTrees Forest | squared_error",
            default_params=extra_trees_base,
            builder_factory=lambda params: TreeEnsembleQuantileIntervalEstimator(
                ensemble_builder=lambda p=dict(params): ExtraTreesRegressor(**p),
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            ),
            summary_builder=lambda params: {
                **add_quantiles(params),
                "interval_construction": "tree-wise empirical quantiles",
            },
        )
    )

    return [
        _build_group(definition, tuning_active=tuning_active)
        for definition in definitions
    ]


def get_experiment_specs(
    lower_quantile: float,
    upper_quantile: float,
) -> list[ExperimentSpec]:
    return [
        group.default_spec
        for group in get_experiment_spec_groups(
            lower_quantile,
            upper_quantile,
            tuning_active=False,
        )
    ]
