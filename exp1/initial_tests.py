from __future__ import annotations

import shutil
from math import sqrt
from pathlib import Path

import matplotlib
import pandas as pd
from lightgbm import LGBMRegressor
from plotly import graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
X_PATH = REPO_ROOT / "data" / "final" / "X.csv"
Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"
RESULTS_DIR = REPO_ROOT / "results" / "test_models"

TARGETS = ["DAID", "DAID3", "DAID1", "DAIMB"]
TEST_YEAR = 2026

BASE_LGBM_PARAMS = {
    "importance_type": "gain",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

OBJECTIVE_CONFIGS = {
    "regression": {
        "objective": "regression",
    },
    "regression_l1": {
        "objective": "regression_l1",
    },
    "huber": {
        "objective": "huber",
        "alpha": 0.9,
    },
    "fair": {
        "objective": "fair",
        "fair_c": 1.0,
    },
}


def load_final_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(X_PATH, index_col=0, parse_dates=True)
    y = pd.read_csv(Y_PATH, index_col=0, parse_dates=True)
    X, y = X.align(y, join="inner", axis=0)

    missing_targets = [target for target in TARGETS if target not in y.columns]
    if missing_targets:
        raise KeyError(f"Missing spread targets in y.csv: {missing_targets}")

    return X.sort_index(), y[TARGETS].sort_index()


def reset_results_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def build_model(objective_params: dict[str, object]) -> LGBMRegressor:
    return LGBMRegressor(**BASE_LGBM_PARAMS, **objective_params)


def plot_predictions(
    target_name: str,
    objective_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    output_dir: Path,
    metrics: dict[str, float],
) -> Path:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(y_true.index, y_true.values, label="True", linewidth=1.6, color="#1d3557")
    ax.plot(y_pred.index, y_pred.values, label="Predicted", linewidth=1.2, color="#e76f51")
    ax.set_title(
        f"{target_name} | {objective_name} | Single Run {TEST_YEAR}\n"
        f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(target_name)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = output_dir / f"{target_name}_predictions.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importances(
    target_name: str,
    objective_name: str,
    importance: pd.Series,
    output_dir: Path,
    top_n: int = 50,
) -> Path:
    top_importance = importance.sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(14, max(8, int(top_n * 0.3))))
    ax.barh(top_importance.index, top_importance.values, color="#2a9d8f")
    ax.set_title(f"{target_name} | {objective_name} | Top {top_n} LightGBM Importances")
    ax.set_xlabel("Gain importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    output_path = output_dir / f"{target_name}_feature_importances.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_combined_plotly_predictions(
    targets: list[str] | tuple[str, ...],
    objective_name: str,
    output_dir: Path,
) -> Path:
    color_map = {
        "DAID": "#1d3557",
        "DAID3": "#2a9d8f",
        "DAID1": "#e76f51",
        "DAIMB": "#8d5fd3",
    }

    fig = go.Figure()
    for target in targets:
        prediction_frame = pd.read_csv(output_dir / f"{target}_predictions.csv", parse_dates=["timestamp"])
        base_color = color_map.get(target, "#264653")

        fig.add_trace(
            go.Scatter(
                x=prediction_frame["timestamp"],
                y=prediction_frame["y_true"],
                mode="lines",
                name=f"{target} true",
                line={"color": base_color, "width": 2},
                legendgroup=target,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prediction_frame["timestamp"],
                y=prediction_frame["y_pred"],
                mode="lines",
                name=f"{target} pred",
                line={"color": base_color, "width": 1.5, "dash": "dash"},
                legendgroup=target,
            )
        )

    fig.update_layout(
        title=f"{objective_name} | Single Run {TEST_YEAR} Spread Predictions vs True Values",
        xaxis_title="Date",
        yaxis_title="Spread",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Series",
    )

    output_path = output_dir / "all_targets_predictions.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def run_target_experiment(
    X: pd.DataFrame,
    y_target: pd.Series,
    objective_name: str,
    objective_params: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    train_mask = X.index.year < TEST_YEAR
    test_mask = X.index.year == TEST_YEAR

    X_train = X.loc[train_mask].copy()
    y_train = y_target.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_test = y_target.loc[test_mask].copy()

    if X_train.empty or X_test.empty:
        raise RuntimeError(f"Empty train/test split for target {y_target.name}.")

    model = build_model(objective_params)
    model.fit(X_train, y_train)

    test_predictions = pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="prediction",
    )
    feature_importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X_train.columns,
        name="importance",
    ).sort_values(ascending=False)

    metrics = compute_metrics(y_test, test_predictions)
    prediction_plot_path = plot_predictions(
        y_target.name,
        objective_name,
        y_test,
        test_predictions,
        output_dir,
        metrics,
    )
    importance_plot_path = plot_feature_importances(
        y_target.name,
        objective_name,
        feature_importance,
        output_dir,
    )

    prediction_frame = pd.DataFrame(
        {
            "timestamp": y_test.index,
            "y_true": y_test.values,
            "y_pred": test_predictions.values,
        }
    )
    prediction_frame.to_csv(output_dir / f"{y_target.name}_predictions.csv", index=False)
    feature_importance.to_csv(output_dir / f"{y_target.name}_feature_importances.csv", header=True)

    return {
        "objective": objective_name,
        "target": y_target.name,
        **metrics,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "prediction_plot": prediction_plot_path,
        "importance_plot": importance_plot_path,
        "lgbm_params": str({**BASE_LGBM_PARAMS, **objective_params}),
    }


def run_objective(
    X: pd.DataFrame,
    y: pd.DataFrame,
    objective_name: str,
    objective_params: dict[str, object],
) -> list[dict[str, object]]:
    objective_dir = RESULTS_DIR / objective_name
    objective_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for target in TARGETS:
        result = run_target_experiment(X, y[target], objective_name, objective_params, objective_dir)
        results.append(result)
        print(
            f"{objective_name} | {target}: MAE={result['mae']:.4f}, "
            f"RMSE={result['rmse']:.4f}, R2={result['r2']:.4f}"
        )

    metrics_path = objective_dir / "metrics_summary.csv"
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    combined_plot_path = save_combined_plotly_predictions(TARGETS, objective_name, objective_dir)
    print(f"{objective_name}: combined Plotly HTML saved to {combined_plot_path}")
    print(f"{objective_name}: metrics saved to {metrics_path}")
    return results


def main() -> None:
    reset_results_directory(RESULTS_DIR)
    X, y = load_final_datasets()

    all_results: list[dict[str, object]] = []
    for objective_name, objective_params in OBJECTIVE_CONFIGS.items():
        all_results.extend(run_objective(X, y, objective_name, objective_params))

    comparison_path = RESULTS_DIR / "objective_comparison.csv"
    pd.DataFrame(all_results).to_csv(comparison_path, index=False)
    print(f"Objective comparison saved to {comparison_path}")
    print(f"Saved outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
