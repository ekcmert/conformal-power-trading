# Regime-Aware Probabilistic Price Forecasts for Short-Term Power Trading

Research code for the LMU MSc Data Science thesis:

**Regime-Aware Probabilistic Price Forecasts for Short-Term Power Trading: From Distributional Uncertainty to Calibrated Market Actions**

**Author:** Mert Ekici

## Overview

This repository implements an experimental forecasting and trading pipeline for short-term German power markets. The code builds feature and target datasets from Montel Energy Quantified data, trains point and interval base learners, calibrates probabilistic forecasts with conformal prediction methods, and evaluates whether calibrated forecast intervals can be translated into market actions.

The main research focus is whether regime-aware uncertainty calibration improves probabilistic price forecasts and downstream trading decisions compared with regime-free calibration.

## How the Code Works

The project is organized as a sequence of runnable Python scripts:

1. **Data ingestion and preparation**  
   `run_data_prep.py` optionally fetches Montel Energy Quantified curves, preprocesses raw curve files, engineers features, and writes final aligned datasets to `data/final/X.csv` and `data/final/y.csv`.

2. **Base model training**  
   `run_exp1.py` trains and benchmarks point and interval learners. The model set includes linear baselines and tree/boosting models from scikit-learn, LightGBM, CatBoost, and XGBoost. Outputs are written under `data/predictions/` and `results/`.

3. **Auxiliary artifact generation**  
   `run_generate.py` creates residual files for conformal calibration, regime labels under `data/regimes/`, and scale series under `data/scales/`.

4. **Regime-free calibration baseline**  
   `run_exp2.py` applies rolling-window conformal calibration without regimes. This serves as the main non-regime-aware benchmark.

5. **Regime-aware experiments**  
   `run_exp3.py` compares regime-aware calibration pipelines, including Mondrian conformal prediction, local adaptive conformal prediction, adaptive conformal inference, and EnbPI-style bootstrap intervals.

6. **Market-action backtesting**  
   `run_bt.py` and `run_bt_batch.py` convert calibrated intervals into interval-band trading signals and evaluate PnL-oriented scenarios for day-ahead, intraday, ID3, ID1, and imbalance-related market actions.

## What Is Experimented

The experiments cover three linked questions:

- **Forecasting models:** which point and quantile/interval learners provide useful base forecasts for short-term power price targets.
- **Calibration methods:** how split conformal prediction, conformalized quantile regression, Mondrian/regime-aware calibration, local scale adaptation, adaptive alpha updates, and EnbPI compare in coverage, sharpness, and stability.
- **Trading translation:** whether calibrated probabilistic intervals can be mapped into robust market actions through configurable position sizing, entry/exit bands, and signal-shaping functions.

Regime discovery is explored with heuristic labels and clustering-based regimes such as k-means, k-medoids, agglomerative/divisive clustering, spectral clustering, Gaussian mixture models, hidden Markov models, and PCA variants.

## Data and API Access

Data loading uses the `energyquantified` Python package through the local wrapper in `energy_quantified/`. A working **Montel Energy Quantified subscription API token** is required to fetch data.

The token can be provided through an environment variable:

```powershell
$env:ENERGYQUANTIFIED_API_KEY="your-token-here"
```

or stored locally in `.env` as either:

```text
ENERGYQUANTIFIED_API_KEY=your-token-here
```

or:

```text
EQ_API_KEY=your-token-here
```

The local `.env` is used by the code but should not be shared or committed.

## Running the Pipeline

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Typical execution order:

```powershell
python run_data_prep.py
python run_exp1.py
python run_generate.py
python run_exp2.py
python run_exp3.py
python run_bt.py
```

Most run scripts contain `RUN_*` switches and configuration constants at the top of the file. Adjust these to select specific stages, targets, calibration settings, regime groups, models, or backtest paths.

## Repository Structure

- `energy_quantified/` - Montel Energy Quantified client wrapper and curve definitions.
- `data_prep/` and `feature_engineer/` - raw data preprocessing, feature engineering, and final dataset creation.
- `base_models/` - point and interval model registries, runners, metrics, and tuning grids.
- `conformal_prediction/` - reusable conformal prediction and conformalized quantile regression logic.
- `regime_discovery/` - heuristic and clustering-based regime assignment.
- `generate/` - residual, regime, and scale artifact generation.
- `exp1/`, `exp2/`, `exp3/` - base learner, regime-free, and regime-aware experiment pipelines.
- `backtest/` and `bt/` - interval-to-action backtesting and strategy parameter optimization.
- `tests/` - unit tests for conformal calibration, regime generation, artifact generation, and backtesting utilities.

## Notes

Generated data, predictions, logs, and results can be large and are stored under `data/` and `results/`. Some runner scripts contain local paths for selected experiment outputs; update these paths when moving the project to another machine. Zipped "results" and "data" folders can be accessed from the link: drive.google.com/drive/folders/1oeNHnu2h55hxNt-WRL4jpZxRzJncQLLv
