# Forecasting Data Preparation Pipeline

A unified, configurable pipeline for preparing and processing raw time-series data for heat demand forecasting. This pipeline ensures a robust evaluation foundation by standardizing missing value treatments, feature creation, and model benchmarking setups.

## Pipeline Steps

1. **Data Cleaning (`data_cleaning.py`)**: Loads raw data, monetizes column names, enforces timezone-aware timestamps, and resamples to a strict hourly frequency. Analyzes and logs missing data blocks.
2. **Missing Data Handling (`fill_missing_data.py`)**: Handles data gaps using configurable imputation strategies:
   - *Linear Interpolation:* Standard forward/backward filling.
   - *Seasonal Imputation:* Advanced smoothed rolling continuous seasonal profiles based on hour-of-day and day-of-year.
3. **Feature Engineering (`feature_engineering.py`)**: Computes temporal features (hour, day of week, month) and applies cyclical encodings (sin/cos transformations) to capture temporal continuity.
4. **Train/Test Split (`feature_engineering.py`)**: Performs a chronological split (e.g., reserving the last 3 months for testing) to prevent temporal data leakage.

## Usage

The pipeline can be executed via the command line interface from the project root.

### Default Execution
Runs the full pipeline using the default `raw_data_measured_demand.csv` dataset and exports separated datasets to `datascience/data/forecasting/`:
```bash
python3 datascience/src/forecasting/data_pipeline.py
```

### Modular Execution
You can toggle individual steps on or off and specify custom paths. For example, skipping the cleaning step and only using linear imputation with a 6-month test split:
```bash
python3 datascience/src/forecasting/data_pipeline.py \
  --input custom_data.csv \
  --output custom_export_dir/ \
  --skip-cleaning \
  --missing-strategy linear \
  --test-months 6
```

## Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | Path to custom input CSV | `raw_data_measured_demand.csv` |
| `--output` | Directory for exported results | `datascience/data/forecasting/` |
| `--skip-cleaning` | Bypasses Step 1 (resampling and structural formatting) | `False` |
| `--missing-strategy` | Imputation strategy (`linear`, `seasonal`, `both`, `none`) | `both` |
| `--skip-features` | Bypasses Step 3 (temporal and cyclical features) | `False` |
| `--skip-split` | Bypasses Step 4 (time-based train/test splitting) | `False` |
| `--test-months` | Number of trailing months to reserve for the test set | `3` |
| `--no-export` | Runs pipeline purely in memory without saving to disk | `False` |
