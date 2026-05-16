import argparse
import pandas as pd
import os
import sys
import json
import logging
from typing import Dict, Any

# Ensure 'src' is in the Python path so 'forecasting' module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forecasting.baseline_models import DailyNaiveForecaster, WeeklyNaiveForecaster, CombinedSeasonalForecaster
from forecasting.data_cleaning import load_and_clean_data
from forecasting.fill_missing_data import fill_missing_linear
from forecasting.export import export_forecast

from forecasting.lstm_model import LSTMForecaster

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_REGISTRY = {
    'daily_naive': DailyNaiveForecaster,
    'weekly_naive': WeeklyNaiveForecaster,
    'combined_seasonal': CombinedSeasonalForecaster,
    'lstm': LSTMForecaster
}

def calculate_and_log_error(model, actuals_df: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> Dict[str, float]:
    """
    Since we don't have stored past predictions easily accessible here, we recreate
    the *past predictions* by making the model predict from (solve_time - horizon_hours)
    and then compare it to the actuals for that window.
    """
    # Define the testing window: we want to evaluate the last `horizon` hours before solve_time.
    past_solve_time = solve_time - pd.Timedelta(hours=horizon)
    history_for_past_pred = actuals_df[actuals_df.index < past_solve_time]
    
    if history_for_past_pred.empty:
        logging.warning("Not enough history to compute error.")
        return {'mae': float('nan'), 'rmse': float('nan')}
        
    # Model generates predictions for the testing window
    past_predictions = model.predict(history_for_past_pred, past_solve_time, horizon=horizon)
    
    # We get the actuals in that window. `actuals_df` contains everything up to solve_time.
    # Convert 'heat_demand_W' to MW for fair comparison since predict returns MW.
    actuals_window = actuals_df[(actuals_df.index >= past_solve_time) & (actuals_df.index < solve_time)].copy()
    actuals_window_mw = actuals_window['heat_demand_W'] / 1e6
    
    error_metrics = model.calculate_error(actuals_window_mw, past_predictions)
    logging.info(f"Model Error metrics (evaluated over last {horizon} hours): {error_metrics}")
    
    return error_metrics

def main():
    parser = argparse.ArgumentParser(description="Hourly Inference Pipeline for Forecasting")
    parser.add_argument('--model', type=str, choices=list(MODEL_REGISTRY.keys()), default='combined_seasonal',
                        help='Which forecasting model to use. Defaults to combined_seasonal.')
    
    # Path relative to standard execution context root (datascience folder)
    default_input = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'forecasting', 'raw', 'raw_data_measured_demand.csv'))
    parser.add_argument('--input_file', type=str, default=default_input,
                        help='Path to the actual heat demand values CSV.')
    
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'forecasting'))
    parser.add_argument('--output_dir', type=str, default=default_output,
                        help='Directory to output the parquet forecast files.')
    parser.add_argument('--solve_time', type=str, default=None,
                        help='Solve time for the forecast (ISO 8601 string). Defaults to current UTC hour.')
    parser.add_argument('--horizon', type=int, default=35,
                        help='Forecast horizon in hours. Defaults to 35.')
    parser.add_argument('--log_metrics', action='store_true',
                        help='If set, calculates and saves model error metrics to a JSON file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found at {args.input_file}")
        return
        
    logging.info(f"Loading data from {args.input_file}")
    raw_df = load_and_clean_data(args.input_file)
    
    # Clean up dataset: Fill missing limits so we don't break baseline.
    cleaned_df = fill_missing_linear(raw_df)

    # 1. Determine Solve Time
    if args.solve_time:
        solve_time = pd.to_datetime(args.solve_time, utc=True)
    else:
        # Default to the most recent data point available + 1 hour (simulating real-time data end)
        solve_time = cleaned_df.index.max() + pd.Timedelta(hours=1)
        
    logging.info(f"Running inference pipeline for solve_time: {solve_time} (Horizon: {args.horizon}h)")
    
    # Filter out anything after the solve time (simulating realtime behavior)
    history_df = cleaned_df[cleaned_df.index < solve_time]

    # 3. Instantiate Active Model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class()
    logging.info(f"Instantiated model: {args.model}")

    # 4. Calculate Current Model Error (Optional based on flag)
    if args.log_metrics:
        error_metrics = calculate_and_log_error(model, history_df, solve_time, horizon=args.horizon)
        
        # Save error metrics to help tracking
        os.makedirs(args.output_dir, exist_ok=True)
        error_log_path = os.path.join(args.output_dir, f"error_metrics_{solve_time.strftime('%Y%m%dT%H%M%SZ')}.json")
        with open(error_log_path, 'w') as f:
            json.dump({'solve_time': str(solve_time), 'model': args.model, 'metrics': error_metrics}, f)

    # 5. Predict
    logging.info(f"Predicting next {args.horizon} hours...")
    predictions = model.predict(history_df, solve_time, horizon=args.horizon)

    # 6. Export Parquet
    exported_path = export_forecast(predictions, solve_time, output_dir=args.output_dir)
    logging.info(f"Exported prediction successfully to {exported_path}")

if __name__ == "__main__":
    main()