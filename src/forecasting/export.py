import pandas as pd
import os

def export_forecast(predictions: pd.Series, solve_time: pd.Timestamp, output_dir: str = '/shared/forecast/') -> str:
    """
    Exports the forecast to a parquet file following the strict forecast contract.

    Parameters:
    predictions (pd.Series or pd.DataFrame): The forecasted values. Must have or be convertable
                                             to a single column named 'demand_mw_th'.
    solve_time (pd.Timestamp): The solve time of the optimization cycle.
    output_dir (str): Directory to save the exported forecast. Defaults to '/shared/forecast/'
                      (the optimization daemon's input directory).
    
    Returns:
    str: The path to the saved parquet file.
    """
    # 1. Format payload
    if isinstance(predictions, pd.Series):
        df = predictions.to_frame(name='demand_mw_th')
    else:
        df = predictions.copy()
        if 'demand_mw_th' not in df.columns:
            raise ValueError("DataFrame must contain 'demand_mw_th' column.")
        # Restrict to only the required column
        df = df[['demand_mw_th']]

    # 2. Index Requirements: DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # 3. Index Requirements: Timezone-aware (normalize to UTC) 
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    # 4. Convert solve_time safely to localized UTC for filename formatting
    if solve_time.tz is None:
        st = solve_time.tz_localize('UTC')
    else:
        st = solve_time.tz_convert('UTC')

    # 5. Index Requirements: Sorted ascending
    df = df.sort_index()

    # 6. Column Requirements: No missing values allowed
    if df['demand_mw_th'].isna().any():
        raise ValueError("Forecast contains missing values, which is strictly prohibited by the contract.")

    # Determine filepath
    os.makedirs(output_dir, exist_ok=True)
    # Hyphens (not colons) in the time portion so the filename is valid on
    # Windows and matches the optimization daemon's FORECAST_FILENAME_RE.
    filename = st.strftime('%Y-%m-%dT%H-%M-%SZ.parquet')
    filepath = os.path.join(output_dir, filename)

    # Atomic write: parquet to a tmp sibling, then rename. Keeps the daemon
    # from ever opening a partially-written file when it scans concurrently.
    tmp_path = filepath + '.tmp'
    df.to_parquet(tmp_path, engine='pyarrow')
    os.replace(tmp_path, filepath)

    return filepath
