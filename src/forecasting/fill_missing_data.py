import pandas as pd
import numpy as np
import os
from forecasting.data_cleaning import load_and_clean_data

def fill_missing_linear(df: pd.DataFrame, target_col: str = 'heat_demand_W') -> pd.DataFrame:
    """
    Fills missing values using linear interpolation.
    For leading NaNs, we backfill to ensure no missing values remain.
    """
    df_filled = df.copy()
    # Interpolate linearly for gaps
    df_filled[target_col] = df_filled[target_col].interpolate(method='linear', limit_direction='both')
    return df_filled

def fill_missing_seasonal(df: pd.DataFrame, target_col: str = 'heat_demand_W') -> pd.DataFrame:
    """
    Fills missing values using a smoothed rolling continuous seasonal profile.
    Prevents boundary 'jumps' by using a rolling window over the days of the year.
    """
    df_filled = df.copy()
    
    # Extract day of year and hour
    doy = df_filled.index.dayofyear
    hour = df_filled.index.hour
    
    # Create temporary dataframe to calculate profiles
    temp_df = pd.DataFrame({
        target_col: df_filled[target_col],
        'doy': doy,
        'hour': hour
    })
    
    # 1. Calculate the raw mean for every exact day of the year and hour
    raw_profiles = temp_df.groupby(['doy', 'hour'])[target_col].mean().unstack(level='hour')
    
    # 2. Smooth the profiles using a rolling window (e.g., 60 days) to prevent jumps
    # min_periods=1 ensures it works even with sparse data, center=True keeps the peak aligned
    smoothed_profiles = raw_profiles.rolling(window=60, min_periods=1, center=True).mean()
    
    # Stack back to a series with index ['doy', 'hour']
    smoothed_profiles = smoothed_profiles.stack()
    
    # 3. Map the smoothed profiles back to original dataframe index
    # We create a MultiIndex to match the smoothed profiles
    multi_idx = pd.MultiIndex.from_arrays([doy, hour])
    
    # Get values from smoothed profile, and fill missing target rows
    # Use reindex so that indices align for valid map logic
    seasonal_fill_values = multi_idx.map(lambda x: smoothed_profiles.get(x, np.nan))
    
    df_filled[target_col] = df_filled[target_col].fillna(pd.Series(seasonal_fill_values, index=df_filled.index))
    
    # 4. Fallback: Generic hourly profile (if some rolling windows completely lack data)
    if df_filled[target_col].isna().any():
        hourly_means = temp_df.groupby('hour')[target_col].transform('mean')
        df_filled[target_col] = df_filled[target_col].fillna(hourly_means)
    
    # Final fallback just in case
    if df_filled[target_col].isna().any():
        overall_mean = df_filled[target_col].mean()
        df_filled[target_col] = df_filled[target_col].fillna(overall_mean)
        
    return df_filled

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "../../data/demand_history/raw_data_measured_demand.csv")
    output_dir = os.path.join(script_dir, "../../data/forecasting")
    
    # Ensure output directory exists (it should, but just in case)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and cleaning raw data...")
    df = load_and_clean_data(input_file)
    
    print("Applying Linear Interpolation...")
    df_linear = fill_missing_linear(df)
    linear_output_path = os.path.join(output_dir, "data_filled_linear.csv")
    df_linear.to_csv(linear_output_path)
    print(f"Saved linear filled data to: {linear_output_path}")
    
    print("Applying Seasonal Filling (Hour-of-day Average)...")
    df_seasonal = fill_missing_seasonal(df)
    seasonal_output_path = os.path.join(output_dir, "data_filled_seasonal.csv")
    df_seasonal.to_csv(seasonal_output_path)
    print(f"Saved seasonal filled data to: {seasonal_output_path}")
    
    print("Missing data handling completed.")
