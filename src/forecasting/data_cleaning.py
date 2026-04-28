import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data, ensures correct column parsing and hourly frequency.
    """
    df = pd.read_csv(file_path)
    
    # Rename columns for easier access
    df.rename(columns={'Time Point': 'timestamp', 'Measured Heat Demand[W]': 'heat_demand_W'}, inplace=True)
    
    # Parse timestamp and make it timezone aware (or convert to UTC, but keeping original is fine)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Drop duplicates if any
    if df['timestamp'].duplicated().any():
        print(f"Dropping {df['timestamp'].duplicated().sum()} duplicate timestamps.")
        df = df.drop_duplicates(subset=['timestamp'])
        
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Ensure correct hourly frequency
    print("Resampling to hourly frequency to ensure consistent timestamps...")
    df = df.resample('1h').asfreq()
    
    return df

def analyze_missing_data(df: pd.DataFrame, target_col: str = 'heat_demand_W') -> dict:
    """
    Identifies and quantifies missing data and contiguous blocks.
    """
    total_records = len(df)
    missing_count = df[target_col].isna().sum()
    missing_percentage = (missing_count / total_records) * 100
    
    print(f"Total records: {total_records}")
    print(f"Missing count: {missing_count} ({missing_percentage:.2f}%)")
    
    # Detect contiguous blocks
    is_missing = df[target_col].isna()
    # Create groups for each contiguous block
    blocks = (is_missing != is_missing.shift()).cumsum()
    # Filter only the missing blocks
    missing_blocks = df[is_missing].groupby(blocks).size()
    
    block_analysis = {
        'total_missing': int(missing_count),
        'missing_percentage': float(missing_percentage),
        'num_missing_blocks': int(len(missing_blocks)),
        'max_block_size': int(missing_blocks.max()) if not missing_blocks.empty else 0,
        'mean_block_size': float(missing_blocks.mean()) if not missing_blocks.empty else 0.0
    }
    
    print(f"Number of contiguous missing blocks: {block_analysis['num_missing_blocks']}")
    print(f"Largest missing block size: {block_analysis['max_block_size']} hours")
    print(f"Average missing block size: {block_analysis['mean_block_size']:.2f} hours")
    
    return block_analysis

if __name__ == "__main__":

    # Resolve path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../data/forecasting/raw_data_measured_demand.csv")
    
    try:
        df_cleaned = load_and_clean_data(file_path)
        stats = analyze_missing_data(df_cleaned)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please run from src/forecasting or adjust path.")
