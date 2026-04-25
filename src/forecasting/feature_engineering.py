import pandas as pd
import numpy as np
import os

def add_temporal_features(df: pd.DataFrame, datetime_col=None) -> pd.DataFrame:
    """
    Adds temporal features to the dataframe.
    Assumes the dataframe has a DatetimeIndex, or specifies datetime_col.
    """
    df_feat = df.copy()
    
    if datetime_col is not None:
        idx = pd.to_datetime(df_feat[datetime_col])
    else:
        idx = df_feat.index
        
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
        
    # Generate temporal features
    df_feat['hour'] = idx.hour
    df_feat['dayofweek'] = idx.dayofweek
    df_feat['month'] = idx.month
    
    # Encode cyclical features
    # Hour: 0-23
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    
    # Day of week: 0-6
    df_feat['dayofweek_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7)
    df_feat['dayofweek_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7)
    
    # Month: 1-12
    df_feat['month_sin'] = np.sin(2 * np.pi * (df_feat['month'] - 1) / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * (df_feat['month'] - 1) / 12)
    
    return df_feat

def create_train_test_split(df: pd.DataFrame, test_size_months: int = 3):
    """
    Creates a time-based train/test split.
    """
    df_sorted = df.sort_index()
    end_date = df_sorted.index.max()
    split_date = end_date - pd.DateOffset(months=test_size_months)
    
    train_df = df_sorted[df_sorted.index <= split_date]
    test_df = df_sorted[df_sorted.index > split_date]
    
    return train_df, test_df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../../data/forecasting")
    
    # Process both filled datasets
    for filename in ["data_filled_linear.csv", "data_filled_seasonal.csv"]:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        print(f"Processing {filename}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Add features
        df_featured = add_temporal_features(df)
        
        # Save featured dataset
        out_filename = filename.replace(".csv", "_featured.csv")
        out_filepath = os.path.join(data_dir, out_filename)
        df_featured.to_csv(out_filepath)
        print(f"Saved featured data to: {out_filepath}")
        
        # Create train/test split
        train_df, test_df = create_train_test_split(df_featured, test_size_months=3)
        
        # Save splits
        base_name = filename.replace(".csv", "")
        train_path = os.path.join(data_dir, f"{base_name}_train.csv")
        test_path = os.path.join(data_dir, f"{base_name}_test.csv")
        
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        print(f"Saved train split to: {train_path}")
        print(f"Saved test split to: {test_path}")

    print("Feature engineering and splitting completed.")
