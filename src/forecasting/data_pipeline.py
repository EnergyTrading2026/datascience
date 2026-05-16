import os
import argparse
import pandas as pd
from typing import Optional, Union, Tuple, Dict

# Import existing steps
from data_cleaning import load_and_clean_data, analyze_missing_data
from fill_missing_data import fill_missing_linear, fill_missing_seasonal
from feature_engineering import add_temporal_features, create_train_test_split

class DataPreparationPipeline:
    """
    A unified data preparation pipeline that consolidates:
    1. Data Cleaning
    2. Missing Data Handling
    3. Feature Engineering
    4. Train/Test Splitting
    
    Each step can be run independently or sequenced together.
    """

    default_input_path = "../../data/demand_history/raw_data_measured_demand.csv"
    default_export_dir = "../../data/forecasting"
    
    def __init__(self, input_file: Optional[str] = None, output_dir: Optional[str] = None):
        # Setup defaults
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if input_file is None:
            self.input_file = os.path.join(self.script_dir, self.default_input_path)
        else:
            self.input_file = input_file
            
        if output_dir is None:
            self.output_dir = os.path.join(self.script_dir, self.default_export_dir)
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def run_step_1_cleaning(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Loads and cleans data. Uses provided df or reads from input_file."""
        print("\n--- Step 1: Data Cleaning ---")
        if df is not None:
            self.df = df
        else:
            self.df = load_and_clean_data(self.input_file)
            
        analyze_missing_data(self.df)
        return self.df

    def run_step_2_missing_data(self, df: Optional[pd.DataFrame] = None, strategy: str = 'both') -> Dict[str, pd.DataFrame]:
        """
        Fills missing data safely. 
        strategy: 'linear', 'seasonal', or 'both'
        Returns a dictionary of dataframes keyed by strategy name.
        """
        print(f"\n--- Step 2: Missing Data Handling (Strategy: {strategy}) ---")
        df_to_process = df if df is not None else self.df
        if df_to_process is None:
            raise ValueError("No data available to process. Run step 1 first or provide a dataframe.")
            
        results = {}
        if strategy in ['linear', 'both']:
            print("Applying Linear Interpolation...")
            results['linear'] = fill_missing_linear(df_to_process)
            
        if strategy in ['seasonal', 'both']:
            print("Applying Seasonal Filling...")
            results['seasonal'] = fill_missing_seasonal(df_to_process)
            
        return results

    def run_step_3_feature_engineering(self, dfs: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generates temporal features. Accepts a single DataFrame or a dict of DataFrames.
        """
        print("\n--- Step 3: Feature Engineering ---")
        if isinstance(dfs, dict):
            results = {}
            for key, df in dfs.items():
                print(f"Engineering features for '{key}' dataset...")
                results[key] = add_temporal_features(df)
            return results
        else:
            print("Engineering features...")
            return add_temporal_features(dfs)

    def run_step_4_split(self, dfs: Union[pd.DataFrame, Dict[str, pd.DataFrame]], test_size_months: int = 3) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Creates train/test splits. Accepts a single DataFrame or a dict of DataFrames.
        """
        print(f"\n--- Step 4: Train/Test Split ({test_size_months} months test) ---")
        if isinstance(dfs, dict):
            results = {}
            for key, df in dfs.items():
                print(f"Splitting '{key}' dataset...")
                results[key] = create_train_test_split(df, test_size_months)
            return results
        else:
            print("Splitting dataset...")
            return create_train_test_split(dfs, test_size_months)
            
    def export_data(self, dfs: Union[pd.DataFrame, Dict[str, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]], suffix: str = ""):
        """
        Exports processed dataframe(s) to the output directory.
        Properly handles dictionaries, tuples (splits), and direct DataFrames.
        """
        if isinstance(dfs, dict):
            for key, df in dfs.items():
                if isinstance(df, tuple):
                    train_df, test_df = df
                    train_df.to_csv(os.path.join(self.output_dir, f"data_{key}{suffix}_train.csv"))
                    test_df.to_csv(os.path.join(self.output_dir, f"data_{key}{suffix}_test.csv"))
                else:
                    df.to_csv(os.path.join(self.output_dir, f"data_{key}{suffix}.csv"))
        else:
            if isinstance(dfs, tuple):
                train_df, test_df = dfs
                train_df.to_csv(os.path.join(self.output_dir, f"data{suffix}_train.csv"))
                test_df.to_csv(os.path.join(self.output_dir, f"data{suffix}_test.csv"))
            else:
                dfs.to_csv(os.path.join(self.output_dir, f"data{suffix}.csv"))
        print(f"\n[Export] Data successfully exported to: {os.path.abspath(self.output_dir)}")

    def run_full_pipeline(
        self, 
        do_cleaning: bool = True, 
        missing_data_strategy: str = 'both', 
        do_features: bool = True, 
        do_split: bool = True,
        test_size_months: int = 3,
        export: bool = True
    ):
        """
        Runs the full configured pipeline sequentially.
        """
        print(" Starting Data Preparation Pipeline ")
        
        current_data = None
        
        if not os.path.exists(self.input_file):
            print(f"\nError: The input file '{self.input_file}' was not found. Please provide a valid file path.")
            return None

        # Step 1: Clean
        if do_cleaning:
            try:
                current_data = self.run_step_1_cleaning()
            except FileNotFoundError:
                print(f"\nError: The input file '{self.input_file}' was not found. Please provide a valid file path.")
                return None
        else:
            # Load raw directly (e.g., if it's already clean or we're skipping standard cleaning)
            print("\n--- Skipping Step 1 (Loading Data directly) ---")
            try:
                current_data = pd.read_csv(self.input_file, index_col=None)
            except FileNotFoundError:
                print(f"\n❌ Error: The input file '{self.input_file}' was not found. Please provide a valid file path.")
                return None
            
            # Simple fallback if we loaded raw data but skipped cleaning
            if 'Measured Heat Demand[W]' in current_data.columns:
                current_data.rename(columns={'Time Point': 'timestamp', 'Measured Heat Demand[W]': 'heat_demand_W'}, inplace=True)
            if 'timestamp' in current_data.columns:
                current_data['timestamp'] = pd.to_datetime(current_data['timestamp'], utc=True)
                current_data.set_index('timestamp', inplace=True)
            elif not isinstance(current_data.index, pd.DatetimeIndex):
                current_data.index = pd.to_datetime(current_data.index)
                
            self.df = current_data
            
        # Step 2: Fill Missing
        if missing_data_strategy is not None and missing_data_strategy.lower() != 'none':
            current_data = self.run_step_2_missing_data(current_data, strategy=missing_data_strategy)
        else:
            print("\n--- Skipping Step 2 (Missing Data Handling) ---")
            # If no strategy, optionally pass as singular 'none' key to keep structure simple
            current_data = {'unfilled': current_data}
            
        # Step 3: Feature Engineering
        if do_features:
            current_data = self.run_step_3_feature_engineering(current_data)
            if export and not do_split:
                # If we stop here, export featured dataset
                self.export_data(current_data, suffix="_featured")
        else:
            print("\n--- Skipping Step 3 (Feature Engineering) ---")
            if export and not do_split:
                self.export_data(current_data)
                
        # Step 4: Split
        if do_split:
            splits = self.run_step_4_split(current_data, test_size_months=test_size_months)
            if export:
                self.export_data(splits, suffix="_featured" if do_features else "")
            return splits
            
        return current_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Data Preparation Pipeline")
    
    # Input/Output paths
    parser.add_argument("--input", type=str, default=None, help="Path to input raw data CSV (default: uses measured demand)")
    parser.add_argument("--output", type=str, default=None, help="Directory to export results (default: data/forecasting)")
    
    # Toggles for features
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip Step 1: Data Cleaning")
    parser.add_argument("--missing-strategy", type=str, choices=['linear', 'seasonal', 'both', 'none'], default='both', help="Step 2: Missing data strategy (default: both)")
    parser.add_argument("--skip-features", action="store_true", help="Skip Step 3: Feature Engineering")
    parser.add_argument("--skip-split", action="store_true", help="Skip Step 4: Train/Test Split")
    
    # Hyperparameters
    parser.add_argument("--test-months", type=int, default=3, help="Months of data for the test set split")
    parser.add_argument("--no-export", action="store_true", help="Disable saving results to CSV")

    args = parser.parse_args()

    pipeline = DataPreparationPipeline(input_file=args.input, output_dir=args.output)
    
    pipeline.run_full_pipeline(
        do_cleaning=not args.skip_cleaning,
        missing_data_strategy=args.missing_strategy if args.missing_strategy != 'none' else None,
        do_features=not args.skip_features,
        do_split=not args.skip_split,
        test_size_months=args.test_months,
        export=not args.no_export
    )