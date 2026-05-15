import os
import pandas as pd
import pytest
from unittest.mock import patch
import sys
import json

# Ensure src is in sys.path to easily import the application code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from forecasting.run_hourly_forecast import main

@pytest.fixture
def mock_history_csv(tmp_path):
    """
    Creates a mock historical data CSV for testing.
    Provides 5 days of fake hourly data (120 hours).
    """
    dates = pd.date_range(start="2026-05-01T00:00:00Z", periods=120, freq='h')
    df = pd.DataFrame({
        'Time Point': dates.strftime('%Y-%m-%dT%H:%M:%S.000000+0000'),
        'Measured Heat Demand[W]': [5000000.0] * 120  # constant 5 MW
    })
    
    file_path = tmp_path / "mock_demand.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_hourly_pipeline_and_parquet_contract(mock_history_csv, tmp_path):
    """
    End-to-End test to ensure the hourly forecast pipeline runs completely
    and the output complies perfectly with 'forecast_contract.md'.
    """
    
    # 1. Arrange: Setup parameters
    output_dir = str(tmp_path / "shared" / "forecast")
    solve_time_str = "2026-05-05T00:00:00Z"
    horizon = 35
    
    # Mock sys.argv to simulate running from the terminal
    test_args = [
        "run_hourly_forecast.py",
        "--model", "daily_naive",
        "--input_file", mock_history_csv,
        "--output_dir", output_dir,
        "--solve_time", solve_time_str,
        "--horizon", str(horizon),
        "--log_metrics"
    ]

    # 2. Act: Run the pipeline
    with patch.object(sys, 'argv', test_args):
        main()

    # 3. Assert - File creation
    assert os.path.exists(output_dir), "Output directory was not created."
    
    expected_parquet = os.path.join(output_dir, "2026-05-05T00-00-00Z.parquet")
    assert os.path.exists(expected_parquet), f"Expected parquet file missing: {expected_parquet}"
    
    expected_json = os.path.join(output_dir, "error_metrics_20260505T000000Z.json")
    assert os.path.exists(expected_json), "Metrics JSON was not generated despite --log_metrics."

    # 4. Assert - Parquet Contract Validation
    df_result = pd.read_parquet(expected_parquet)

    # Column Contract
    assert "demand_mw_th" in df_result.columns, "Missing required column 'demand_mw_th'."
    assert len(df_result.columns) == 1, "There should be exactly one column in the payload."
    assert not df_result["demand_mw_th"].isna().any(), "Missing values found, which is forbidden."
    
    # Values check (5000000 W = 5.0 MW)
    assert (df_result["demand_mw_th"] == 5.0).all(), "Values were not correctly converted to Megawatts."

    # Index Contract
    assert isinstance(df_result.index, pd.DatetimeIndex), "Index must be a DatetimeIndex."
    assert df_result.index.tz is not None, "Index must be timezone-aware."
    assert str(df_result.index.tz) == "UTC", "Index should be UTC."
    assert df_result.index.is_monotonic_increasing, "Index is not mathematically sorted ascending."
    
    # Time Semantics
    assert df_result.index[0] == pd.to_datetime(solve_time_str), "First timestamp does not match solve_time."
    
    # Check spacing / horizon length
    assert len(df_result) == horizon, f"Expected {horizon} forecasted rows, got {len(df_result)}."
    
    # Optional check: Verify JSON structure
    with open(expected_json, 'r') as f:
        metrics_data = json.load(f)
        assert 'metrics' in metrics_data
        assert 'mae' in metrics_data['metrics']
        assert metrics_data['model'] == 'daily_naive'
