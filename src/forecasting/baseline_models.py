import pandas as pd
import numpy as np
from forecasting.base_model import BaseForecaster

def seasonal_naive_daily(y: pd.Series) -> pd.Series:
    """
    Seasonal Naïve (Daily) baseline.
    Forecast: y_hat_t = y_{t-24}
    """
    if isinstance(y.index, pd.DatetimeIndex):
        # Shift by exact time delta to handle gaps in data
        shifted = y.shift(freq='24h')
        return shifted.reindex(y.index)
    return y.shift(24)

def seasonal_naive_weekly(y: pd.Series) -> pd.Series:
    """
    Seasonal Naïve (Weekly) baseline.
    Forecast: y_hat_t = y_{t-168}
    """
    if isinstance(y.index, pd.DatetimeIndex):
        shifted = y.shift(freq='168h')
        return shifted.reindex(y.index)
    return y.shift(168)

def combined_seasonal_baseline(y: pd.Series) -> pd.Series:
    """
    Combined Seasonal Baseline.
    Forecast: average of daily and weekly:
    y_hat_t = 0.5 * (y_{t-24} + y_{t-168})
    """
    daily = seasonal_naive_daily(y)
    weekly = seasonal_naive_weekly(y)
    
    # Handle beginning of the series where 168 is not available but 24 is
    # by falling back to daily if weekly is NaN, but normally average them.
    # The requirement says average or weighted combination.
    # Let's compute average ignoring nans (which means mean) or explicitly:
    df = pd.DataFrame({'daily': daily, 'weekly': weekly})
    return df.mean(axis=1)

def evaluate_baselines(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Evaluate the baseline predictions against true values using MAE, RMSE, MAPE, and R2.
    """
    mask = ~y_true.isna() & ~y_pred.isna()
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p)**2))
    
    # Add MAPE and R2
    mape = np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, 1e-10, y_t))) * 100
    r2 = 1 - (np.sum((y_t - y_p)**2) / np.sum((y_t - np.mean(y_t))**2))
    
    return {'mae': mae, 'rmse': rmse, 'mape_pct': mape, 'r2': r2}


class DailyNaiveForecaster(BaseForecaster):
    """
    Object-oriented wrapper for the Daily Naive Baseline.
    Implements the BaseForecaster interface.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
        
    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        future_index = pd.date_range(start=solve_time, periods=horizon, freq='h')
        preds = []
        
        # Convert to MW
        history_mw = history['heat_demand_W'] / 1e6
        
        for t in future_index:
            delta_hours = int((t - solve_time).total_seconds() // 3600)
            hist_time = solve_time - pd.Timedelta(hours=24 - (delta_hours % 24))
            
            if hist_time in history_mw.index:
                preds.append(history_mw.loc[hist_time])
            else:
                preds.append(float(history_mw.mean()) if not history_mw.empty else 0.0)
                
        return pd.Series(preds, index=future_index, name='demand_mw_th')

class WeeklyNaiveForecaster(BaseForecaster):
    """
    Object-oriented wrapper for the Weekly Naive Baseline.
    Implements the BaseForecaster interface.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
        
    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        future_index = pd.date_range(start=solve_time, periods=horizon, freq='h')
        preds = []
        
        # Convert to MW
        history_mw = history['heat_demand_W'] / 1e6
        
        for t in future_index:
            delta_hours = int((t - solve_time).total_seconds() // 3600)
            hist_time = solve_time - pd.Timedelta(hours=168 - (delta_hours % 168))
            
            if hist_time in history_mw.index:
                preds.append(history_mw.loc[hist_time])
            else:
                preds.append(float(history_mw.mean()) if not history_mw.empty else 0.0)
                
        return pd.Series(preds, index=future_index, name='demand_mw_th')

class CombinedSeasonalForecaster(BaseForecaster):
    """
    Object-oriented wrapper for the Combined Seasonal Baseline.
    Implements the BaseForecaster interface.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
        
    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        daily = DailyNaiveForecaster().predict(history, solve_time, horizon)
        weekly = WeeklyNaiveForecaster().predict(history, solve_time, horizon)
        
        return (daily + weekly) / 2.0
