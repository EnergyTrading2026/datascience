from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional
import numpy as np

class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    Enforces a consistent API for modularity while integrating
    with the optimization pipeline.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Train the model.
        
        Parameters:
        X (pd.DataFrame): Training features / historical data
        y (pd.Series, optional): Target variable
        """
        pass
        
    @abstractmethod
    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        """
        Generate forecasts for a specific future horizon.
        
        Parameters:
        history (pd.DataFrame): Historical data up to the solve_time.
        solve_time (pd.Timestamp): The starting timestamp for the forecast.
        horizon (int): The number of future steps (hours) to forecast. Defaults to 35.
        
        Returns:
        pd.Series: Forecasted values starting from solve_time for the given horizon.
        """
        pass

    def calculate_error(self, actuals: pd.Series, predictions: pd.Series) -> Dict[str, float]:
        """
        Compute the current model error based on the latest actuals vs past predictions.
        
        Parameters:
        actuals (pd.Series): The actual observed values for a specific time window.
        predictions (pd.Series): The previously forecasted values for that same window.
        
        Returns:
        Dict[str, float]: A dictionary containing error metrics (MAE, RMSE, etc.)
        """
        # Align by index to ensure correct time-based comparison
        df = pd.DataFrame({'actual': actuals, 'pred': predictions}).dropna()
        
        if df.empty:
            return {'mae': np.nan, 'rmse': np.nan}
            
        y_true = df['actual']
        y_pred = df['pred']
        
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        
        return {'mae': mae, 'rmse': rmse}
