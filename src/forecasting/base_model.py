from abc import ABC, abstractmethod
import pandas as pd

class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    Enforces a consistent scikit-learn style API for modularity.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Train the model.
        
        Parameters:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable (optional depending on model)
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions.
        
        Parameters:
        X (pd.DataFrame): Features to predict on
        
        Returns:
        pd.Series: Forecasted values
        """
        pass
