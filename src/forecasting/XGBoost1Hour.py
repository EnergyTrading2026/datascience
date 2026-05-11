import pandas as pd
from forecasting.base_model import BaseForecaster
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

#XGBoost requires Lag-Features
def create_lags(data):
    df = data.copy()
    for lag in range(1, 24 + 1):
        df[f"lag_{lag}"] = df["value"].shift(lag)
    #Drop first rows which contain NaN due to shifting
    df.dropna(inplace=True)
    return df

#XGBoost for 1h Forecast
class XGBoost(BaseForecaster):
    def __init__(self):

        model = XGBRegressor(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
        )

        self.model = model
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Train the model on the feature set X and target y"""
        #Create Lag Features
        X = create_lags(X)
        #Training
        self.model.fit(X,y)
        return self
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions based on the feature set X"""
        #Create Lag Features
        X = create_lags(X)
        #Predict
        predictions = self.model.predict(X)
        
        # Ensure it returns a pandas Series matching the index of X!
        return pd.Series(predictions, index=X.index)