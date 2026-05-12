# How to Add a New Forecasting Model

Our forecasting pipeline uses a modular, object-oriented architecture to make adding, comparing, and evaluating new models as simple as possible.

## Step 1: Create the Model Class

All forecasting models must inherit from the `BaseForecaster` abstract base class located in `src/forecasting/base_model.py`. This ensures a standardized `fit/predict` API compatible with Scikit-learn patterns.

Create your new model class in a relevant file under `src/forecasting/` (e.g., `src/forecasting/ml_models.py`) and implement the required `fit` and `predict` methods:

```python
import pandas as pd
from forecasting.base_model import BaseForecaster

class MyNewForecaster(BaseForecaster):
    def __init__(self, my_params=None):
        # Initialize model and hyper-parameters here
        self.model = ...
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Train the model on the feature set X and target y"""
        # Add training logic here (e.g., self.model.fit(X, y))
        return self
        
    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        """Generate forecasts for a specific future horizon."""
        # Create the exact future datetime index expected by the optimizer
        future_index = pd.date_range(start=solve_time, periods=horizon, freq='h')
        
        # Add prediction logic here based on historical data up to solve_time
        predictions_mw = ... # (Make sure your predictions are converted to Megawatts thermal)
        
        # Ensure it returns a pandas Series matching the future_index named 'demand_mw_th'
        return pd.Series(predictions_mw, index=future_index, name='demand_mw_th')
```

## Step 2: Add to the Evaluation Notebook

To evaluate your new model alongside the baselines, open `notebooks/forecasting/evaluate_baselines.ipynb`.

1. **Import your model** in the first code cell:
   ```python
   from forecasting.ml_models import MyNewForecaster
   ```

2. **Add your model to the dictionary** in the evaluation cell. Find the `models` dictionary and simply append your instantiated class:
   ```python
   models = {
       'Daily Naive': DailyNaiveForecaster(),
       'Weekly Naive': WeeklyNaiveForecaster(),
       'Combined Naive': CombinedSeasonalForecaster(),
       # Add your new model here:
       'My Custom ML Model': MyNewForecaster(my_params=...)
   }
   ```

### What happens automatically?
Once you run the notebook loops, your new model will automatically:
- Receive a dynamically split `history` dataset anchored perfectly before your configured `solve_time`.
- Have its metrics (**MAE, RMSE, MAPE, R²**) calculated dynamically and displayed side-by-side in the comparison dataframes and bar charts.
- Render accurately on the standard 35-hour forecasting horizon plot.

### Note on Training (fit)
If your model requires training, make sure you configure your data splitting and call `model.fit(df_train, y_train)` inside the notebook prediction loop prior to calling `.predict()`.

## Step 3: Add to the Hourly Inference Pipeline

For your new model to be completely accessible to the optimization group via the terminal or scheduler, it needs to be registered.

Open `src/forecasting/run_hourly_forecast.py`. 
1. Import your model at the top of the file:
   ```python
   from forecasting.ml_models import MyNewForecaster
   ```

2. Add it to the `MODEL_REGISTRY` dictionary:
   ```python
   MODEL_REGISTRY = {
       'daily_naive': DailyNaiveForecaster,
       'weekly_naive': WeeklyNaiveForecaster,
       'combined_seasonal': CombinedSeasonalForecaster,
       'my_custom_model': MyNewForecaster  # <--- Added here
   }
   ```

Now you can seamlessly trigger forecasts utilizing your new model directly in the command line!
`python3 src/forecasting/run_hourly_forecast.py --model my_custom_model`