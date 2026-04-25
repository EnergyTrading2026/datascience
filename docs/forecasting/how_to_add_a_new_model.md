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
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions based on the feature set X"""
        # Add prediction logic here
        predictions = self.model.predict(X)
        
        # Ensure it returns a pandas Series matching the index of X!
        return pd.Series(predictions, index=X.index)
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
- Be passed the test data to generate predictions.
- Have its metrics (**MAE, RMSE, MAPE, R²**) calculated dynamically and displayed side-by-side in the comparison dataframes and bar charts.
- Render on all time-series plot comparisons (1-month, 1-week, and 1-day views).

### Note on Training (fit)
If your model requires training, make sure you configure your data splitting and call `model.fit(df_train, y_train)` inside the notebook prediction loop prior to calling `.predict()`.