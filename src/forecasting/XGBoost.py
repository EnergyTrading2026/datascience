import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


# Load CSV file
df = pd.read_csv('dataHeat.csv')

# Convert the date column to datetime
df['time'] = pd.to_datetime(df['time'])
df = df.set_index("time")

def create_lags(data, target_col, n_lags):
    df = data.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    #Zusätzlich 24h und 168h Lags
    df["lag_24"] = df[target_col].shift(24)
    df["lag_168"] = df[target_col].shift(168)
    return df

df_lagged = create_lags(df, "value", n_lags=12)

#Einträge mit NaN verwerfen
df_lagged.dropna(inplace=True)

X = df_lagged.drop(columns=["value"])
y = df_lagged["value"]

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

print(mean_absolute_percentage_error(y_test, y_pred))
