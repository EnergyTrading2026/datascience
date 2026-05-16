import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Bidirectional, LSTM, Dense, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from forecasting.base_model import BaseForecaster # Ensure this path is correct

# 1. Custom RevIN Layer
class RevIN(Layer):
    """Reversible Instance Normalization Layer"""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, **kwargs):
        super(RevIN, self).__init__(**kwargs)
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        if self.affine:
            self.gamma = self.add_weight(name="gamma", shape=(self.num_features,), initializer="ones", trainable=True)
            self.beta = self.add_weight(name="beta", shape=(self.num_features,), initializer="zeros", trainable=True)
        super(RevIN, self).build(input_shape)

    def call(self, inputs, mode='norm'):
        if mode == 'norm':
            self.mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
            self.var = tf.math.reduce_variance(inputs, axis=1, keepdims=True)
            self.stdev = tf.sqrt(self.var + self.eps)
            x = (inputs - self.mean) / self.stdev
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == 'denorm':
            x = inputs
            if self.affine:
                x = (x - self.beta) / self.gamma
            if len(x.shape) == 2:
                x = tf.expand_dims(x, axis=1)
                x = x * self.stdev + self.mean
                x = tf.squeeze(x, axis=1)
            else:
                x = x * self.stdev + self.mean
            return x

    def get_config(self):
        config = super(RevIN, self).get_config()
        config.update({"num_features": self.num_features, "eps": self.eps, "affine": self.affine})
        return config


# 2. The Main Pipeline Wrapper
class LSTMForecaster(BaseForecaster):
    def __init__(self, lookback=168, horizon=35, epochs=100, batch_size=64, learning_rate=1e-3):
        self.lookback = lookback
        self.horizon = horizon # Train for a specific max horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.covariate_scaler = MinMaxScaler()
        self.weather_cols = ['Temperature'] # Define exogenous cols here
        self.original_covariate_columns = []
        self.covariate_feature_names = []

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates the sin/cos cyclical features internally."""
        df_out = df.copy()
        df_out['hour_sin'] = np.sin(2 * np.pi * df_out.index.hour / 24)
        df_out['hour_cos'] = np.cos(2 * np.pi * df_out.index.hour / 24)
        df_out['dow_sin'] = np.sin(2 * np.pi * df_out.index.dayofweek / 7)
        df_out['dow_cos'] = np.cos(2 * np.pi * df_out.index.dayofweek / 7)
        return df_out

    def _build_ci_lstm(self, target_shape, covariate_shape):
        """Builds the Keras model. Adjusted Dense layer to output self.horizon steps."""
        target_input = Input(shape=target_shape, name="target_sequence_input")
        covariate_input = Input(shape=covariate_shape, name="future_covariate_input")

        revin_layer = RevIN(num_features=target_shape[-1], affine=True, name="revin_layer")
        x_norm = revin_layer(target_input, mode='norm')

        lstm_1 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(x_norm)
        lstm_2 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(lstm_1)
        res_connection = Add()([lstm_1, lstm_2])
        lstm_3 = Bidirectional(LSTM(32, return_sequences=False, activation='tanh'))(res_connection)

        merged = Concatenate()([lstm_3, covariate_input])
        
        # CHANGED: Output layer matches the multi-step horizon instead of 1
        dense_out = Dense(self.horizon, name="dense_projection")(merged)

        # Denormalize output back to physical scale
        final_output = revin_layer(dense_out, mode='denorm')

        model = Model(inputs=[target_input, covariate_input], outputs=final_output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=Huber(delta=1.0), metrics=['mae'])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on the feature set X and target y"""
        # 1. Convert y from Watts to Megawatts (Pipeline Requirement)
        y_mw = y / 1e6
        
        # 2. Prepare Covariates
        self.original_covariate_columns = X.columns.tolist()
        X_features = self._create_temporal_features(X)
        self.covariate_feature_names = X_features.columns.tolist()

        # Scale weather data if present in X
        available_weather = [col for col in self.weather_cols if col in X_features.columns]
        if available_weather:
            X_features[available_weather] = self.covariate_scaler.fit_transform(X_features[available_weather])
        
        # 3. Create Multi-step Sequences
        X_target_list, X_cov_list, y_list = [], [], []
        
        target_arr = y_mw.values.reshape(-1, 1)
        cov_arr = X_features.values

        for i in range(len(target_arr) - self.lookback - self.horizon + 1):
            # Target History
            X_target_list.append(target_arr[i : i + self.lookback])
            # Last known covariates
            X_cov_list.append(cov_arr[i + self.lookback - 1]) 
            # Future Target Horizon
            y_list.append(target_arr[i + self.lookback : i + self.lookback + self.horizon, 0])

        X_train_t = np.array(X_target_list)
        X_train_c = np.array(X_cov_list)
        y_train = np.array(y_list)

        # 4. Build & Train Model
        target_shape = (self.lookback, 1)
        covariate_shape = (X_train_c.shape[1],)
        self.covariate_dim = X_train_c.shape[1]
        
        self.model = self._build_ci_lstm(target_shape, covariate_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]

        self.model.fit(
            x=[X_train_t, X_train_c],
            y=y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        return self

    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        """Generate forecasts for a specific future horizon."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        # Ensure horizon requested doesn't exceed what the model was compiled for
        pred_horizon = min(horizon, self.horizon)
        future_index = pd.date_range(start=solve_time, periods=horizon, freq='h')

        # 1. Filter history exactly up to the solve time
        hist_cutoff = history[history.index < solve_time].copy()
        
        if len(hist_cutoff) < self.lookback:
            raise ValueError(f"Not enough history. Need {self.lookback} points, got {len(hist_cutoff)}")

        # 2. Extract last 'lookback' steps
        recent_hist = hist_cutoff.iloc[-self.lookback:]

        target_col = 'heat_demand_W'
        if target_col in recent_hist.columns:
            target_seq = recent_hist[target_col].values.reshape(1, self.lookback, 1)
        else:
            target_seq = recent_hist.iloc[:, 0].values.reshape(1, self.lookback, 1)

        # Convert W to MW if the history being passed to predict is still in Watts
        # (Check your pipeline to see if history is already in MW. If yes, remove this line)
        target_seq = target_seq / 1e6

        # 3. Process Covariates for the prediction state (last known hour)
        target_col = 'heat_demand_W'
        covariate_df = recent_hist.drop(columns=[target_col], errors='ignore')
        covariate_df = covariate_df.reindex(columns=self.original_covariate_columns, fill_value=0)

        covariates = self._create_temporal_features(covariate_df)
        if self.covariate_feature_names:
            covariates = covariates.reindex(columns=self.covariate_feature_names, fill_value=0)

        available_weather = [col for col in self.weather_cols if col in covariates.columns]
        if available_weather:
            covariates[available_weather] = self.covariate_scaler.transform(covariates[available_weather])
        
        cov_seq = covariates.iloc[-1].values.reshape(1, -1)

        expected_dim = getattr(self, 'covariate_dim', cov_seq.shape[1])
        if cov_seq.shape[1] != expected_dim:
            if cov_seq.shape[1] > expected_dim:
                cov_seq = cov_seq[:, :expected_dim]
            else:
                padding = np.zeros((cov_seq.shape[0], expected_dim - cov_seq.shape[1]))
                cov_seq = np.concatenate([cov_seq, padding], axis=1)

        # 4. Predict
        preds_mw = self.model.predict([target_seq, cov_seq])[0] # Shape: (self.horizon,)

        # Slice to the requested horizon if the user asked for less than self.horizon
        preds_mw = preds_mw[:pred_horizon]

        # If requested horizon > trained horizon, pad the rest (Naive approach to satisfy pipeline)
        if horizon > self.horizon:
            padding = [preds_mw[-1]] * (horizon - self.horizon)
            preds_mw = np.append(preds_mw, padding)

        return pd.Series(preds_mw, index=future_index, name='demand_mw_th')