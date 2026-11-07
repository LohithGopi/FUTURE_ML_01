from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class BaselineForecaster:
    """
    Baseline benchmark using lag-7 (same day last week) or rolling average.
    """
    def __init__(self, lag_col: str = "sales_lag_7"):
        self.lag_col = lag_col

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.lag_col in X.columns:
            preds = X[self.lag_col].fillna(0).values
            return preds
        if "rolling_mean_7" in X.columns:
            return X["rolling_mean_7"].fillna(0).values
        return np.full(len(X), fill_value=X.mean().values[0])


def get_models(random_state: int = 42) -> dict:
    """
    Returns a dictionary of candidate models.
    """
    return {
        "Baseline (Lag-7 Persistence)": BaselineForecaster(lag_col="sales_lag_7"),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=random_state))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1
        )
    }


def train_and_evaluate_all(models: dict, X_train, y_train, X_test, y_test) -> tuple:
    """
    Fits all models on X_train, y_train and evaluates them on X_test, y_test.
    Returns evaluation summary dataframe and dictionary of fitted models & predictions.
    """
    from src.evaluate import calculate_metrics, compare_models

    results = []
    predictions = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, a_min=0, a_max=None)

        metrics = calculate_metrics(y_test, preds, model_name=name)
        results.append(metrics)
        predictions[name] = preds
        fitted_models[name] = model

    summary_df = compare_models(results)
    return summary_df, fitted_models, predictions


def generate_future_forecast(
    model,
    historical_daily_df: pd.DataFrame,
    feature_cols: list,
    forecast_days: int = 30,
    service_level_z: float = 1.65,  # 90% confidence for safety stock
) -> pd.DataFrame:
    """
    Recursively forecasts future daily sales for the next N days.
    Also computes safety stock bounds (upper and lower confidence bands)
    essential for inventory stocking decisions.
    """
    from src.feature_engineering import create_time_features, create_lag_and_rolling_features

    # Use only clean core columns from history
    core_cols = ["order_date", "total_sales", "total_orders", "total_quantity", "avg_discount", "total_profit"]
    df_extended = historical_daily_df[[c for c in core_cols if c in historical_daily_df.columns]].copy()
    last_date = df_extended["order_date"].max()

    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_preds = []

    for f_date in future_dates:
        new_row = pd.DataFrame([{
            "order_date": f_date,
            "total_sales": np.nan,
            "total_orders": 0,
            "total_quantity": 0,
            "avg_discount": 0.0,
            "total_profit": 0.0,
        }])
        df_extended = pd.concat([df_extended, new_row], ignore_index=True)

        # Re-compute features dynamically
        feat_df = create_time_features(df_extended, date_col="order_date")
        feat_df = create_lag_and_rolling_features(feat_df, target_col="total_sales")

        # Extract features for the latest date using exact training feature_cols
        current_features = feat_df.iloc[[-1]]
        X_curr = current_features[feature_cols].fillna(0)

        # Predict next day
        pred_val = model.predict(X_curr)[0]
        pred_val = max(0.0, float(pred_val))
        future_preds.append(pred_val)

        # Update the predicted value back into df_extended for future lag propagation
        df_extended.loc[df_extended.index[-1], "total_sales"] = pred_val

    # Estimate standard error of residuals from historical baseline for confidence bands
    hist_std = historical_daily_df["total_sales"].tail(60).std()
    safety_stock_buffer = service_level_z * (hist_std * 0.4)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_sales": np.round(future_preds, 2),
        "safety_stock_upper": np.round([p + safety_stock_buffer for p in future_preds], 2),
        "conservative_lower": np.round([max(0.0, p - safety_stock_buffer) for p in future_preds], 2),
    })

    return forecast_df

