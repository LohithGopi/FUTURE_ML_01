import pandas as pd
import numpy as np


def create_time_features(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    """
    Generates rich calendar and cyclical time features from datetime index/column.
    """
    df = df.copy()
    dt = df[date_col].dt

    df["day_of_week"] = dt.dayofweek
    df["day_of_month"] = dt.day
    df["day_of_year"] = dt.dayofyear
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["year"] = dt.year

    # Business calendar flags
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"] = dt.is_month_end.astype(int)

    # Cyclical representations (sine / cosine transformations)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Linear time index to model secular growth/trend
    df["time_step"] = (df[date_col] - df[date_col].min()).dt.days

    return df


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "total_sales",
    lags: list = [1, 7, 14, 21, 28, 30],
    rolling_windows: list = [7, 14, 30],
) -> pd.DataFrame:
    """
    Generates autoregressive lag features and rolling window statistics.
    CRITICAL: Uses shifted target values to prevent future data leakage into the features.
    """
    df = df.copy()

    # Shifted target series (yesterday's values)
    shifted_target = df[target_col].shift(1)

    # Lag features
    for lag in lags:
        df[f"sales_lag_{lag}"] = df[target_col].shift(lag)

    # Rolling window aggregations
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = shifted_target.rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = shifted_target.rolling(window=window, min_periods=1).std().fillna(0)
        df[f"rolling_min_{window}"] = shifted_target.rolling(window=window, min_periods=1).min()
        df[f"rolling_max_{window}"] = shifted_target.rolling(window=window, min_periods=1).max()

    # Exponential Moving Average (7 and 30 day)
    df["sales_ema_7"] = shifted_target.ewm(span=7, adjust=False).mean()
    df["sales_ema_30"] = shifted_target.ewm(span=30, adjust=False).mean()

    return df


def prepare_modeling_dataset(
    daily_df: pd.DataFrame,
    target_col: str = "total_sales",
    test_days: int = 90,
) -> tuple:
    """
    Generates all features, drops warmup NaN rows, and applies a strict
    chronological train-test split (test_days for out-of-time evaluation).
    """
    # 1. Add calendar & cyclical features
    featured_df = create_time_features(daily_df, date_col="order_date")

    # 2. Add lags & rolling statistics
    featured_df = create_lag_and_rolling_features(featured_df, target_col=target_col)

    # 3. Drop warmup rows where 30-day lag/rolling windows are NaN
    max_warmup = 30
    clean_df = featured_df.iloc[max_warmup:].reset_index(drop=True)

    # Define feature list (exclude non-feature columns)
    exclude_cols = [
        "order_date",
        target_col,
        "total_orders",
        "total_quantity",
        "avg_discount",
        "total_profit",
    ]
    feature_cols = [col for col in clean_df.columns if col not in exclude_cols]

    # Chronological Split
    split_idx = len(clean_df) - test_days
    train_df = clean_df.iloc[:split_idx].copy()
    test_df = clean_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return clean_df, feature_cols, train_df, test_df, X_train, y_train, X_test, y_test
