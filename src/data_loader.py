import os
import pandas as pd
import numpy as np


def load_raw_superstore(file_path: str) -> pd.DataFrame:
    """
    Loads raw Superstore sales data from CSV with encoding fallbacks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")

    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transaction records:
    - Normalizes column names (lowercase, stripped, underscores)
    - Converts order date to datetime format
    - Drops duplicates and handles missing values
    - Sorts records chronologically
    """
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]

    # Convert order_date and ship_date
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # Drop nulls in essential columns
    df = df.dropna(subset=["order_date", "sales"])

    # Ensure positive sales and quantities
    df = df[df["sales"] > 0]
    if "quantity" in df.columns:
        df = df[df["quantity"] > 0]

    # Sort by order_date
    df = df.sort_values(by="order_date").reset_index(drop=True)
    return df


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactional sales to a continuous daily frequency.
    Ensures missing days are imputed with 0 to prevent gaps in time-series models.
    """
    daily = (
        df.groupby("order_date")
        .agg(
            total_sales=("sales", "sum"),
            total_orders=("order_id", "nunique"),
            total_quantity=("quantity", "sum") if "quantity" in df.columns else ("sales", "count"),
            avg_discount=("discount", "mean") if "discount" in df.columns else ("sales", lambda x: 0.0),
            total_profit=("profit", "sum") if "profit" in df.columns else ("sales", lambda x: 0.0),
        )
        .reset_index()
    )

    # Build continuous calendar date range
    min_date = daily["order_date"].min()
    max_date = daily["order_date"].max()
    full_idx = pd.date_range(start=min_date, end=max_date, freq="D", name="order_date")

    daily = (
        daily.set_index("order_date")
        .reindex(full_idx)
        .fillna({
            "total_sales": 0.0,
            "total_orders": 0,
            "total_quantity": 0,
            "avg_discount": 0.0,
            "total_profit": 0.0,
        })
        .reset_index()
    )

    return daily
