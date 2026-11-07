import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Computes business-friendly and statistical accuracy metrics:
    - MAE: Mean dollar error per day
    - RMSE: Root mean squared error penalizing extreme demand misses
    - wMAPE: Weighted Mean Absolute Percentage Error (Retail Industry Standard)
    - R2 Score: Goodness of fit
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Industry standard Weighted MAPE (wMAPE): sum(|y - y_hat|) / sum(y)
    total_actual = np.sum(y_true)
    if total_actual > 0:
        wmape = (np.sum(np.abs(y_true - y_pred)) / total_actual) * 100
    else:
        wmape = np.nan

    # Non-zero safe MAPE
    non_zero_mask = y_true > 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    return {
        "Model": model_name,
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "wMAPE (%)": round(wmape, 2),
        "MAPE (%)": round(mape, 2),
        "R2 Score": round(r2, 4),
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Consolidates metric dictionaries into a sorted leaderboard dataframe.
    """
    df = pd.DataFrame(results)
    df = df.sort_values(by="MAE ($)", ascending=True).reset_index(drop=True)
    return df
