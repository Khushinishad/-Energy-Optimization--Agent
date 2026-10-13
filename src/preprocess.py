"""Feature engineering and time-based train/test split."""
import pandas as pd

from . import config


def build_features(df):
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    d["hour"] = d["timestamp"].dt.hour
    d["dayofweek"] = d["timestamp"].dt.dayofweek
    d["month"] = d["timestamp"].dt.month
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["lag_1"] = d["total_kwh"].shift(1)
    d["lag_24"] = d["total_kwh"].shift(24)
    d["lag_168"] = d["total_kwh"].shift(168)
    d["roll_24"] = d["total_kwh"].shift(1).rolling(24).mean()
    return d.dropna(subset=["lag_1", "lag_24", "lag_168", "roll_24"]).reset_index(drop=True)


def feature_columns(df):
    cols = ["hour", "dayofweek", "month", "is_weekend", "lag_1", "lag_24", "lag_168", "roll_24"]
    for optional in ("temperature", "occupied"):
        if optional in df.columns:
            cols.append(optional)
    return cols


def time_split(df, test_fraction=config.TEST_FRACTION):
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
