"""Unsupervised anomaly detection (Isolation Forest) for wasteful or faulty consumption."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def anomaly_features(df):
    x = pd.DataFrame({
        "total_kwh": df["total_kwh"],
        "dev_roll24": df["total_kwh"] - df["roll_24"],
        "dev_lag24": df["total_kwh"] - df["lag_24"],
        "hour_sin": np.sin(2 * np.pi * df["hour"] / 24),
        "hour_cos": np.cos(2 * np.pi * df["hour"] / 24),
    })
    if "occupied" in df.columns:
        x["occupied"] = df["occupied"]
    return x


def train_detector(train, save=True):
    pipe = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=200, contamination=config.ANOMALY_CONTAMINATION,
                        random_state=config.SEED),
    )
    pipe.fit(anomaly_features(train))
    if save:
        config.ensure_dirs()
        joblib.dump(pipe, config.MODELS_DIR / "anomaly_model.joblib")
    return pipe


def detect(pipe, df):
    x = anomaly_features(df)
    out = df[["timestamp", "total_kwh"]].copy()
    out["flag"] = (pipe.predict(x) == -1).astype(int)
    out["score"] = -pipe.decision_function(x)
    out["excess_kwh"] = (df["total_kwh"] - df["roll_24"]).round(3)
    if "is_anomaly" in df.columns:
        out["is_anomaly"] = df["is_anomaly"].values
    return out


def evaluate(flags):
    """Precision / recall / F1 against injected labels (only possible for synthetic data)."""
    if "is_anomaly" not in flags.columns:
        return {"n_flagged": int(flags["flag"].sum())}
    y, p = flags["is_anomaly"], flags["flag"]
    return {"n_flagged": int(p.sum()), "n_true": int(y.sum()),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0))}
