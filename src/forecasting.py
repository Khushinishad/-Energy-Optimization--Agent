"""Hourly energy demand forecasting (one-step-ahead) with model comparison."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def score(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def train_forecaster(train, test, cols, save=True):
    y_tr, y_te = train["total_kwh"], test["total_kwh"]
    rows = [{"model": "Naive (same hour yesterday)", **score(y_te, test["lag_24"])}]
    models = {
        "Ridge (baseline)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1,
                                               random_state=config.SEED),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                                random_state=config.SEED),
    }
    fitted, preds = {}, {}
    for name, m in models.items():
        m.fit(train[cols], y_tr)
        preds[name] = m.predict(test[cols])
        fitted[name] = m
        rows.append({"model": name, **score(y_te, preds[name])})
    metrics = pd.DataFrame(rows)
    best = metrics[metrics["model"].isin(models)].sort_values("RMSE").iloc[0]["model"]
    if save:
        config.ensure_dirs()
        joblib.dump({"model": fitted[best], "features": cols, "name": best},
                    config.MODELS_DIR / "forecast_model.joblib")
    return metrics, pd.Series(preds[best], index=test.index, name="predicted_kwh"), best
