"""K-Means clustering of daily load profiles to discover usage patterns."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config

NAMES3 = ["Low-usage days", "Moderate-usage days", "High-usage days"]


def cluster_days(df, k=config.N_CLUSTERS):
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    d["hour"] = d["timestamp"].dt.hour
    prof = d.pivot_table(index="date", columns="hour", values="total_kwh", aggfunc="mean").dropna()
    km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(prof.values)
    order = np.argsort(km.cluster_centers_.mean(axis=1))          # low -> high
    names = NAMES3 if k == 3 else [f"Cluster {i + 1}" for i in range(k)]
    rank_name = {int(c): names[r] for r, c in enumerate(order)}
    labels = [rank_name[int(c)] for c in km.labels_]
    days = pd.DataFrame({"date": [str(x) for x in prof.index], "cluster": labels,
                         "daily_kwh": prof.sum(axis=1).round(2).values})
    centers = pd.DataFrame(km.cluster_centers_.T, columns=[rank_name[i] for i in range(k)])
    centers.index.name = "hour"
    sil = float(silhouette_score(prof.values, km.labels_))
    return days, centers[names[:k]], sil
