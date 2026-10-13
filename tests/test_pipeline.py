"""Fast checks. Run with:  pytest -q   (or  python tests/test_pipeline.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import anomaly, carbon, clustering, config, data_generator, forecasting, preprocess, rl_agent


def _data(days=60):
    return data_generator.generate(days)


def test_generator_shape_and_columns():
    df = _data(30)
    assert len(df) == 30 * 24
    assert {"timestamp", "temperature", "occupied", "total_kwh", "is_anomaly"} <= set(df.columns)
    assert (df["total_kwh"] > 0).all()


def test_features_have_no_nans():
    f = preprocess.build_features(_data())
    assert not f[preprocess.feature_columns(f)].isna().any().any()


def test_forecaster_beats_naive_baseline():
    f = preprocess.build_features(_data(120))
    tr, te = preprocess.time_split(f)
    metrics, pred, best = forecasting.train_forecaster(tr, te, preprocess.feature_columns(f), save=False)
    naive = metrics.loc[0, "RMSE"]
    assert metrics[metrics["model"] == best]["RMSE"].iloc[0] < naive


def test_anomaly_and_clustering_run():
    f = preprocess.build_features(_data(120))
    tr, te = preprocess.time_split(f)
    flags = anomaly.detect(anomaly.train_detector(tr, save=False), te)
    assert flags["flag"].isin([0, 1]).all()
    days, centers, sil = clustering.cluster_days(_data(120))
    assert centers.shape[1] == config.N_CLUSTERS and -1 <= sil <= 1


def test_rl_learns_to_beat_habit():
    _, days = rl_agent.df_to_days(_data(90))
    Q = rl_agent.train_q_learning(days, episodes=3000)
    assert Q.shape == (rl_agent.N_STATES, rl_agent.N_ACTIONS)
    res = rl_agent.evaluate_policies(days, {"habit": rl_agent.policy_habit,
                                            "rl": rl_agent.make_rl_policy(Q)})
    assert res["rl"]["score"] < res["habit"]["score"]


def test_carbon_conversion():
    assert np.isclose(carbon.kwh_to_co2(100, 0.8), 80.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
