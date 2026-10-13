"""Run the full pipeline: data -> forecasting -> anomalies -> clustering -> RL agent -> reports.

Usage (from the project root):
    python -m src.train_pipeline
    python -m src.train_pipeline --days 365 --episodes 8000
    python -m src.train_pipeline --source uci --path data/household_power_consumption.txt
"""
import argparse
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import anomaly, clustering, config, data_generator, forecasting, preprocess, rl_agent
from .carbon import kwh_to_co2, trees_equivalent


def _fig(name):
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / name, dpi=130)
    plt.close()


def run(source="synthetic", path=None, n_days=config.N_DAYS, episodes=config.RL_EPISODES):
    config.ensure_dirs()
    summary = {"meta": {"source": source, "generated_at": datetime.now().isoformat(timespec="seconds")}}

    print("[1/5] Loading data ...")
    if source == "uci":
        df = data_generator.load_uci_hourly(path)
    else:
        df = data_generator.generate(n_days)
    df.to_csv(config.RAW_CSV, index=False)
    summary["meta"]["rows"] = int(len(df))

    feats = preprocess.build_features(df)
    train, test = preprocess.time_split(feats)
    cols = preprocess.feature_columns(feats)

    print("[2/5] Training demand-forecasting models ...")
    metrics, pred, best = forecasting.train_forecaster(train, test, cols)
    print(metrics.round(3).to_string(index=False))
    summary["forecast"] = {"best_model": best, "metrics": metrics.round(4).to_dict("records")}
    fc = pd.DataFrame({"timestamp": test["timestamp"], "actual_kwh": test["total_kwh"],
                       "predicted_kwh": pred.round(3)})
    fc.to_csv(config.REPORTS_DIR / "forecast_test.csv", index=False)
    last = fc.tail(24 * 7)
    plt.figure(figsize=(10, 3.6))
    plt.plot(last["timestamp"], last["actual_kwh"], label="Actual")
    plt.plot(last["timestamp"], last["predicted_kwh"], label=f"Predicted ({best})")
    plt.ylabel("kWh per hour"); plt.title("Demand forecast - last 7 days of test set"); plt.legend()
    _fig("forecast.png")

    print("[3/5] Detecting anomalies ...")
    det = anomaly.train_detector(train)
    flags = anomaly.detect(det, test)
    flags.to_csv(config.REPORTS_DIR / "anomaly_flags_test.csv", index=False)
    summary["anomaly"] = anomaly.evaluate(flags)
    print(summary["anomaly"])
    plt.figure(figsize=(10, 3.6))
    plt.plot(flags["timestamp"], flags["total_kwh"], lw=0.6, label="Consumption")
    f = flags[flags["flag"] == 1]
    plt.scatter(f["timestamp"], f["total_kwh"], c="red", s=10, label="Flagged")
    plt.ylabel("kWh per hour"); plt.title("Anomaly detection (Isolation Forest)"); plt.legend()
    _fig("anomalies.png")

    print("[4/5] Clustering daily usage patterns ...")
    days_df, centers, sil = clustering.cluster_days(df)
    days_df.to_csv(config.REPORTS_DIR / "daily_clusters.csv", index=False)
    centers.to_csv(config.REPORTS_DIR / "cluster_profiles.csv")
    summary["clustering"] = {"k": config.N_CLUSTERS, "silhouette": round(sil, 3),
                             "counts": days_df["cluster"].value_counts().to_dict()}
    centers.plot(figsize=(8, 3.6), title="Average daily load profile per cluster")
    plt.xlabel("Hour of day"); plt.ylabel("kWh")
    _fig("clusters.png")

    if {"temperature", "occupied"}.issubset(df.columns):
        print("[5/5] Training reinforcement-learning AC agent ...")
        dates, days = rl_agent.df_to_days(df)
        cut = int(len(days) * (1 - config.TEST_FRACTION))
        Q = rl_agent.train_q_learning(days[:cut], episodes=episodes)
        rl_agent.save_q(Q)
        policies = {"Always-on habit": rl_agent.policy_habit,
                    "Smart thermostat (rule)": rl_agent.policy_thermostat,
                    "RL agent (Q-learning)": rl_agent.make_rl_policy(Q)}
        res = rl_agent.evaluate_policies(days[cut:], policies)
        base, rl = res["Always-on habit"], res["RL agent (Q-learning)"]
        saved = base["ac_kwh"] - rl["ac_kwh"]
        co2 = kwh_to_co2(saved)
        summary["rl"] = {
            "test_days": len(days) - cut, "policies": res,
            "kwh_saved": round(saved, 2),
            "kwh_saved_pct_vs_habit": round(saved / base["ac_kwh"] * 100, 1) if base["ac_kwh"] else 0.0,
            "cost_saved": round(base["cost"] - rl["cost"], 2),
            "co2_saved_kg": round(co2, 2), "trees_equivalent": round(trees_equivalent(co2), 1)}
        print(json.dumps(summary["rl"], indent=2))
        names = list(res)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].bar(names, [res[n]["ac_kwh"] for n in names], color=["#999", "#6aa", "#1F6F78"])
        ax[0].set_title("AC energy on test days (kWh)")
        ax[1].bar(names, [res[n]["score"] for n in names], color=["#999", "#6aa", "#1F6F78"])
        ax[1].set_title("Energy + comfort score (lower is better)")
        for a in ax:
            a.tick_params(axis="x", labelrotation=15, labelsize=7)
        _fig("rl_comparison.png")
    else:
        print("[5/5] Skipped RL: needs 'temperature' and 'occupied' columns.")

    with open(config.REPORTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Outputs in reports/ and models/. Launch the dashboard with:\n"
          "    streamlit run app/dashboard.py")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "uci"], default="synthetic")
    ap.add_argument("--path", default=None, help="path to UCI household_power_consumption.txt")
    ap.add_argument("--days", type=int, default=config.N_DAYS)
    ap.add_argument("--episodes", type=int, default=config.RL_EPISODES)
    a = ap.parse_args()
    run(a.source, a.path, a.days, a.episodes)
