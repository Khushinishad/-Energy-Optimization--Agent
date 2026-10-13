"""Streamlit dashboard. Run the pipeline first:  python -m src.train_pipeline
Then:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, rl_agent                      # noqa: E402
from src.insights import generate_tips, llm_summary    # noqa: E402

st.set_page_config(page_title="AI Energy Optimization Agent", page_icon="⚡", layout="wide")
st.title("⚡ AI-Powered Energy Optimization Agent")
st.caption("SDG 13 – Climate Action | IBM SkillsBuild AI/ML Internship Final Project")

SUMMARY = config.REPORTS_DIR / "summary.json"
if not SUMMARY.exists():
    # First start on a fresh deployment without committed results: build them automatically.
    from src import train_pipeline                      # noqa: E402
    with st.spinner("First run: generating data and training models (about 1 minute)..."):
        train_pipeline.run(episodes=8000)
    st.rerun()


@st.cache_data
def load():
    s = json.loads(SUMMARY.read_text())
    df = pd.read_csv(config.RAW_CSV, parse_dates=["timestamp"])
    fc = pd.read_csv(config.REPORTS_DIR / "forecast_test.csv", parse_dates=["timestamp"])
    fl = pd.read_csv(config.REPORTS_DIR / "anomaly_flags_test.csv", parse_dates=["timestamp"])
    cl = pd.read_csv(config.REPORTS_DIR / "daily_clusters.csv")
    prof = pd.read_csv(config.REPORTS_DIR / "cluster_profiles.csv", index_col=0)
    return s, df, fc, fl, cl, prof


summary, df, fc, flags, clusters, profiles = load()
rl = summary.get("rl")
best = next(m for m in summary["forecast"]["metrics"] if m["model"] == summary["forecast"]["best_model"])

# ---- KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast error (MAE)", f"{best['MAE']:.2f} kWh", f"R² {best['R2']:.2f}", delta_color="off")
c2.metric("Unusual hours flagged", summary["anomaly"]["n_flagged"])
if rl:
    c3.metric("AC energy saved vs habit", f"{rl['kwh_saved_pct_vs_habit']:.0f}%", f"{rl['kwh_saved']:.0f} kWh")
    c4.metric("CO₂ avoided (simulated)", f"{rl['co2_saved_kg']:.0f} kg", f"≈ {rl['trees_equivalent']:.0f} trees/yr")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast", "Anomalies", "Usage patterns", "RL agent", "Tips"])

with tab1:
    st.subheader(f"Demand forecast – {summary['forecast']['best_model']}")
    n = st.slider("Days to show (end of test period)", 2, 30, 7)
    st.line_chart(fc.tail(24 * n).set_index("timestamp")[["actual_kwh", "predicted_kwh"]])
    st.dataframe(pd.DataFrame(summary["forecast"]["metrics"]).round(3), hide_index=True)

with tab2:
    st.subheader("Anomaly detection (Isolation Forest)")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(flags["timestamp"], flags["total_kwh"], lw=0.6)
    f = flags[flags["flag"] == 1]
    ax.scatter(f["timestamp"], f["total_kwh"], c="red", s=12, label="Flagged")
    ax.set_ylabel("kWh per hour"); ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    a = summary["anomaly"]
    if "precision" in a:
        st.caption(f"Against injected faults in the synthetic data: precision {a['precision']:.2f}, "
                   f"recall {a['recall']:.2f}, F1 {a['f1']:.2f}")
    st.dataframe(f.sort_values("excess_kwh", ascending=False).head(15)[["timestamp", "total_kwh", "excess_kwh"]],
                 hide_index=True)

with tab3:
    st.subheader("Daily usage patterns (K-Means)")
    st.line_chart(profiles)
    st.bar_chart(clusters["cluster"].value_counts())
    st.caption(f"Silhouette score: {summary['clustering']['silhouette']}")

with tab4:
    if not rl:
        st.info("RL needs temperature and occupancy columns (synthetic data).")
    else:
        st.subheader("Reinforcement-learning AC controller")
        res = pd.DataFrame(rl["policies"]).T
        st.dataframe(res[["ac_kwh", "cost", "co2_kg", "uncomfortable_hours", "score"]])
        st.caption("Score = peak-weighted energy + comfort penalty (lower is better). "
                   "All numbers are simulated on the test days.")
        dates, days = rl_agent.df_to_days(df)
        test_dates = dates[int(len(dates) * (1 - config.TEST_FRACTION)):]
        pick = st.selectbox("Replay a test day", test_dates, index=len(test_dates) - 1)
        temps, occ = days[dates.index(pick)]
        Q = rl_agent.load_q()
        habit = rl_agent.simulate_day(temps, occ, rl_agent.policy_habit)
        agent = rl_agent.simulate_day(temps, occ, rl_agent.make_rl_policy(Q))
        cmp = pd.DataFrame({"Outdoor": agent["outdoor"], "Indoor (habit)": habit["indoor"],
                            "Indoor (RL agent)": agent["indoor"]}, index=agent["hour"])
        st.line_chart(cmp)
        d1, d2 = st.columns(2)
        d1.metric("Habit AC energy", f"{habit['kwh'].sum():.1f} kWh")
        d2.metric("RL agent AC energy", f"{agent['kwh'].sum():.1f} kWh",
                  f"{agent['kwh'].sum() - habit['kwh'].sum():.1f} kWh")
        st.dataframe(agent[["hour", "occupied", "action", "indoor", "kwh"]], hide_index=True)

with tab5:
    st.subheader("Personalised energy tips")
    tips = generate_tips(df, flags, summary)
    for t in tips:
        st.write("• " + t)
    if st.button("Rewrite with IBM watsonx.ai (optional)"):
        out = llm_summary(tips)
        st.write(out if out else "watsonx.ai is not configured – see README (optional step).")
