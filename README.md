# AI-Powered Energy Optimization Agent

**SDG 13 – Climate Action** | IBM SkillsBuild AI/ML Internship – Final Project

An AI agent for smart homes/buildings that **forecasts** electricity demand, **detects** wasteful or faulty
usage, **discovers** usage patterns, and **learns** (reinforcement learning) when to run the air conditioner
to cut energy use and CO2 while keeping people comfortable. Everything uses **free, open-source** tools and
runs on a normal laptop (no GPU, no paid API).

---

## 1. AI/ML techniques used

| Task | Technique | Module |
|---|---|---|
| Demand forecasting | Ridge, Random Forest, Histogram Gradient Boosting (best auto-selected) vs naive baseline | `src/forecasting.py` |
| Waste / fault detection | Isolation Forest (unsupervised anomaly detection) | `src/anomaly.py` |
| Usage-pattern discovery | K-Means clustering of daily load profiles + silhouette score | `src/clustering.py` |
| Energy optimization "agent" | Q-learning (reinforcement learning) controlling AC: OFF / ECO / COOL | `src/rl_agent.py` |
| Climate impact | kWh to CO2 and tree-equivalent conversion | `src/carbon.py` |
| Explanations | Rule-based tips; optional IBM watsonx.ai (Granite) rewrite | `src/insights.py` |

## 2. Tech stack (all free)

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data / numerics | pandas, NumPy |
| Machine learning | scikit-learn |
| Model saving | joblib |
| Charts | matplotlib, Streamlit built-in charts |
| Dashboard | Streamlit |
| Testing | pytest |
| Version control / hosting | Git + GitHub, Streamlit Community Cloud (free deploy) |
| Optional GenAI | IBM watsonx.ai (check current free-plan limits), `ibm-watsonx-ai` |

## 3. Free resources

- **Python:** python.org | **Editor:** VS Code | **Notebook alternative:** Google Colab (colab.research.google.com)
- **Real dataset (optional):** UCI "Individual household electric power consumption"
  https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
  (Kaggle also hosts many free smart-meter and building-energy datasets)
- **Deploy dashboard free:** share.streamlit.io (needs a public GitHub repo)
- **IBM:** IBM SkillsBuild (skillsbuild.org) for learning; IBM Cloud / watsonx.ai free tier for the optional GenAI step

## 4. Architecture

```
 Smart-meter data (synthetic or UCI)
            |
   preprocess.py  (lags, rolling mean, time features, time-based split)
            |
  +---------+-----------+------------------+
  |         |           |                  |
forecasting anomaly   clustering        rl_agent
(regression) (Isolation (K-Means)    (Q-learning AC control)
             Forest)                          |
  +---------+-----------+------------------+
            |
   carbon.py + insights.py  -> kWh, cost, CO2, tips
            |
 reports/*.csv|json|png  +  models/*.joblib|npy
            |
   app/dashboard.py (Streamlit)
```

## 5. Folder structure

```
energy-optimization-agent/
├── README.md
├── requirements.txt            # modules to install (used by Streamlit Cloud)
├── requirements-dev.txt        # adds pytest for local testing
├── requirements-optional.txt   # IBM watsonx.ai (optional)
├── .streamlit/config.toml      # dashboard theme/settings
├── .gitignore
├── src/
│   ├── config.py               # all settings (emission factor, tariff, RL parameters ...)
│   ├── data_generator.py       # synthetic data + UCI loader
│   ├── preprocess.py           # features + time split
│   ├── forecasting.py          # regression models
│   ├── anomaly.py              # Isolation Forest
│   ├── clustering.py           # K-Means
│   ├── rl_agent.py             # Q-learning AC controller
│   ├── carbon.py               # CO2 / trees
│   ├── insights.py             # tips (+ optional watsonx.ai)
│   └── train_pipeline.py       # runs everything
├── app/dashboard.py            # Streamlit dashboard
├── tests/test_pipeline.py      # quick tests
├── data/                       # energy_hourly.csv is generated here
├── models/                     # trained models are saved here
├── reports/                    # metrics, CSVs, figures (summary.json drives the dashboard)
└── docs/                       # Lean Canvas PDF, report outline
```

## 6. Setup and run

**Windows (Command Prompt / PowerShell)**
```
cd energy-optimization-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```
cd energy-optimization-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run the project (same on all systems, from the project root)**
```
python -m src.train_pipeline      # generates data, trains all models, writes reports/ (about 1 minute)
pip install -r requirements-dev.txt   # optional: only needed for pytest
python tests/test_pipeline.py     # optional: quick checks (or: pytest -q)
streamlit run app/dashboard.py    # opens the dashboard at http://localhost:8501
```

Useful options:
```
python -m src.train_pipeline --days 365 --episodes 20000
python -m src.train_pipeline --source uci --path data/household_power_consumption.txt
```
With `--source uci` the forecasting, anomaly and clustering parts run on real data; the RL part is skipped
because that dataset has no temperature or occupancy columns.

**Modules installed by `requirements.txt`:** numpy, pandas, scikit-learn, matplotlib, joblib, streamlit. (`requirements-dev.txt` adds pytest.)

## 7. What the pipeline does

1. **Data** – 2 years of hourly household data (AC, lighting, other loads, temperature, occupancy) with
   labelled injected faults. Settings: `src/config.py`.
2. **Forecasting** – one-step-ahead hourly forecast using lag features (1h, 24h, 168h), rolling mean,
   time features, temperature and occupancy. Time-based split (last 20% is test, no shuffling).
3. **Anomalies** – Isolation Forest on consumption, deviation from recent average and time-of-day.
4. **Clustering** – K-Means (k=3) on 24-hour profiles: low / moderate / high usage days.
5. **RL agent** – Q-learning over daily episodes. State: time period, occupied now/next hour, indoor and
   outdoor temperature bins. Reward: peak-weighted energy cost plus a comfort penalty when someone is home.
   It is compared with an "always-on habit" and a rule-based smart thermostat.
6. **Impact** – savings converted to cost, CO2 (`EMISSION_FACTOR`) and tree equivalents.

## 8. Results from a reference run

Synthetic data, seed 42, test set = last 146 days. Your numbers may differ slightly with another seed.

| Item | Result |
|---|---|
| Forecast (Hist Gradient Boosting) | MAE 0.12 kWh, R² 0.92 (naive baseline R² 0.07) |
| Anomaly detection | precision 0.53, recall 0.45 against injected faults |
| RL vs always-on habit | about 42% less simulated AC energy (about 824 kg CO2 avoided over 146 days) |
| RL vs smart thermostat rule | similar energy + comfort score, but about 18 vs 228 uncomfortable occupied hours |

## 9. Honest limitations (mention these in your report)

- Data and the thermal/AC model are **synthetic**; savings are **simulated**, not measured in a real building.
- The RL agent assumes an occupancy prediction for the next hour is available (for example from routines or sensors).
- The simple rule-based thermostat uses less energy than the RL agent; the RL agent's advantage is its
  better energy and comfort balance. Change `COMFORT_WEIGHT` in `src/config.py` to shift this trade-off.
- `EMISSION_FACTOR` (0.82 kg CO2/kWh) is an approximate grid average. Replace it with your country's current official value.
- Tabular Q-learning is used for clarity; deep RL (DQN) is a natural extension.

## 10. Optional: IBM watsonx.ai tips

```
pip install -r requirements-optional.txt
# Windows:  set WATSONX_API_KEY=...   set WATSONX_PROJECT_ID=...
# macOS/Linux:  export WATSONX_API_KEY=...  export WATSONX_PROJECT_ID=...
```
Then press **Rewrite with IBM watsonx.ai** in the Tips tab. Set `WATSONX_MODEL_ID` if the default Granite
model is not available in your account. This step is optional and the project works fully without it.

## 11. Deploy for free (GitHub + Streamlit Community Cloud)

Full step-by-step guide: **`docs/DEPLOYMENT_GUIDE.md`**.

Key rules: the repo root must be the folder that contains `requirements.txt`, and the main file path on
Streamlit Cloud is `app/dashboard.py`.

**Live demo:** https://energy-optimization--agent.streamlit.app  
**GitHub repository:** https://github.com/Khushinishad/-Energy-Optimization--Agent

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` on Streamlit Cloud | `requirements.txt` must be at the repo root (see deployment guide) |
| `ModuleNotFoundError: src` | Run commands from the project root, using `python -m src.train_pipeline` |
| Dashboard says "No results found" | Run `python -m src.train_pipeline` first |
| `streamlit` not found | Activate the venv and run `pip install -r requirements.txt` |
| Training too slow | Use `--episodes 5000` |

## 13. Ideas to extend

Real smart-meter data (UCI/Kaggle), LSTM forecasting with TensorFlow/PyTorch, DQN agent, MQTT/Home Assistant
control, appliance-level disaggregation, mobile app, multi-building comparison.
