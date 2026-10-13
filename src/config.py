"""Central configuration. Change values here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RAW_CSV = DATA_DIR / "energy_hourly.csv"

SEED = 42

# ---- synthetic data ----
N_DAYS = 730
START_DATE = "2024-07-01"

# ---- carbon and cost (edit to your region) ----
EMISSION_FACTOR = 0.82   # kg CO2 per kWh (approximate grid average; verify for your country/year)
TARIFF = 7.0             # currency units per kWh (INR used as example)
PEAK_HOURS = (18, 22)    # [start, end) hours with higher tariff
PEAK_MULTIPLIER = 1.5
KG_CO2_PER_TREE_YEAR = 21.0  # approximate absorption of one mature tree per year

# ---- modelling ----
TEST_FRACTION = 0.2
N_CLUSTERS = 3
ANOMALY_CONTAMINATION = 0.02

# ---- reinforcement learning (AC control simulation) ----
ACTIONS = ["OFF", "ECO", "COOL"]
AC_KWH = [0.0, 0.8, 1.5]       # energy used per hour for each action
COOLING = [0.0, 1.2, 2.5]      # degrees C of cooling per hour for each action
HEAT_LEAK = 0.2                # how fast indoor temp drifts to outdoor temp
MIN_INDOOR = 21.0
COMFORT_LIMIT = 26.0           # occupants feel discomfort above this indoor temp
COMFORT_WEIGHT = 0.6           # penalty per degree above the limit (kWh-equivalent)
RL_EPISODES = 15000
ALPHA_MIN = 0.03              # learning rate decays per state-action down to this value
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05


def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
