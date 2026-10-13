"""Reinforcement-learning agent (tabular Q-learning) that controls an air conditioner.

Each day is an episode of 24 hourly decisions.
State  : time-of-day period, occupied now, occupied next hour (occupancy prediction),
         indoor temperature bin, outdoor temperature bin
Actions: OFF / ECO / COOL
Reward : -(energy x tariff multiplier) - comfort penalty (only when someone is home)

The thermal model and occupancy come from the (synthetic) dataset, so results are a simulation.
"""
import numpy as np
import pandas as pd

from . import config
from .carbon import kwh_to_co2

N_ACTIONS = len(config.ACTIONS)
INDOOR_EDGES = [23, 24, 25, 26, 27, 28, 29, 30]    # -> 9 bins
OUTDOOR_EDGES = [24, 27, 30, 33]                   # -> 5 bins
N_STATES = 5 * 2 * 2 * 9 * 5


# ---------- environment ----------
def price_mult(h):
    return config.PEAK_MULTIPLIER if config.PEAK_HOURS[0] <= h < config.PEAK_HOURS[1] else 1.0


def period(h):
    """night, morning, afternoon, evening peak, late evening"""
    return 0 if h < 6 else 1 if h < 12 else 2 if h < 18 else 3 if h < 22 else 4


def encode(h, occ, occ_next, indoor, outdoor):
    ib = int(np.digitize(indoor, INDOOR_EDGES))
    ob = int(np.digitize(outdoor, OUTDOOR_EDGES))
    return (((period(h) * 2 + int(occ)) * 2 + int(occ_next)) * 9 + ib) * 5 + ob


def thermal_step(indoor, outdoor, action):
    nxt = indoor + config.HEAT_LEAK * (outdoor - indoor) - config.COOLING[action]
    return max(config.MIN_INDOOR, nxt)


def comfort_penalty(occ, indoor):
    return config.COMFORT_WEIGHT * max(0.0, indoor - config.COMFORT_LIMIT) if occ else 0.0


def reward_fn(h, occ, indoor_next, action):
    return -(config.AC_KWH[action] * price_mult(h) + comfort_penalty(occ, indoor_next))


def initial_indoor(outdoor0):
    return float(np.clip(outdoor0 - 2, 22, 30))


def _next(occ, h):
    return int(occ[min(h + 1, 23)])


# ---------- training ----------
def train_q_learning(days, episodes=config.RL_EPISODES, seed=config.SEED):
    """days: list of (outdoor_temps[24], occupancy[24]). Returns the Q-table."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    visits = np.zeros_like(Q)
    for ep in range(episodes):
        eps = max(config.EPS_END,
                  config.EPS_START - (config.EPS_START - config.EPS_END) * ep / (0.7 * episodes))
        temps, occ = days[int(rng.integers(len(days)))]
        indoor = initial_indoor(temps[0])
        for h in range(24):
            s = encode(h, occ[h], _next(occ, h), indoor, temps[h])
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            ind2 = thermal_step(indoor, temps[h], a)
            r = reward_fn(h, occ[h], ind2, a)
            if h < 23:
                s2 = encode(h + 1, occ[h + 1], _next(occ, h + 1), ind2, temps[h + 1])
                target = r + config.GAMMA * Q[s2].max()
            else:
                target = r
            visits[s, a] += 1
            alpha = max(config.ALPHA_MIN, 1.0 / (1.0 + 0.1 * visits[s, a]))
            Q[s, a] += alpha * (target - Q[s, a])
            indoor = ind2
    return Q


# ---------- policies ----------
def policy_habit(h, occ, occ_next, indoor, outdoor):
    """Typical habit: AC on full whenever it is hot outside, whether or not anyone is home."""
    return 2 if outdoor >= 28 else 0


def policy_thermostat(h, occ, occ_next, indoor, outdoor):
    """Rule-based smart thermostat: cool only when someone is home and it is warm inside."""
    return 2 if (occ and indoor > config.COMFORT_LIMIT) else 0


def make_rl_policy(Q):
    def policy(h, occ, occ_next, indoor, outdoor):
        return int(np.argmax(Q[encode(h, occ, occ_next, indoor, outdoor)]))
    return policy


# ---------- simulation / evaluation ----------
def simulate_day(temps, occ, policy):
    indoor = initial_indoor(temps[0])
    rows = []
    for h in range(24):
        a = policy(h, int(occ[h]), _next(occ, h), indoor, temps[h])
        indoor = thermal_step(indoor, temps[h], a)
        kwh = config.AC_KWH[a]
        rows.append({"hour": h, "outdoor": float(temps[h]), "occupied": int(occ[h]),
                     "action": config.ACTIONS[a], "indoor": round(indoor, 2), "kwh": kwh,
                     "cost": kwh * config.TARIFF * price_mult(h),
                     "penalty": comfort_penalty(occ[h], indoor),
                     "uncomfortable": int(occ[h] and indoor > config.COMFORT_LIMIT + 1)})
    return pd.DataFrame(rows)


def evaluate_policies(days, policies):
    """Lower 'score' is better: weighted energy + comfort penalty (what the agent optimises)."""
    out = {}
    for name, pol in policies.items():
        s = pd.concat([simulate_day(t, o, pol) for t, o in days])
        kwh = float(s["kwh"].sum())
        weighted = float((s["kwh"] * [price_mult(h) for h in s["hour"]]).sum())
        out[name] = {"ac_kwh": round(kwh, 2),
                     "cost": round(float(s["cost"].sum()), 2),
                     "co2_kg": round(kwh_to_co2(kwh), 2),
                     "uncomfortable_hours": int(s["uncomfortable"].sum()),
                     "occupied_hours": int(s["occupied"].sum()),
                     "score": round(weighted + float(s["penalty"].sum()), 1)}
    return out


def df_to_days(df):
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    d["hour"] = d["timestamp"].dt.hour
    dates, days = [], []
    for date, g in d.groupby("date"):
        if len(g) == 24:
            g = g.sort_values("hour")
            days.append((g["temperature"].to_numpy(float), g["occupied"].to_numpy(int)))
            dates.append(str(date))
    return dates, days


def save_q(Q):
    config.ensure_dirs()
    np.save(config.MODELS_DIR / "q_table.npy", Q)


def load_q():
    return np.load(config.MODELS_DIR / "q_table.npy")
