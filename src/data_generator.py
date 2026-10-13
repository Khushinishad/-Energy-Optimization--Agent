"""Synthetic smart-home hourly energy data + loader for the UCI household dataset."""
import numpy as np
import pandas as pd

from . import config


def _occupancy(n_days, start_date, rng):
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    occ = np.zeros((n_days, 24), dtype=int)
    for d, date in enumerate(dates):
        if rng.random() < 0.04:          # away day (trip)
            continue
        if date.dayofweek < 5:           # weekday: out during working hours
            leave = rng.integers(7, 10)
            back = rng.integers(16, 20)
            occ[d, :leave] = 1
            occ[d, back:] = 1
        else:                            # weekend: mostly home
            occ[d, :] = 1
            if rng.random() < 0.25:
                s = rng.integers(10, 15)
                occ[d, s:s + rng.integers(2, 5)] = 0
    return occ.reshape(-1)


def generate(n_days=config.N_DAYS, seed=config.SEED, start_date=config.START_DATE):
    """Create realistic hourly household data with injected anomalies (labelled)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_date, periods=n_days * 24, freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()

    temp = (27 + 7 * np.sin(2 * np.pi * (doy - 105) / 365)
            + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
            + rng.normal(0, 1.0, n))
    occ = _occupancy(n_days, start_date, rng)

    # Air conditioner: used when hot; on wasteful days it is left running when nobody is home
    waste = np.repeat(rng.random(n_days) < 0.30, 24)
    ac_on = (temp > 28) & ((occ == 1) | waste)
    ac = np.where(ac_on, 0.9 + 0.12 * (temp - 28) + rng.normal(0, 0.08, n), 0.0).clip(0)

    dark = (hour >= 18) | (hour < 6)
    light = np.where((occ == 1) & dark, 0.25 + rng.normal(0, 0.04, n), 0.0)
    light += np.where((occ == 0) & dark & (rng.random(n) < 0.15), 0.2, 0.0)

    cooking = np.isin(hour, [7, 8, 13, 19, 20, 21]) & (occ == 1)
    other = (0.12 + np.where(cooking, 0.5 + rng.uniform(0, 0.4, n), 0.0)
             + np.where(occ == 1, 0.1 + rng.uniform(0, 0.15, n), 0.0)
             + rng.normal(0, 0.02, n)).clip(0.05)

    # Inject anomalies: appliance left on / fault for a few hours
    is_anom = np.zeros(n, dtype=int)
    for _ in range(max(3, int(n * 0.012 / 4))):
        s = int(rng.integers(0, n - 6))
        length = int(rng.integers(3, 7))
        other[s:s + length] += rng.uniform(1.5, 3.0)
        is_anom[s:s + length] = 1

    total = ac + light + other
    return pd.DataFrame({
        "timestamp": idx, "hour": hour, "dayofweek": idx.dayofweek.to_numpy(),
        "month": idx.month.to_numpy(), "temperature": temp.round(2), "occupied": occ,
        "ac_kwh": ac.round(3), "lighting_kwh": light.round(3), "other_kwh": other.round(3),
        "total_kwh": total.round(3), "is_anomaly": is_anom,
    })


def load_uci_hourly(path):
    """Load the UCI 'Individual household electric power consumption' text file as hourly kWh.

    Download (free): https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    Only 'timestamp' and 'total_kwh' are available; the RL part is skipped for this source.
    """
    raw = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], dayfirst=True)
    kw = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    s = pd.Series(kw.values, index=ts).resample("h").mean()   # mean kW over 1 hour == kWh
    s = s.interpolate(limit=6).dropna().rename("total_kwh").rename_axis("timestamp")
    return s.reset_index()
