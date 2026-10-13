"""Plain-language energy tips. Rule-based by default; optional IBM watsonx.ai rewrite."""
import os

import pandas as pd

from . import config


def generate_tips(df, flags=None, summary=None):
    tips = []
    total = df["total_kwh"].sum()
    if "ac_kwh" in df.columns and total > 0:
        ac = df["ac_kwh"].sum()
        tips.append(f"Air conditioning is {ac / total * 100:.0f}% of total consumption, "
                    "making it the largest controllable load.")
        if "occupied" in df.columns and ac > 0:
            empty = df.loc[df["occupied"] == 0, "ac_kwh"].sum()
            hrs = int(((df["occupied"] == 0) & (df["ac_kwh"] > 0)).sum())
            tips.append(f"The AC ran for {hrs} hours when nobody was home, wasting "
                        f"{empty:.0f} kWh ({empty / ac * 100:.0f}% of AC energy).")
    a, b = config.PEAK_HOURS
    peak = df.loc[(df["timestamp"].dt.hour >= a) & (df["timestamp"].dt.hour < b), "total_kwh"].sum()
    tips.append(f"{peak / total * 100:.0f}% of energy is used between {a}:00 and {b}:00 "
                "(peak tariff). Shifting flexible loads out of this window lowers cost.")
    if flags is not None and len(flags):
        f = flags[flags["flag"] == 1]
        if len(f):
            worst = f.sort_values("excess_kwh", ascending=False).iloc[0]
            tips.append(f"{len(f)} unusual hours were detected; the largest was on "
                        f"{pd.Timestamp(worst['timestamp']):%d %b at %H:%M} "
                        f"(+{worst['excess_kwh']:.1f} kWh above the 24-hour average).")
    rl = (summary or {}).get("rl")
    if rl:
        tips.append(f"The learned AC policy cut simulated AC energy by {rl['kwh_saved_pct_vs_habit']:.0f}% "
                    f"versus the always-on habit, avoiding {rl['co2_saved_kg']:.0f} kg CO2 "
                    "over the test period.")
    return tips


def llm_summary(tips):
    """Optional: rewrite the tips as a friendly note using IBM watsonx.ai (free Lite plan).

    Needs: pip install ibm-watsonx-ai  and env vars WATSONX_API_KEY, WATSONX_PROJECT_ID
    (optional WATSONX_URL, WATSONX_MODEL_ID). Returns None if not configured or on any error.
    """
    key, project = os.getenv("WATSONX_API_KEY"), os.getenv("WATSONX_PROJECT_ID")
    if not (key and project):
        return None
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        model = ModelInference(
            model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct"),
            credentials=Credentials(url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
                                    api_key=key),
            project_id=project, params={"max_new_tokens": 200})
        prompt = ("Rewrite these household energy findings as a short, friendly note with "
                  "3 actionable steps:\n- " + "\n- ".join(tips))
        return model.generate_text(prompt=prompt)
    except Exception:
        return None
