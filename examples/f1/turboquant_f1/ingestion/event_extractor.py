"""Extract structured race events: flags, pit stops, derived anomalies."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_race_control_events(session: fastf1.core.Session) -> list[dict]:
    """Convert race_control_messages DataFrame to a list of event dicts."""
    rc = session.race_control_messages
    if rc is None or rc.empty:
        return []
    records = rc.to_dict(orient="records")
    return [{k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in r.items()} for r in records]


def extract_pit_stops(laps: pd.DataFrame) -> list[dict]:
    """
    Identify pit stops from lap data.
    A pit-in is signalled by the PitInTime column being non-null in FastF1 laps.
    """
    if "PitInTime" not in laps.columns:
        return []
    pit_laps = laps[laps["PitInTime"].notna()].copy()
    events = []
    for _, row in pit_laps.iterrows():
        events.append({
            "event_type": "pit_stop",
            "driver": row.get("Driver", row.get("DriverNumber", "?")),
            "lap": int(row["LapNumber"]) if pd.notna(row.get("LapNumber")) else None,
            "pit_in_time": str(row["PitInTime"]),
            "pit_out_time": str(row["PitOutTime"]) if pd.notna(row.get("PitOutTime")) else None,
            "compound": row.get("Compound"),
        })
    return events


# ---------------------------------------------------------------------------
# Derived event detectors
# ---------------------------------------------------------------------------

def detect_lock_ups(telemetry: pd.DataFrame, brake_threshold: float = 80.0, speed_drop_kph: float = 20.0) -> list[dict]:
    """
    Heuristic: sudden high brake + rapid speed drop without proportional throttle lift
    suggests a lock-up or heavy braking event.
    """
    events = []
    for drv, grp in telemetry.groupby("DriverNumber"):
        grp = grp.sort_values("SessionSeconds").reset_index(drop=True)
        if "Brake" not in grp.columns or "Speed" not in grp.columns:
            continue
        speed = grp["Speed"].values
        brake = grp["Brake"].values
        for i in range(1, len(grp) - 1):
            if brake[i] >= brake_threshold and (speed[i - 1] - speed[i]) >= speed_drop_kph:
                events.append({
                    "event_type": "lock_up",
                    "driver": drv,
                    "session_seconds": float(grp["SessionSeconds"].iloc[i]),
                    "lap": int(grp["LapNumber"].iloc[i]) if "LapNumber" in grp.columns else None,
                    "speed_before": float(speed[i - 1]),
                    "speed_at": float(speed[i]),
                    "brake_pct": float(brake[i]),
                })
    return events


def detect_anomalous_laps(laps: pd.DataFrame, sigma_threshold: float = 2.0) -> list[dict]:
    """Flag laps where the driver's lap time deviates > sigma_threshold σ from their stint rolling mean."""
    events = []
    if "LapTime" not in laps.columns or "Driver" not in laps.columns:
        return events

    for drv, grp in laps.groupby("Driver"):
        grp = grp.sort_values("LapNumber").copy()
        lap_times = pd.to_timedelta(grp["LapTime"]).dt.total_seconds()
        mean_t = lap_times.mean()
        std_t = lap_times.std()
        if std_t == 0 or np.isnan(std_t):
            continue
        for idx, lt in zip(grp.index, lap_times):
            if abs(lt - mean_t) > sigma_threshold * std_t:
                events.append({
                    "event_type": "anomalous_lap",
                    "driver": drv,
                    "lap": int(grp.loc[idx, "LapNumber"]),
                    "lap_time_s": float(lt),
                    "mean_lap_s": float(mean_t),
                    "z_score": float((lt - mean_t) / std_t),
                })
    return events


def save_events(events: list[dict], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(events, f, indent=2, default=str)
    logger.info("Saved %d events → %s", len(events), out_path)


def load_events(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
