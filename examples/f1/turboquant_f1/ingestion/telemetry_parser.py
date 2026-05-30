"""Normalize and clean raw telemetry DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

TELEMETRY_CHANNELS = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]

# Expected value ranges for basic sanity checking
_CHANNEL_RANGES: dict[str, tuple[float, float]] = {
    "Speed": (0, 400),
    "Throttle": (0, 100),
    "Brake": (0, 1),          # FastF1 returns 0/1 boolean or 0-100 float depending on year
    "RPM": (0, 20_000),
    "nGear": (0, 8),
    "DRS": (0, 14),            # 0 = closed, 10/12/14 = open
}


def normalize_telemetry(df: pd.DataFrame, channels: list[str] = TELEMETRY_CHANNELS) -> pd.DataFrame:
    """
    Return a cleaned telemetry dataframe with only the requested channels
    plus session-relative time as a float column 'SessionSeconds'.

    Works with both per-driver telemetry (has LapNumber) and the merged
    session-level telemetry saved by fastf1_loader (no LapNumber).
    Uses SessionTime when available; falls back to Time.
    """
    present = [c for c in channels if c in df.columns]
    optional = [c for c in ["LapNumber"] if c in df.columns]
    # Prefer SessionTime (session-relative) over Time for merged telemetry
    time_col = "SessionTime" if "SessionTime" in df.columns else "Time"
    out = df[[time_col, "DriverNumber", *optional, *present]].copy()

    # Convert timedelta to float seconds
    if pd.api.types.is_timedelta64_dtype(out[time_col]):
        out["SessionSeconds"] = out[time_col].dt.total_seconds()
    else:
        out["SessionSeconds"] = pd.to_timedelta(out[time_col]).dt.total_seconds()

    # Brake: normalise to 0-100 float if it's boolean
    if "Brake" in out.columns and out["Brake"].dtype == bool:
        out["Brake"] = out["Brake"].astype(float) * 100.0

    # Drop rows where all telemetry channels are NaN
    out = out.dropna(subset=present, how="all")

    # Clip to sane ranges
    for ch, (lo, hi) in _CHANNEL_RANGES.items():
        if ch in out.columns:
            out[ch] = out[ch].clip(lo, hi)

    return out.reset_index(drop=True)


def compute_window_features(window: pd.DataFrame, channels: list[str] = TELEMETRY_CHANNELS) -> np.ndarray:
    """
    Reduce a telemetry window to a fixed-size numerical feature vector.
    Per channel: mean, std, min, max, delta (last - first).
    Returns a 1-D array of length len(channels) * 5.
    """
    feats = []
    for ch in channels:
        if ch not in window.columns or window[ch].isna().all():
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        vals = window[ch].dropna().values.astype(float)
        feats.append(float(np.mean(vals)))
        feats.append(float(np.std(vals)))
        feats.append(float(np.min(vals)))
        feats.append(float(np.max(vals)))
        feats.append(float(vals[-1] - vals[0]))
    return np.array(feats, dtype=np.float32)
