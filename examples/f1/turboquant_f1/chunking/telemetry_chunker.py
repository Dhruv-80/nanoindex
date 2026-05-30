"""Convert telemetry DataFrames into text narrative chunks."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .schemas import TelemetryChunk
from ..ingestion.telemetry_parser import TELEMETRY_CHANNELS, compute_window_features

logger = logging.getLogger(__name__)


def _window_to_text(
    window: pd.DataFrame,
    driver: str,
    lap: int,
    compound: Optional[str],
    position: Optional[int],
    channels: list[str],
) -> str:
    """Generate a plain-English narrative for a telemetry window."""
    lines = []

    speed = window["Speed"] if "Speed" in window.columns else None
    throttle = window["Throttle"] if "Throttle" in window.columns else None
    brake = window["Brake"] if "Brake" in window.columns else None
    rpm = window["RPM"] if "RPM" in window.columns else None
    gear = window["nGear"] if "nGear" in window.columns else None
    drs = window["DRS"] if "DRS" in window.columns else None

    # Opening context line
    ctx_parts = [f"{driver} lap {lap}"]
    if position is not None:
        ctx_parts.append(f"P{position}")
    if compound:
        ctx_parts.append(f"{compound} compound")
    lines.append(", ".join(ctx_parts) + ":")

    # Speed narrative
    if speed is not None and not speed.isna().all():
        s_min, s_max = speed.min(), speed.max()
        s_delta = speed.iloc[-1] - speed.iloc[0]
        direction = "accelerating" if s_delta > 5 else "braking" if s_delta < -5 else "steady"
        lines.append(f"  Speed {direction} from {s_min:.0f} to {s_max:.0f} kph (Δ{s_delta:+.0f} kph).")

    # Throttle / brake narrative
    if throttle is not None and brake is not None:
        avg_thr = throttle.mean()
        avg_brk = brake.mean()
        max_brk = brake.max()
        if avg_thr > 80:
            lines.append(f"  Full throttle ({avg_thr:.0f}% avg).")
        elif avg_thr > 40:
            lines.append(f"  Partial throttle ({avg_thr:.0f}% avg).")
        if max_brk > 50:
            lines.append(f"  Heavy braking applied (peak {max_brk:.0f}%, avg {avg_brk:.0f}%).")
        elif max_brk > 10:
            lines.append(f"  Light braking (peak {max_brk:.0f}%).")

    # Gear narrative
    if gear is not None and not gear.isna().all():
        g_min, g_max = int(gear.min()), int(gear.max())
        if g_min != g_max:
            lines.append(f"  Gear changes: {g_max} → {g_min}." if g_max > g_min else f"  Upshifting {g_min} → {g_max}.")
        else:
            lines.append(f"  Holding gear {g_min}.")

    # DRS
    if drs is not None and not drs.isna().all():
        drs_open = (drs > 8).any()
        if drs_open:
            lines.append("  DRS active.")

    # RPM
    if rpm is not None and not rpm.isna().all():
        lines.append(f"  RPM range {rpm.min():.0f}–{rpm.max():.0f}.")

    return "\n".join(lines)


def _build_lap_intervals(laps_df: pd.DataFrame, driver_number: str) -> list[tuple[float, float, int, Optional[str], Optional[int]]]:
    """Return list of (start_s, end_s, lap_num, compound, position) for a driver."""
    drv_laps = laps_df[laps_df["DriverNumber"] == driver_number].copy()
    intervals = []
    for _, row in drv_laps.iterrows():
        lap_num = int(row["LapNumber"]) if pd.notna(row.get("LapNumber")) else 0
        start_s = pd.to_timedelta(row["LapStartTime"]).total_seconds() if pd.notna(row.get("LapStartTime")) else 0.0
        end_s   = pd.to_timedelta(row["Time"]).total_seconds() if pd.notna(row.get("Time")) else float("inf")
        compound = str(row["Compound"]) if pd.notna(row.get("Compound")) else None
        pos_val  = row.get("Position")
        position = int(pos_val) if pos_val is not None and pd.notna(pos_val) else None
        intervals.append((start_s, end_s, lap_num, compound, position))
    return intervals


def _lap_at_time(intervals: list, t: float) -> tuple[int, Optional[str], Optional[int]]:
    """Return (lap_num, compound, position) for the lap covering session time t."""
    for start_s, end_s, lap_num, compound, position in intervals:
        if start_s <= t < end_s:
            return lap_num, compound, position
    return 0, None, None


def chunk_driver_telemetry(
    telemetry: pd.DataFrame,
    driver: str,
    year: int,
    race: str,
    session: str,
    window_seconds: float = 5.0,
    channels: list[str] = TELEMETRY_CHANNELS,
    laps_df: Optional[pd.DataFrame] = None,
    driver_map: Optional[dict] = None,
) -> list[TelemetryChunk]:
    """
    Slice a driver's telemetry into fixed-length windows and convert each
    to a TelemetryChunk with text narrative and numerical feature vector.

    driver_map: optional {DriverNumber_str -> three-letter code} mapping
    """
    drv_telem = telemetry[telemetry["DriverNumber"] == driver].sort_values("SessionSeconds")
    if drv_telem.empty:
        return []

    driver_code = driver_map.get(str(driver), str(driver)) if driver_map else str(driver)

    lap_intervals: list = []
    if laps_df is not None and not laps_df.empty:
        lap_intervals = _build_lap_intervals(laps_df, driver)

    chunks: list[TelemetryChunk] = []
    t_start = drv_telem["SessionSeconds"].iloc[0]
    t_end_max = drv_telem["SessionSeconds"].iloc[-1]

    t = t_start
    while t < t_end_max:
        t_next = t + window_seconds
        window = drv_telem[(drv_telem["SessionSeconds"] >= t) & (drv_telem["SessionSeconds"] < t_next)]
        if len(window) < 3:
            t = t_next
            continue

        if lap_intervals:
            lap_num, compound, position = _lap_at_time(lap_intervals, t + window_seconds / 2)
        elif "LapNumber" in window.columns:
            lap_num = int(window["LapNumber"].mode().iloc[0])
            compound, position = None, None
        else:
            lap_num, compound, position = 0, None, None

        text = _window_to_text(window, driver_code, lap_num, compound, position, channels)
        num_features = compute_window_features(window, channels)

        chunk_id = hashlib.md5(
            f"{year}:{race}:{session}:{driver}:{t:.2f}".encode()
        ).hexdigest()[:16]

        chunks.append(TelemetryChunk(
            id=chunk_id,
            text=text,
            chunk_type="telemetry",
            year=year,
            race=race,
            session=session,
            driver=driver_code,
            lap=lap_num,
            timestamp_start=float(t),
            timestamp_end=float(t_next),
            compound=compound,
            position=position,
            numerical_features=num_features.tolist(),
        ))
        t = t_next

    logger.debug("Driver %s (%s): generated %d telemetry chunks", driver, driver_code, len(chunks))
    return chunks
