"""FastF1 data extraction — fetches and caches sessions to Parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

# Sessions FastF1 recognises for a race weekend
SESSION_IDENTIFIERS = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "SQ": "Sprint Qualifying",
    "S": "Sprint",
    "R": "Race",
}


def setup_cache(cache_dir: str | Path) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def load_session(
    year: int,
    round_number: int,
    session_identifier: str,
    cache_dir: str | Path,
) -> fastf1.core.Session | None:
    """Load a FastF1 session, returning None if it doesn't exist for that event."""
    setup_cache(cache_dir)
    try:
        session = fastf1.get_session(year, round_number, session_identifier)
        session.load(telemetry=True, laps=True, weather=True, messages=True)
        return session
    except Exception as exc:
        logger.warning("Could not load %s round %d %s: %s", year, round_number, session_identifier, exc)
        return None


def extract_session_data(session: fastf1.core.Session) -> dict[str, pd.DataFrame]:
    """Pull the raw tables we care about from a loaded session."""
    data: dict[str, pd.DataFrame] = {}

    # Lap-level data
    if session.laps is not None and not session.laps.empty:
        data["laps"] = session.laps.reset_index(drop=True)

    # Telemetry (all drivers merged, keyed by DriverNumber)
    telem_frames = []
    for drv in session.drivers:
        try:
            drv_laps = session.laps.pick_drivers(drv)
            telem = drv_laps.get_telemetry()
            telem["DriverNumber"] = drv
            telem_frames.append(telem)
        except Exception as exc:
            logger.debug("No telemetry for driver %s: %s", drv, exc)
    if telem_frames:
        data["telemetry"] = pd.concat(telem_frames, ignore_index=True)

    # Weather
    if session.weather_data is not None and not session.weather_data.empty:
        data["weather"] = session.weather_data.reset_index(drop=True)

    # Race control messages (flags, SC, VSC, penalties …)
    if session.race_control_messages is not None and not session.race_control_messages.empty:
        data["race_control"] = session.race_control_messages.reset_index(drop=True)

    return data


def save_session_data(
    data: dict[str, pd.DataFrame],
    out_dir: str | Path,
) -> None:
    """Write each dataframe to a Parquet file under out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        dest = out_path / f"{name}.parquet"
        df.to_parquet(dest, index=False)
        logger.info("Saved %s → %s (%d rows)", name, dest, len(df))


def load_session_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read previously saved Parquet files back into dataframes."""
    data_path = Path(data_dir)
    return {p.stem: pd.read_parquet(p) for p in sorted(data_path.glob("*.parquet"))}
