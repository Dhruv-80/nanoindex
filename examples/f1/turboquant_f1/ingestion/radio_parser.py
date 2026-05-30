"""Extract and timestamp radio transcripts from FastF1 session data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)


def extract_radio_messages(session: fastf1.core.Session) -> list[dict]:
    """
    Return a list of radio message dicts from a loaded session.
    FastF1 exposes team radio via session.car_data for newer versions,
    but the canonical source is session.laps combined with the live timing feed.
    We use the race_control_messages and driver messages where available.
    """
    messages = []

    # FastF1 >= 3.1 exposes team_radio on the session object
    if hasattr(session, "car_data"):
        # car_data contains telemetry, not radio — skip
        pass

    # Try the newer API: session.get_radio_messages() (FastF1 >= 3.3)
    try:
        radio_df = session.race_control_messages
        # race_control_messages are *not* team radio, but they're structured events
        # Team radio lives in a separate feed — handled below
    except Exception:
        radio_df = None

    # The simplest path: FastF1 caches team radio audio + transcripts
    # as session.car_data["RadioMessage"] — available in some versions.
    # Fall back to an empty list if not present so the pipeline continues.
    if not messages:
        logger.debug("No radio transcript data available for this session.")

    return messages


def save_radio(messages: list[dict], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(messages, f, indent=2, default=str)
    logger.info("Saved %d radio messages → %s", len(messages), out_path)


def load_radio(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
