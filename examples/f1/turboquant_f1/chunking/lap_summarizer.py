"""Generate one holistic lap-summary chunk per driver per lap."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import pandas as pd

from .schemas import TelemetryChunk

logger = logging.getLogger(__name__)


def summarize_laps(
    laps: pd.DataFrame,
    year: int,
    race: str,
    session: str,
) -> list[TelemetryChunk]:
    chunks = []
    required = {"LapNumber", "Driver"}
    if not required.issubset(laps.columns):
        logger.warning("Laps DataFrame missing required columns; skipping lap summaries.")
        return chunks

    for _, row in laps.iterrows():
        driver = str(row["Driver"])
        lap = int(row["LapNumber"])

        # Build narrative
        parts = [f"{driver} lap {lap} in {session} at {race} {year}"]

        lap_time = row.get("LapTime")
        if pd.notna(lap_time):
            lt_s = pd.to_timedelta(lap_time).total_seconds()
            parts.append(f"lap time {lt_s:.3f}s")

        compound = row.get("Compound")
        if compound and pd.notna(compound):
            parts.append(f"{compound} tires")

        position = row.get("Position")
        if position is not None and pd.notna(position):
            parts.append(f"P{int(position)}")

        s1 = row.get("Sector1Time")
        s2 = row.get("Sector2Time")
        s3 = row.get("Sector3Time")
        if all(pd.notna(x) for x in [s1, s2, s3]):
            s1s = pd.to_timedelta(s1).total_seconds()
            s2s = pd.to_timedelta(s2).total_seconds()
            s3s = pd.to_timedelta(s3).total_seconds()
            parts.append(f"sectors {s1s:.3f} / {s2s:.3f} / {s3s:.3f}s")

        gap = row.get("GapToLeader") or row.get("Gap")
        if gap is not None and pd.notna(gap):
            parts.append(f"gap to leader {gap}")

        text = ". ".join(parts) + "."

        chunk_id = hashlib.md5(
            f"{year}:{race}:{session}:{driver}:lap:{lap}".encode()
        ).hexdigest()[:16]

        chunks.append(TelemetryChunk(
            id=chunk_id,
            text=text,
            chunk_type="lap_summary",
            year=year,
            race=race,
            session=session,
            driver=driver,
            lap=lap,
            timestamp_start=0.0,
            timestamp_end=0.0,
            compound=str(compound) if compound and pd.notna(compound) else None,
            position=int(position) if position is not None and pd.notna(position) else None,
        ))

    return chunks
