"""Convert race control events and derived events into TelemetryChunks."""

from __future__ import annotations

import hashlib

from .schemas import TelemetryChunk


def chunk_events(
    events: list[dict],
    year: int,
    race: str,
    session: str,
) -> list[TelemetryChunk]:
    chunks = []
    for evt in events:
        evt_type = evt.get("event_type") or evt.get("Category", "event")
        driver = str(evt.get("driver") or evt.get("Driver", "ALL"))
        lap = int(evt["lap"]) if evt.get("lap") is not None else 0
        t_start = float(evt.get("session_seconds") or 0.0)

        # Build a plain-text description
        msg = evt.get("Message") or evt.get("msg") or ""
        parts = [f"[{session} {race} {year}] {evt_type.upper()}"]
        if driver != "ALL":
            parts.append(f"Driver: {driver}")
        if lap:
            parts.append(f"Lap {lap}")
        if msg:
            parts.append(msg)
        text = ". ".join(parts) + "."

        chunk_id = hashlib.md5(
            f"{year}:{race}:{session}:{evt_type}:{driver}:{t_start:.2f}".encode()
        ).hexdigest()[:16]

        chunks.append(TelemetryChunk(
            id=chunk_id,
            text=text,
            chunk_type="event",
            year=year,
            race=race,
            session=session,
            driver=driver,
            lap=lap,
            timestamp_start=t_start,
            timestamp_end=t_start,
        ))
    return chunks
