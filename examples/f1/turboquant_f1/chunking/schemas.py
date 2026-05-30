"""Shared dataclass schema for all chunk types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TelemetryChunk:
    id: str
    text: str
    chunk_type: str                        # "telemetry" | "radio" | "event" | "lap_summary"
    year: int
    race: str
    session: str
    driver: str
    lap: int
    timestamp_start: float
    timestamp_end: float
    compound: Optional[str] = None
    position: Optional[int] = None
    numerical_features: Optional[list[float]] = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        if isinstance(d.get("numerical_features"), np.ndarray):
            d["numerical_features"] = d["numerical_features"].tolist()
        return d
