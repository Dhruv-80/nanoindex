"""Pre-search metadata filtering — returns a boolean mask over the index."""

from __future__ import annotations

import numpy as np


def apply_filters(metadata: list[dict], filters: dict | None) -> np.ndarray | None:
    """
    Build a boolean mask (True = keep) from a filter dict.

    Supported keys:
      driver       str | list[str]   three-letter driver code(s)
      race         str | list[str]   race slug(s)
      session      str | list[str]   session identifier(s)
      chunk_type   str | list[str]   "telemetry" | "event" | "lap_summary"
      lap_min      int               inclusive lower bound on lap number
      lap_max      int               inclusive upper bound on lap number

    Returns None if filters is empty/None (means "no filter, keep all").
    """
    if not filters:
        return None

    N = len(metadata)
    mask = np.ones(N, dtype=bool)

    def _match(value, criterion) -> bool:
        if isinstance(criterion, list):
            return str(value).lower() in [str(c).lower() for c in criterion]
        return str(value).lower() == str(criterion).lower()

    for i, m in enumerate(metadata):
        if not mask[i]:
            continue

        for field in ("driver", "race", "session", "chunk_type"):
            if field in filters and not _match(m.get(field, ""), filters[field]):
                mask[i] = False
                break

        if not mask[i]:
            continue

        if "lap_min" in filters and (m.get("lap") or 0) < filters["lap_min"]:
            mask[i] = False
        if "lap_max" in filters and (m.get("lap") or 0) > filters["lap_max"]:
            mask[i] = False

    return mask
