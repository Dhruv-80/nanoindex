"""Metadata filtering — returns a boolean mask over the index."""

from __future__ import annotations

import numpy as np


def apply_filters(metadata: list[dict], filters: dict | None) -> np.ndarray | None:
    """
    Build a boolean mask (True = keep) from a filter dict.

    Filter semantics:
      key              str | list[str]   exact match on metadata[key] (case-insensitive)
      key_min          numeric           metadata[key] >= value
      key_max          numeric           metadata[key] <= value

    Examples:
      {"driver": "VER"}
      {"driver": ["VER", "HAM"]}
      {"year": 2023, "lap_min": 10, "lap_max": 20}
      {"chunk_type": "event", "score_min": 0.8}

    Returns None if filters is None/empty (no filtering needed).
    """
    if not filters:
        return None

    N = len(metadata)
    mask = np.ones(N, dtype=bool)

    # Separate range filters (key_min / key_max) from equality filters
    range_min: dict[str, float] = {}
    range_max: dict[str, float] = {}
    equality: dict[str, object] = {}

    for k, v in filters.items():
        if k.endswith("_min"):
            range_min[k[:-4]] = v
        elif k.endswith("_max"):
            range_max[k[:-4]] = v
        else:
            equality[k] = v

    def _match(value, criterion) -> bool:
        if isinstance(criterion, list):
            return str(value).lower() in [str(c).lower() for c in criterion]
        return str(value).lower() == str(criterion).lower()

    for i, m in enumerate(metadata):
        if not mask[i]:
            continue

        for field, criterion in equality.items():
            if not _match(m.get(field, ""), criterion):
                mask[i] = False
                break

        if not mask[i]:
            continue

        for field, bound in range_min.items():
            val = m.get(field)
            if val is None or val < bound:
                mask[i] = False
                break

        if not mask[i]:
            continue

        for field, bound in range_max.items():
            val = m.get(field)
            if val is None or val > bound:
                mask[i] = False
                break

    return mask
