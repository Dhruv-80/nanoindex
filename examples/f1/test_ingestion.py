"""Unit tests for ingestion and chunking — no FastF1 network calls."""

import numpy as np
import pandas as pd
import pytest

from turboquant_f1.ingestion.telemetry_parser import normalize_telemetry, compute_window_features
from turboquant_f1.chunking.telemetry_chunker import chunk_driver_telemetry
from turboquant_f1.chunking.lap_summarizer import summarize_laps
from turboquant_f1.chunking.event_chunker import chunk_events
from turboquant_f1.chunking.schemas import TelemetryChunk


def _make_telemetry(driver: str = "1", n_seconds: int = 30) -> pd.DataFrame:
    """Synthetic telemetry: constant-ish values over n_seconds at ~10 Hz."""
    n = n_seconds * 10
    t = np.linspace(0, n_seconds, n)
    return pd.DataFrame({
        "Time": pd.to_timedelta(t, unit="s"),
        "SessionSeconds": t,
        "DriverNumber": driver,
        "LapNumber": np.ones(n, dtype=int),
        "Speed": np.clip(np.random.normal(250, 30, n), 0, 380),
        "Throttle": np.clip(np.random.normal(60, 30, n), 0, 100),
        "Brake": np.clip(np.random.normal(20, 20, n), 0, 100),
        "RPM": np.clip(np.random.normal(11000, 2000, n), 0, 18000),
        "nGear": np.random.randint(4, 8, n).astype(float),
        "DRS": np.zeros(n),
    })


# ---------------------------------------------------------------------------
# normalize_telemetry
# ---------------------------------------------------------------------------

def test_normalize_telemetry_returns_session_seconds():
    df = _make_telemetry()
    out = normalize_telemetry(df)
    assert "SessionSeconds" in out.columns
    assert out["SessionSeconds"].dtype == float


def test_normalize_telemetry_clips_speed():
    df = _make_telemetry()
    df["Speed"] = 999.0  # out of range
    out = normalize_telemetry(df)
    assert out["Speed"].max() <= 400


def test_normalize_telemetry_bool_brake():
    df = _make_telemetry()
    df["Brake"] = df["Brake"].astype(bool)
    out = normalize_telemetry(df)
    assert out["Brake"].max() <= 100.0


# ---------------------------------------------------------------------------
# compute_window_features
# ---------------------------------------------------------------------------

def test_compute_window_features_shape():
    df = _make_telemetry()
    feats = compute_window_features(df)
    # 6 channels × 5 stats = 30
    assert feats.shape == (30,)
    assert feats.dtype == np.float32


# ---------------------------------------------------------------------------
# chunk_driver_telemetry
# ---------------------------------------------------------------------------

def test_chunk_driver_telemetry_produces_chunks():
    df = _make_telemetry(driver="1", n_seconds=30)
    chunks = chunk_driver_telemetry(df, driver="1", year=2023, race="bahrain", session="R")
    assert len(chunks) > 0
    assert all(isinstance(c, TelemetryChunk) for c in chunks)


def test_chunk_driver_telemetry_window_size():
    df = _make_telemetry(driver="1", n_seconds=30)
    chunks = chunk_driver_telemetry(df, driver="1", year=2023, race="bahrain", session="R", window_seconds=5.0)
    for c in chunks:
        assert c.timestamp_end - c.timestamp_start == pytest.approx(5.0)


def test_chunk_driver_telemetry_text_nonempty():
    df = _make_telemetry(driver="1", n_seconds=10)
    chunks = chunk_driver_telemetry(df, driver="1", year=2023, race="bahrain", session="R")
    for c in chunks:
        assert len(c.text) > 10


def test_chunk_driver_telemetry_numerical_features():
    df = _make_telemetry(driver="1", n_seconds=10)
    chunks = chunk_driver_telemetry(df, driver="1", year=2023, race="bahrain", session="R")
    for c in chunks:
        assert c.numerical_features is not None
        assert len(c.numerical_features) == 30


def test_chunk_driver_telemetry_unknown_driver_returns_empty():
    df = _make_telemetry(driver="1", n_seconds=10)
    chunks = chunk_driver_telemetry(df, driver="99", year=2023, race="bahrain", session="R")
    assert chunks == []


# ---------------------------------------------------------------------------
# summarize_laps
# ---------------------------------------------------------------------------

def test_summarize_laps():
    laps = pd.DataFrame({
        "Driver": ["VER", "HAM"],
        "LapNumber": [1, 1],
        "LapTime": [pd.Timedelta(seconds=91.5), pd.Timedelta(seconds=92.1)],
        "Compound": ["SOFT", "MEDIUM"],
        "Position": [1, 2],
        "Sector1Time": [pd.Timedelta(seconds=28), pd.Timedelta(seconds=28.5)],
        "Sector2Time": [pd.Timedelta(seconds=32), pd.Timedelta(seconds=32.3)],
        "Sector3Time": [pd.Timedelta(seconds=31.5), pd.Timedelta(seconds=31.3)],
    })
    chunks = summarize_laps(laps, year=2023, race="bahrain", session="R")
    assert len(chunks) == 2
    assert all(c.chunk_type == "lap_summary" for c in chunks)
    texts = [c.text for c in chunks]
    assert any("VER" in t for t in texts)


# ---------------------------------------------------------------------------
# chunk_events
# ---------------------------------------------------------------------------

def test_chunk_events():
    events = [
        {"event_type": "lock_up", "driver": "VER", "lap": 5, "session_seconds": 300.0},
        {"event_type": "pit_stop", "driver": "HAM", "lap": 12, "session_seconds": 720.0, "compound": "HARD"},
    ]
    chunks = chunk_events(events, year=2023, race="bahrain", session="R")
    assert len(chunks) == 2
    assert all(c.chunk_type == "event" for c in chunks)


# ---------------------------------------------------------------------------
# TelemetryChunk serialisation
# ---------------------------------------------------------------------------

def test_chunk_to_dict_roundtrip():
    c = TelemetryChunk(
        id="abc123", text="test", chunk_type="telemetry",
        year=2023, race="bahrain", session="R",
        driver="VER", lap=1, timestamp_start=0.0, timestamp_end=5.0,
        numerical_features=[1.0, 2.0, 3.0],
    )
    d = c.to_dict()
    assert d["id"] == "abc123"
    assert isinstance(d["numerical_features"], list)
