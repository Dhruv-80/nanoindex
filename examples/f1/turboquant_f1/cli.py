"""tq-f1 CLI — ingestion, chunking, embedding, index building, and search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="tq-f1", help="TurboQuant F1 semantic search CLI")
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=level)


# ------------------------------------------------------------------
# ingest
# ------------------------------------------------------------------

@app.command()
def ingest(
    year: int = typer.Option(2023, help="Season year"),
    round_num: int = typer.Option(1, help="Race round number"),
    session: str = typer.Option("R", help="Session identifier: FP1/FP2/FP3/Q/R"),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Output directory for raw Parquet"),
    cache_dir: Path = typer.Option(Path("data/cache"), help="FastF1 cache directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download and cache raw FastF1 data for a single session."""
    _setup_logging(verbose)
    from .ingestion.fastf1_loader import load_session, extract_session_data, save_session_data
    from .ingestion.event_extractor import extract_race_control_events, extract_pit_stops, save_events

    console.print(f"[bold]Loading[/bold] {year} round {round_num} {session} …")
    sess = load_session(year, round_num, session, cache_dir)
    if sess is None:
        console.print("[red]Session not available.[/red]")
        raise typer.Exit(1)

    event_key = sess.event["EventName"].lower().replace(" ", "_")
    out_dir = raw_dir / str(year) / event_key / session.lower()

    data = extract_session_data(sess)
    save_session_data(data, out_dir)

    events = extract_race_control_events(sess)
    if "laps" in data:
        events += extract_pit_stops(data["laps"])
    save_events(events, out_dir / "events.json")

    console.print(f"[green]Done.[/green] Saved to {out_dir}")


# ------------------------------------------------------------------
# chunk
# ------------------------------------------------------------------

@app.command()
def chunk(
    data_dir: Path = typer.Option(..., help="Path to a session's raw Parquet directory"),
    year: int = typer.Option(2023),
    race: str = typer.Option(..., help="Race name slug"),
    session: str = typer.Option("R"),
    window: float = typer.Option(5.0, help="Telemetry window size in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Chunk a session's raw data into text narratives."""
    _setup_logging(verbose)
    import json
    from .ingestion.fastf1_loader import load_session_data
    from .ingestion.telemetry_parser import normalize_telemetry
    from .chunking.telemetry_chunker import chunk_driver_telemetry
    from .chunking.lap_summarizer import summarize_laps
    from .chunking.event_chunker import chunk_events
    from .ingestion.event_extractor import load_events

    data = load_session_data(data_dir)
    chunks = []

    telem = data.get("telemetry")
    laps = data.get("laps")

    driver_map: dict = {}
    if laps is not None and "DriverNumber" in laps.columns and "Driver" in laps.columns:
        driver_map = dict(zip(laps["DriverNumber"].astype(str), laps["Driver"]))

    if telem is not None:
        telem = normalize_telemetry(telem)
        for drv in telem["DriverNumber"].unique():
            chunks += chunk_driver_telemetry(telem, drv, year, race, session, window, laps_df=laps, driver_map=driver_map)

    if laps is not None:
        chunks += summarize_laps(laps, year, race, session)

    events_path = data_dir / "events.json"
    if events_path.exists():
        events = load_events(events_path)
        chunks += chunk_events(events, year, race, session)

    out_path = data_dir / "chunks.json"
    with open(out_path, "w") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)
    console.print(f"[green]{len(chunks)} chunks[/green] → {out_path}")


# ------------------------------------------------------------------
# embed
# ------------------------------------------------------------------

@app.command()
def embed(
    chunks_path: Path = typer.Option(..., help="Path to chunks.json"),
    out_dir: Optional[Path] = typer.Option(None, help="Output directory (defaults to same dir as chunks.json)"),
    model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
    batch_size: int = typer.Option(256),
    device: str = typer.Option("cpu"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Embed chunks.json into float32 vectors and save as embeddings.npz."""
    _setup_logging(verbose)
    import json
    import numpy as np
    from .embedding.text_embedder import TextEmbedder

    dest = out_dir or chunks_path.parent

    console.print(f"Loading chunks from [bold]{chunks_path}[/bold] …")
    with open(chunks_path) as f:
        chunks_raw = json.load(f)

    console.print(f"  {len(chunks_raw)} chunks loaded.")
    console.print(f"Loading model [bold]{model}[/bold] on {device} …")
    embedder = TextEmbedder(model_name=model, device=device)

    texts = [c["text"] for c in chunks_raw]

    console.print("Embedding …")
    embeddings = embedder.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    out_path = Path(dest) / "embeddings.npz"
    np.savez_compressed(out_path, embeddings=embeddings)

    size_mb = out_path.stat().st_size / 1e6
    console.print(
        f"[green]Saved[/green] {embeddings.shape[0]} × {embeddings.shape[1]} "
        f"float32 embeddings → {out_path} ({size_mb:.1f} MB)"
    )


# ------------------------------------------------------------------
# build
# ------------------------------------------------------------------

@app.command()
def build(
    chunks_path: Path = typer.Option(..., help="Path to chunks.json"),
    embeddings_path: Path = typer.Option(..., help="Path to embeddings.npz"),
    output: Path = typer.Option(Path("data/index/index"), help="Output path prefix (no extension)"),
    bits: int = typer.Option(3, help="Bits per quantized angle (2–8)"),
    qjl_m: int = typer.Option(64, help="QJL projection dimensions"),
    seed: int = typer.Option(42),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compress embeddings into a TurboQuant index and save to disk."""
    _setup_logging(verbose)
    import json
    import numpy as np
    from .index.turbo_index import TurboQuantIndex

    console.print(f"Loading chunks from [bold]{chunks_path}[/bold] …")
    with open(chunks_path) as f:
        chunks = json.load(f)

    console.print(f"Loading embeddings from [bold]{embeddings_path}[/bold] …")
    data = np.load(embeddings_path)
    embeddings = data["embeddings"].astype(np.float32)

    if len(embeddings) != len(chunks):
        # embeddings.npz might not have ids aligned — just use positional alignment
        console.print(f"[yellow]Warning: {len(embeddings)} embeddings vs {len(chunks)} chunks[/yellow]")
        n = min(len(embeddings), len(chunks))
        embeddings, chunks = embeddings[:n], chunks[:n]

    dim = embeddings.shape[1]
    console.print(f"  {len(chunks)} vectors, dim={dim}, bits={bits}, qjl_m={qjl_m}")

    idx = TurboQuantIndex(dim=dim, bits=bits, qjl_m=qjl_m, seed=seed)
    idx.add(embeddings, chunks)

    idx.save(output)

    stats = idx.stats()
    console.print(f"[green]Index saved[/green] → {output}.npz")
    console.print(f"  Vectors:           {stats['n_vectors']:,}")
    console.print(f"  Chunk types:       {stats['chunk_types']}")
    console.print(f"  Memory:            {stats['memory_mb']} MB")
    console.print(f"  Compression ratio: {stats['compression_ratio']}×")


# ------------------------------------------------------------------
# search
# ------------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    index_path: Path = typer.Option(Path("data/index/index"), help="Index path prefix"),
    top_k: int = typer.Option(10, "-k", help="Number of results"),
    model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
    driver: Optional[str] = typer.Option(None, help="Filter by driver code (e.g. VER)"),
    race: Optional[str] = typer.Option(None, help="Filter by race slug"),
    session: Optional[str] = typer.Option(None, help="Filter by session (R, Q, FP1 …)"),
    chunk_type: Optional[str] = typer.Option(None, help="Filter by chunk type"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Semantic search over an F1 TurboQuant index."""
    _setup_logging(verbose)
    import numpy as np
    from .index.turbo_index import TurboQuantIndex
    from .embedding.text_embedder import TextEmbedder

    console.print(f"Loading index from [bold]{index_path}[/bold] …")
    idx = TurboQuantIndex.load(index_path)
    console.print(f"  {idx.n_vectors:,} vectors loaded.")

    # Trigger Numba JIT compilation now (cached after first run) so search latency
    # reflects steady-state performance rather than compile time.
    idx.search(np.zeros(idx.dim, dtype=np.float32), k=1)

    console.print(f"Embedding query with [bold]{model}[/bold] …")
    embedder = TextEmbedder(model_name=model)
    query_vec = embedder.embed(query)

    filters: dict = {}
    if driver:      filters["driver"]     = driver.upper()
    if race:        filters["race"]       = race
    if session:     filters["session"]    = session.upper()
    if chunk_type:  filters["chunk_type"] = chunk_type

    results = idx.search(query_vec, k=top_k, filters=filters or None)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f'Results for "{query}"', show_lines=True)
    table.add_column("#",        width=3,  style="dim")
    table.add_column("Score",    width=6)
    table.add_column("Type",     width=11)
    table.add_column("Driver",   width=6)
    table.add_column("Lap",      width=4)
    table.add_column("Race / Session", width=22)
    table.add_column("Text", no_wrap=False)

    for r in results:
        table.add_row(
            str(r.rank + 1),
            f"{r.score:.3f}",
            r.chunk_type,
            r.driver,
            str(r.lap),
            f"{r.race} / {r.session}",
            r.text[:200],
        )

    console.print(table)


# ------------------------------------------------------------------
# index-info
# ------------------------------------------------------------------

@app.command(name="index-info")
def index_info(
    index_path: Path = typer.Option(Path("data/index/index"), help="Index path prefix"),
) -> None:
    """Show statistics for a saved index."""
    from .index.turbo_index import TurboQuantIndex
    idx = TurboQuantIndex.load(index_path)
    stats = idx.stats()

    console.print(f"\n[bold]TurboQuant Index[/bold] — {index_path}")
    for k, v in stats.items():
        console.print(f"  {k:<22} {v}")


# ------------------------------------------------------------------
# bench
# ------------------------------------------------------------------

@app.command()
def bench(
    embeddings_path: Path = typer.Option(
        Path("data/raw/2023/bahrain_grand_prix/r/embeddings.npz"),
        help="Path to embeddings.npz",
    ),
    n_queries: int = typer.Option(200, help="Number of query vectors to use"),
    bits: int = typer.Option(3, help="TurboQuant bits (2–8)"),
    qjl_m: int = typer.Option(64, help="QJL projection dimensions"),
    output: Path = typer.Option(Path("benchmarks/reports/results.json"), help="JSON output path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run compression, recall, and latency benchmarks vs Faiss baselines."""
    _setup_logging(verbose)
    import subprocess, sys
    result = subprocess.run(
        [
            sys.executable, "benchmarks/run_benchmarks.py",
            "--embeddings", str(embeddings_path),
            "--n-queries", str(n_queries),
            "--bits", str(bits),
            "--qjl-m", str(qjl_m),
            "--output", str(output),
        ],
        check=False,
    )
    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
