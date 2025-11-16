"""Typer CLI for ingestion, indexing, and querying."""
from __future__ import annotations

from pathlib import Path

import typer

from .embeddings import embedding_service
from .evaluation import run_batch_evaluation
from .graph import run_graph
from .ingestion import IngestionPipeline

app = typer.Typer(help="CLI for the Intelligent Document Q&A system")
pipeline = IngestionPipeline()


@app.command()
def ingest(path: Path, uploaded_by: str = "cli") -> None:
    """Ingest a single document from the filesystem."""

    chunks = pipeline.ingest(path, uploaded_by)
    embedding_service.index_chunks(chunks)
    typer.echo(f"Indexed {len(chunks)} chunks for document {path.name}")


@app.command()
def ask(question: str) -> None:
    """Ask a question via the LangGraph pipeline."""

    response = run_graph(question)
    typer.echo(response)


@app.command()
def evaluate() -> None:
    """Run the predefined evaluation suite."""

    results = run_batch_evaluation()
    typer.echo(f"Completed evaluation for {len(results)} queries")


if __name__ == "__main__":
    app()
