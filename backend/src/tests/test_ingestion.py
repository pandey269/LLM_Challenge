from pathlib import Path

from src.app.ingestion import IngestionPipeline


def test_ingestion_chunking(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Line one.\nLine two.\nLine three.")
    pipeline = IngestionPipeline()
    chunks = pipeline.ingest(sample, uploaded_by="test")
    assert chunks, "Chunks should be created"
    assert chunks[0].document_id
    assert chunks[0].text
