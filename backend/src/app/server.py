"""FastAPI server exposing ingestion and question answering endpoints."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .embeddings import embedding_service
from .graph import run_graph
from .ingestion import IngestionPipeline
from .models import QAResponse

app = FastAPI(title="Intelligent Document Q&A", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestion_pipeline = IngestionPipeline()


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    filters: dict | None = Field(default=None, description="Metadata filters")


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    evidence: list[str]
    confidence: float
    latency_ms: float
    not_found_reason: str | None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", status_code=201)
async def ingest(file: UploadFile = File(...), uploaded_by: str = "anonymous") -> dict:  # noqa: B008
    if file.content_type not in IngestionPipeline.SUPPORTED_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")
    uploads_dir = settings.paths.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = uploads_dir / file.filename
    with open(destination, "wb") as buffer:
        buffer.write(await file.read())
    chunks = await asyncio.to_thread(ingestion_pipeline.ingest, destination, uploaded_by)
    await asyncio.to_thread(embedding_service.index_chunks, chunks)
    return {"document_id": chunks[0].document_id if chunks else None, "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    start = perf_counter()
    answer: QAResponse = await asyncio.to_thread(run_graph, payload.question)
    latency = (perf_counter() - start) * 1000
    return QueryResponse(
        answer=answer.answer,
        citations=[asdict(citation) if hasattr(citation, "document_id") else citation for citation in answer.citations],
        evidence=answer.evidence,
        confidence=answer.confidence,
        latency_ms=latency,
        not_found_reason=answer.not_found_reason,
    )


__all__ = ["app"]
