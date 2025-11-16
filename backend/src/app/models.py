"""Core domain models for the RAG system."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class DocumentMetadata:
    document_id: str
    source_name: str
    mime_type: str
    language: str
    checksum: str
    uploaded_by: str
    page_count: int
    created_at: datetime
    version: int = 1


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    section: str | None
    page_number: int | None
    chunk_index: int
    token_count: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class Citation:
    document_id: str
    source_name: str
    page_number: int | None
    section: str | None


@dataclass(slots=True)
class QAResponse:
    answer: str
    citations: list[Citation]
    evidence: list[str]
    confidence: float
    not_found_reason: str | None = None
    latency_ms: float | None = None


@dataclass(slots=True)
class EvaluationResult:
    query_id: str
    exact_match: float
    semantic_similarity: float
    grounding_score: float
    latency_stats: dict[str, float]
    token_usage: dict[str, int]
