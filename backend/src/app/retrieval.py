"""Hybrid retrieval utilities with deduplication and reranking."""
from __future__ import annotations

import json

from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import settings
from .embeddings import embedding_service
from .models import Chunk, RetrievedChunk


def _load_all_chunks() -> list[Chunk]:
    metadata_dir = settings.paths.data_dir / "metadata"
    if not metadata_dir.exists():
        return []
    chunks: list[Chunk] = []
    for chunk_file in metadata_dir.glob("*_chunks.jsonl"):
        with open(chunk_file, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunk = Chunk(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    text=payload["text"],
                    section=payload.get("section"),
                    page_number=payload.get("page_number"),
                    chunk_index=payload.get("chunk_index", 0),
                    token_count=payload.get("token_count", len(payload["text"].split())),
                    metadata=payload.get("metadata", {}),
                )
                chunks.append(chunk)
    return chunks


class HybridRetriever:
    """Combines dense (Chroma) and sparse (BM25) retrievers with reranking."""

    def __init__(self) -> None:
        self.dense = embedding_service.store.as_retriever(search_kwargs={"k": settings.rag.top_k_dense})
        docs = [chunk_to_document(chunk) for chunk in _load_all_chunks()]
        if docs:
            self.sparse = BM25Retriever.from_documents(docs)
            self.sparse.k = settings.rag.top_k_sparse
        else:
            self.sparse = None
        self.reorder = LongContextReorder()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        dense_docs = self.dense.invoke(query) if self.dense else []
        sparse_docs = self.sparse.invoke(query) if self.sparse else []
        weighted_docs = [
            (doc, 0.7 * (1 - idx * 0.05)) for idx, doc in enumerate(dense_docs)
        ] + [
            (doc, 0.3 * (1 - idx * 0.05)) for idx, doc in enumerate(sparse_docs)
        ]
        merged: dict[str, RetrievedChunk] = {}
        for doc, score in weighted_docs:
            chunk_id = (
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
                or doc.metadata.get("source")
                or str(len(merged))
            )
            if chunk_id in merged and merged[chunk_id].score >= score:
                continue
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc.metadata.get("document_id", "unknown"),
                text=doc.page_content,
                section=doc.metadata.get("section"),
                page_number=doc.metadata.get("page_number"),
                chunk_index=int(doc.metadata.get("chunk_index", 0)),
                token_count=len(doc.page_content.split()),
                metadata=doc.metadata,
            )
            merged[chunk_id] = RetrievedChunk(chunk=chunk, score=score)
        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        docs = [chunk_to_document(item.chunk) for item in ordered]
        reordered = self.reorder.transform_documents(docs)
        final_chunks: list[RetrievedChunk] = []
        for rank, doc in enumerate(reordered):
            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id") or f"combined-{rank}"
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc.metadata.get("document_id", "unknown"),
                text=doc.page_content,
                section=doc.metadata.get("section"),
                page_number=doc.metadata.get("page_number"),
                chunk_index=int(doc.metadata.get("chunk_index", rank)),
                token_count=len(doc.page_content.split()),
                metadata=doc.metadata,
            )
            final_chunks.append(RetrievedChunk(chunk=chunk, score=1 - (rank * 0.05)))
        return final_chunks


def chunk_to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "section": chunk.section,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            **chunk.metadata,
        },
    )


hybrid_retriever = HybridRetriever()
