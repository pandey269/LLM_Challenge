"""Embedding utilities using the Nomic model + Chroma vector store."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .config import settings
from .models import Chunk


class EmbeddingService:
    """Handles embedding generation and vector store operations."""

    def __init__(self) -> None:
        self.embedder = HuggingFaceEmbeddings(
            model_name=settings.model.embed_model,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_dir = Path(settings.paths.vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = "document_chunks"
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=str(self.vector_dir),
        )

    def index_chunks(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        texts = [chunk.text for chunk in chunk_list]
        metadatas: list[dict[str, Any]] = []
        for chunk in chunk_list:
            metadata = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "page_number": chunk.page_number,
                "source_name": (chunk.metadata or {}).get("source_name"),
            }
            metadata.update(self._filter_metadata(chunk.metadata))
            metadatas.append(metadata)
        ids = [chunk.chunk_id for chunk in chunk_list]

        existing_ids: set[str] = set()
        try:
            response = self.store._collection.get(ids=ids, include=[])  # type: ignore[attr-defined]
            existing_ids = set(response.get("ids", []))
        except Exception:  # noqa: BLE001
            pass

        payload = [
            (text, meta, chunk_id)
            for text, meta, chunk_id in zip(texts, metadatas, ids, strict=True)
            if chunk_id not in existing_ids
        ]

        if not payload:
            return

        new_texts, new_metadatas, new_ids = zip(*payload, strict=True)
        self.store.add_texts(texts=list(new_texts), metadatas=list(new_metadatas), ids=list(new_ids))
        self.store.persist()

    @staticmethod
    def _filter_metadata(raw: dict[str, Any] | None) -> dict[str, Any]:
        allowed_types = (str, int, float, bool)
        cleaned: dict[str, Any] = {}
        if not raw:
            return cleaned
        for key, value in raw.items():
            if value is None or isinstance(value, allowed_types):
                cleaned[key] = value
        return cleaned

    def delete_document(self, document_id: str) -> None:
        if not self.store._collection.count():  # pylint: disable=protected-access
            return
        ids = self.store._collection.get(where={"document_id": document_id}).get("ids", [])  # noqa: SLF001
        if ids:
            self.store._collection.delete(ids)  # noqa: SLF001
            self.store.persist()

    def similarity_search(self, query: str, k: int) -> list[dict]:
        return self.store.similarity_search_with_score(query, k=k)


embedding_service = EmbeddingService()
