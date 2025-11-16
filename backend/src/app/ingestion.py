"""Document ingestion and chunking pipeline."""
from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import langchain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .models import Chunk, DocumentMetadata

langchain.verbose = False

NON_ASCII_THRESHOLD = 128


class IngestionPipeline:
    """Handles parsing, normalization, and chunking of uploaded documents."""

    SUPPORTED_MIME_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
        "application/msword",
        "text/plain",
        "text/csv",
    }

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag.chunk_size_tokens,
            chunk_overlap=settings.rag.chunk_overlap_tokens,
            separators=["\n\n", "\n", ". ", " "],
        )

    def load(self, path: Path) -> list[Document]:
        """Load the document with an appropriate LangChain loader."""

        mime_type, _ = mimetypes.guess_type(path)
        loader: object | None = None
        if mime_type == "application/pdf":
            loader = PyPDFLoader(str(path))
        elif mime_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }:
            loader = UnstructuredWordDocumentLoader(str(path), mode="elements")
        elif mime_type in {
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.ms-powerpoint",
        }:
            loader = UnstructuredPowerPointLoader(str(path))
        elif mime_type == "text/csv":
            loader = CSVLoader(file_path=str(path))
        else:
            loader = TextLoader(str(path), autodetect_encoding=True)

        docs = loader.load()
        html_docs = [doc for doc in docs if doc.metadata.get("source", "").endswith(".html")]
        if html_docs:
            transformer = Html2TextTransformer()
            docs = list(transformer.transform_documents(docs))
        return docs

    def chunk(self, docs: Iterable[Document], metadata: DocumentMetadata) -> list[Chunk]:
        """Split documents into overlapping chunks with metadata."""

        lc_docs = [Document(page_content=doc.page_content, metadata={**doc.metadata}) for doc in docs]
        split_docs = self.splitter.split_documents(lc_docs)
        chunks: list[Chunk] = []
        for idx, split_doc in enumerate(split_docs):
            chunks.append(
                Chunk(
                    chunk_id=f"{metadata.document_id}:{idx}",
                    document_id=metadata.document_id,
                    text=split_doc.page_content.strip(),
                    section=split_doc.metadata.get("section") or split_doc.metadata.get("heading"),
                    page_number=split_doc.metadata.get("page") or split_doc.metadata.get("page_number"),
                    chunk_index=idx,
                    token_count=len(split_doc.page_content.split()),
                    metadata={**split_doc.metadata, "source_name": metadata.source_name},
                )
            )
        return chunks

    def ingest(self, file_path: Path, uploaded_by: str = "anonymous") -> list[Chunk]:
        """Full ingestion pipeline for a single file."""

        docs = self.load(file_path)
        checksum = self._checksum(file_path)
        metadata = DocumentMetadata(
            document_id=checksum[:12],
            source_name=file_path.name,
            mime_type=mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
            language=self._detect_language(docs),
            checksum=checksum,
            uploaded_by=uploaded_by,
            page_count=len(docs),
            created_at=datetime.now(tz=timezone.utc),
        )
        chunks = self.chunk(docs, metadata)
        self._persist_metadata(metadata, chunks)
        return chunks

    @staticmethod
    def _checksum(file_path: Path) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _detect_language(docs: Iterable[Document]) -> str:
        sample_text = " ".join(doc.page_content for doc in docs[:3]) if isinstance(docs, list) else ""
        if any(ord(ch) > NON_ASCII_THRESHOLD for ch in sample_text):
            return "multilingual"
        return "en"

    @staticmethod
    def _persist_metadata(metadata: DocumentMetadata, chunks: list[Chunk]) -> None:
        metadata_dir = settings.paths.data_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_path = metadata_dir / f"{metadata.document_id}.json"
        chunk_path = metadata_dir / f"{metadata.document_id}_chunks.jsonl"
        meta_path.write_text(
            "{\n"
            f'  "document_id": "{metadata.document_id}",\n'
            f'  "source_name": "{metadata.source_name}",\n'
            f'  "mime_type": "{metadata.mime_type}",\n'
            f'  "language": "{metadata.language}",\n'
            f'  "checksum": "{metadata.checksum}",\n'
            f'  "uploaded_by": "{metadata.uploaded_by}",\n'
            f'  "page_count": {metadata.page_count},\n'
            f'  "created_at": "{metadata.created_at.isoformat()}"\n'
            "}\n"
        )

        with open(chunk_path, "w", encoding="utf-8") as handle:
            for chunk in chunks:
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
