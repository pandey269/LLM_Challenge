"""LangGraph definition for self-reflective RAG."""
from __future__ import annotations

import json
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import settings
from .llm import llm_service
from .models import Citation, QAResponse, RetrievedChunk
from .retrieval import hybrid_retriever


class GraphState(TypedDict, total=False):
    question: str
    retrieved: list[RetrievedChunk]
    answer: QAResponse
    reflections: list[dict]
    attempts: int


def format_context(chunks: list[RetrievedChunk]) -> str:
    formatted = []
    for chunk in chunks:
        citation = f"Doc {chunk.chunk.metadata.get('source_name','unknown')} p.{chunk.chunk.page_number}"
        formatted.append(f"[{chunk.chunk.chunk_id}] {citation}:\n{chunk.chunk.text}")
    return "\n\n".join(formatted)


def _normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _normalize_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            mapping = {"low": 0.3, "medium": 0.5, "high": 0.8, "very high": 0.9}
            return mapping.get(value.lower().strip(), 0.5)
    return 0.5


_SOURCE_NAME_CACHE: dict[str, str] = {}


def _resolve_source_name(document_id: str | None, default: str | None = None) -> str | None:
    """Resolve the human-friendly file name for a document ID."""
    if default:
        return default
    if not document_id:
        return None
    cached = _SOURCE_NAME_CACHE.get(document_id)
    if cached:
        return cached
    metadata_dir = settings.paths.data_dir / "metadata"
    meta_path = metadata_dir / f"{document_id}.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source_name = data.get("source_name")
    if source_name:
        _SOURCE_NAME_CACHE[document_id] = source_name
    return source_name


def _normalize_citations(raw_citations: Any, retrieved: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    items = raw_citations if isinstance(raw_citations, list) else ([raw_citations] if raw_citations else [])
    chunk_by_id = {item.chunk.chunk_id: item.chunk for item in retrieved}
    for entry in items:
        document_id: str | None = None
        source_name: str | None = None
        page_number: int | None = None
        section: str | None = None
        chunk = None

        if isinstance(entry, dict):
            chunk = chunk_by_id.get(entry.get("chunk_id"))
            document_id = entry.get("document_id") or entry.get("doc_id") or (chunk.document_id if chunk else None)
            source_name = (
                entry.get("source_name")
                or entry.get("doc_name")
                or entry.get("document_name")
                or (chunk.metadata.get("source_name") if chunk else None)
            )
            page_raw = entry.get("page_number") or entry.get("page")
            try:
                page_number = int(page_raw) if page_raw is not None else None
            except (TypeError, ValueError):
                page_number = None
            if page_number is None and chunk:
                page_number = chunk.page_number
            section = entry.get("section") or (chunk.section if chunk else None)
        elif isinstance(entry, str):
            chunk = chunk_by_id.get(entry)
            if chunk:
                document_id = chunk.document_id
                source_name = chunk.metadata.get("source_name", chunk.document_id)
                page_number = chunk.page_number
                section = chunk.section
            else:
                document_id = entry
                source_name = entry

        if chunk and not source_name:
            source_name = chunk.metadata.get("source_name") or chunk.metadata.get("source")

        if document_id:
            citations.append(
                Citation(
                    document_id=document_id,
                    source_name=_resolve_source_name(document_id, source_name) or document_id,
                    page_number=page_number,
                    section=section,
                )
            )

    if not citations and retrieved:
        chunk = retrieved[0].chunk
        citations.append(
            Citation(
                document_id=chunk.document_id,
                source_name=_resolve_source_name(chunk.document_id, chunk.metadata.get("source_name")) or chunk.document_id,
                page_number=chunk.page_number,
                section=chunk.section,
            )
        )
    return citations


def retrieve_node(state: GraphState) -> GraphState:
    retrieved = hybrid_retriever.retrieve(state["question"])
    return {**state, "retrieved": retrieved}


def draft_node(state: GraphState) -> GraphState:
    context = format_context(state["retrieved"])
    raw = llm_service.draft(state["question"], context)
    response = QAResponse(
        answer=_normalize_answer(raw.get("answer") or raw.get("text")),
        citations=_normalize_citations(raw.get("citations"), state["retrieved"]),
        evidence=_normalize_evidence(raw.get("evidence", [])),
        confidence=_normalize_confidence(raw.get("confidence", 0.5)),
        not_found_reason=raw.get("not_found_reason"),
    )
    attempts = state.get("attempts", 0) + 1
    return {**state, "answer": response, "attempts": attempts}


def reflect_node(state: GraphState) -> GraphState:
    context = format_context(state["retrieved"])
    report = llm_service.reflect(state["question"], state["answer"].answer, context)
    reflections = state.get("reflections", []) + [report]
    return {**state, "reflections": reflections}


def needs_reflection(state: GraphState) -> str:
    reflections = state.get("reflections", [])
    if not reflections:
        return "reflect"
    latest = reflections[-1]
    score = float(latest.get("score", latest.get("confidence", 0)))
    if score >= settings.rag.reflection_threshold or state.get("attempts", 0) >= settings.rag.max_reflection_loops:
        return END
    return "retrieve"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("reflect", reflect_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "reflect")
    graph.add_conditional_edges("reflect", needs_reflection)
    graph.add_edge("draft", END)
    return graph


def run_graph(question: str) -> QAResponse:
    use_cache = getattr(getattr(settings, "cache", None), "enabled", False)
    cache_key = f"qa:{question}"

    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    graph = build_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    state = {"question": question}
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = compiled.invoke(state, config=config)
    if use_cache:
        _CACHE[cache_key] = final_state["answer"]
    return final_state["answer"]


_CACHE: dict[str, QAResponse] = {}
