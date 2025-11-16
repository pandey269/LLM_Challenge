# Solution Overview

## Technology Selections (per provided instruction set)
- **LangChain + LangGraph**: Implements the agentic/self-reflective RAG workflow inspired by the LangChain blog and LangGraph tutorials.
- **Nomic Embeddings (`nomic-ai/nomic-embed-text-v1.5`)**: Used for dense vectorization via HuggingFace; aligns with the provided embedding model reference.
- **Chroma DB (local)**: Persistent vector store for embeddings, matching the "Chroma DB local tutorial" guidance.
- **LLM Providers**: OpenAI GPT-4o-mini is the default runtime for accuracy and latency; toggling `RAG_MODEL__LLM_PROVIDER=ollama` switches to the requested local `mistral-7b-instruct` build.

## Pipeline Recap
1. **Ingestion**: LangChain loaders parse PDF/DOCX/PPT/TXT, normalize text, and chunk with overlap. Metadata (doc id, section, page) captured and persisted.
2. **Embedding & Indexing**: Nomic embeddings stored in Chroma with metadata for filtering/re-indexing; embed cache keyed by SHA256.
3. **Hybrid Retrieval**: Dense (Chroma) + sparse (BM25) retrieval via LangChain, deduplicated and reranked (LongContextReorder). Supports metadata filters and duplicate guarding.
4. **Agentic Self-RAG (LangGraph)**:
   - `retrieve` node fetches context.
   - `draft` node builds grounded answer with citations.
   - `reflect` node critiques groundedness (per "Self-Reflective RAG" blog) and can request another retrieval loop if score < threshold.
5. **LLM Answering**: ChatOpenAI (GPT-4o-mini) enforces structured JSON outputs with citations and an explicit "I don't know" branch; Ollama Mistral remains a drop-in alternative for offline inference.
6. **Observability**: `observability.py` wires OTLP tracing + Prometheus histograms/counters so latency, token usage, and chunk hits are recorded.
7. **Evaluation**: RAGAS-based harness (`src/app/evaluation.py`) executes the README test cases and captures latency stats (p50/p95) plus cache/concurrency comparisons.
8. **Deployment**: Dockerfile + docker-compose for local stack (backend + frontend, optional Jaeger/Prometheus/Grafana sidecars).

## Deliverables Checklist
- Source code with modular structure under `src/app` and tests in `src/tests` ✅
- Architecture diagram & documentation (`docs/architecture.md` + `docs/architecture.png`) ✅
- README + implementation guide aligned with the current stack ✅
- Evaluation report via `run_batch_evaluation` outputs to `data/evaluations/latest.json` ✅
- Sample queries with answers/citations under `sample_data/sample_queries.json` ✅
- Streamlit UI delivered in `frontend/` (bonus scope) ✅

## Next Steps
- Populate `sample_data/documents/` with example PDFs to run the full evaluation plan.
- Pull `mistral-7b-instruct` in Ollama if you want to operate without OpenAI.
- Optionally extend FastAPI with WebSocket streaming, richer dashboards, or multilingual UX polish.
