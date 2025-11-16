# Implementation & Run Guide

## 1. Prerequisites
- Python 3.11+
- OpenAI API key (default LLM) **or** a local Ollama runtime with `mistral-7b-instruct` pulled
- Optional: Docker Desktop if you plan to run the stack via `docker-compose`

## 2. Install
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

## 3. Environment
Minimal `.env` example:
```
RAG_ENVIRONMENT=local
RAG_DEBUG=true

# LLMs
RAG_MODEL__LLM_PROVIDER=openai
RAG_MODEL__LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
# To use Ollama instead:
# RAG_MODEL__LLM_PROVIDER=ollama
# RAG_MODEL__LLM_MODEL=mistral-7b-instruct:latest
# RAG_MODEL__LLM_BASE_URL=http://localhost:11434

RAG_MODEL__EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5

RAG_OBSERVABILITY__ENABLE_TRACING=false
RAG_OBSERVABILITY__OTLP_ENDPOINT=http://localhost:4318

# LangChain cache (optional)
RAG_CACHE__ENABLED=false
RAG_CACHE__PATH=data/cache/lc_cache.db
```

## 4. Ingest Documents
```bash
# CLI ingestion
python -m src.app.cli ingest sample_data/rag_sample.txt
```
Or call the API directly:
```bash
curl -F "file=@sample_data/rag_sample.txt" http://localhost:8000/ingest
```
Supported types: PDF, DOCX, PPT/PPTX, TXT.

## 5. Ask Questions
```bash
python -m src.app.cli ask "What benefits of RAG are described?"
```
Via API:
```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What benefits of RAG are described?"}'
```

## 6. Streamlit UI (Frontend)
```bash
cd ../frontend
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # set BACKEND_URL if needed
streamlit run app/main.py --server.port 8501
```
The UI lets you upload docs, triggers `/ingest`, and displays `/query` responses with citations, latency, and confidence.

## 7. LangGraph Flow (Self-RAG)
- Graph defined in `src/app/graph.py` follows retrieve → draft → reflect loops per the LangGraph tutorials.
- Reflection threshold + loop cap configurable via `RAG_RAG__REFLECTION_THRESHOLD` and `RAG_RAG__MAX_REFLECTION_LOOPS`.
- Hybrid retriever (`src/app/retrieval.py`) combines Chroma dense results with BM25 sparse hits.

## 8. Observability
- `observability.py` registers OTLP tracing + Prometheus histograms/counters. Set `RAG_OBSERVABILITY__ENABLE_TRACING=true` to activate exporters.
- Expose `/metrics` by adding the Prometheus ASGI middleware if you need scrapes.
- `docs/prometheus.yml` + `docker-compose.yml` show how to wire Jaeger/Prometheus/Grafana locally.

## 9. Evaluation
```bash
python -m src.app.cli evaluate
```
This runs the sample queries in `sample_data/sample_queries.json`, calculates RAGAS metrics (answer relevancy + faithfulness), records latencies, and writes `data/evaluations/latest.json`.

## 10. Cleanup & Reset
- Generated artifacts live under `data/uploads`, `data/metadata`, and `data/vectorstore`. Delete their contents to reset the knowledge base.
- `make clean` (see `Makefile`) can automate wiping those directories.

## 11. Docker & Compose
```bash
cd ..
docker-compose up --build
```
This brings up the backend and Streamlit frontend. Update `.env` files before building if you need non-default credentials.
