# Intelligent Document Q&A – Monorepo Layout

This repository now contains two isolated packages that can be developed and deployed independently:

| Folder     | Description |
|------------|-------------|
| `backend/` | FastAPI + LangGraph service handling document ingestion, retrieval, and question answering. Contains all data/ML assets and the original documentation. |
| `frontend/`| Streamlit UI for uploading files and running queries against the backend API. |

## Local Development

1. **Backend**
   ```bash
   cd backend
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   cp .env.example .env  # set OPENAI_API_KEY or switch to Ollama as needed
   uvicorn src.app.server:app --reload --port 8000
   ```

2. **Frontend**
   ```bash
   cd frontend
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   cp .env.example .env  # set BACKEND_URL if not using default
   streamlit run app/main.py --server.port 8501
   ```

## Docker Compose

From the repo root:

```bash
docker-compose up --build
```

This starts both services with the frontend reaching the backend at `http://backend:8000`.

## Docs & Architecture
- Backend documentation (architecture, implementation guide, evaluation plan, Prometheus config) lives under `backend/docs/`.
- `backend/docs/architecture.png` is a lightweight visualization of the client → Streamlit → FastAPI → LangGraph/LLM flow.

## Data Directories
The backend writes runtime artifacts to `backend/data/uploads`, `backend/data/metadata`, and `backend/data/vectorstore`. These folders are tracked only with `.gitkeep` placeholders; delete their contents whenever you want to reset the vector store.

## Additional Notes
- Customize environment variables (`RAG_*`, `BACKEND_URL`) as needed for staging/production deployments.

==============
Added for files->
Putting it all together: When a user uploads docs and asks a question, the ingestion pipeline (2) parses and chunks them, the embedding service (3) adds them to Chroma, and the retriever (4) supplies context to the LangGraph workflow (5). The LLM client (6) produces a grounded answer, which the server (1) returns along with citations, while observability hooks (8) log metrics and traces. You can manage everything via CLI commands (9) and configure behavior (OpenAI/Open-source LLM, caching, tracing) via config.py (10).