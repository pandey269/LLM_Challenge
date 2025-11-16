# Frontend (Streamlit)

This package provides the UI for uploading documents and querying the backend API.

## Local Dev
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # or export BACKEND_URL manually
streamlit run app/main.py --server.port 8501
```

## Environment
- `.env` → `BACKEND_URL` (default `http://localhost:8000`): base URL of the FastAPI backend.
- `st.secrets["api_base"]`: optional Streamlit secret overriding the backend URL.

## Docker
A separate `Dockerfile` is provided; it expects the backend service to be reachable at `http://backend:8000`.
