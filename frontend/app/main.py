"""Streamlit UI for uploading documents and asking questions."""
from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Any, Dict

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API = "http://localhost:8000"
API_BASE = os.getenv("BACKEND_URL") or os.getenv("RAG_API_BASE")
if not API_BASE:
    try:
        API_BASE = st.secrets["api_base"]
    except Exception:  # secrets file optional
        API_BASE = DEFAULT_API

st.set_page_config(page_title="Intelligent Document QA", layout="wide")
st.title("📄 Intelligent Document Q&A – Streamlit UI")

st.sidebar.header("1. Upload Documents")
uploaded = st.sidebar.file_uploader(
    "Upload PDF / DOCX / PPTX / TXT",
    type=["pdf", "docx", "pptx", "txt", "csv"],
    accept_multiple_files=False,
)
uploader_name = st.sidebar.text_input("Uploaded by", value="streamlit-user")
ingest_btn = st.sidebar.button("Ingest Document", use_container_width=True, type="primary")

ingest_status = st.sidebar.empty()
if ingest_btn:
    if not uploaded:
        ingest_status.warning("Please choose a file before ingesting.")
    else:
        files = {"file": (uploaded.name, BytesIO(uploaded.read()), uploaded.type or "application/octet-stream")}
        params = {"uploaded_by": uploader_name}
        ingest_status.info("Uploading…")
        response = requests.post(f"{API_BASE}/ingest", files=files, params=params, timeout=600)
        if response.ok:
            ingest_status.success(f"Ingested document id: {response.json().get('document_id')}")
        else:
            ingest_status.error(f"Failed: {response.status_code} – {response.text}")

st.header("2. Ask Questions")
question = st.text_area("Enter your question", height=120)
ask_btn = st.button("Ask", type="primary")

if ask_btn:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Running self-RAG pipeline…"):
            payload: Dict[str, Any] = {"question": question}
            resp = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
        if resp.ok:
            data = resp.json()
            st.subheader("Answer")
            st.write(data.get("answer"))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{data.get('confidence', 0):.2f}")
                st.metric("Latency (ms)", f"{data.get('latency_ms', 0):.0f}")
            with col2:
                not_found = data.get("not_found_reason")
                if not_found:
                    st.info(f"Not found reason: {not_found}")
            st.subheader("Citations")
            for idx, citation in enumerate(data.get("citations", []), start=1):
                st.markdown(f"**{idx}.** `{json.dumps(citation)}`")
            if data.get("evidence"):
                st.subheader("Evidence Chunks")
                for chunk in data["evidence"]:
                    st.code(chunk)
        else:
            st.error(f"Query failed: {resp.status_code} – {resp.text}")

st.caption("API base: %s" % API_BASE)
