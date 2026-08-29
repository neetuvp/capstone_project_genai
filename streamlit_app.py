"""
Streamlit UI.
A thin client on top of the FastAPI backend — all real logic
(ingestion, retrieval, agents) lives in the API, so this file only
handles user interaction and displaying results.

Run with (in a separate terminal from the API):
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="GenAI Document Assistant", page_icon="📄")
st.title("📄 GenAI Document Assistant")
st.caption("RAG + Agentic AI capstone — upload documents, then ask questions about them.")

# --- Health check ---
with st.sidebar:
    st.subheader("System status")
    try:
        health = requests.get(f"{API_URL}/health-check", timeout=3)
        if health.status_code == 200:
            st.success("API is running")
        else:
            st.error("API returned an error")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API. Start it with:\n`python -m uvicorn app.api.main:app --reload`")

# --- Document upload ---
st.subheader("1. Upload a document")
uploaded_file = st.file_uploader(
    "Supported formats: PDF, TXT, CSV, XLSX",
    type=["pdf", "txt", "csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    if st.button("Ingest document"):
        with st.spinner("Processing and embedding document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{API_URL}/upload-document", files=files, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Ingested '{data['filename']}' — {data['chunks_added']} chunks added.")
                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API.")

st.divider()

# --- Question asking ---
st.subheader("2. Ask a question")
question = st.text_input("Your question about the uploaded documents:")

if st.button("Ask") and question:
    with st.spinner("Thinking... (Planner -> Retriever -> Reasoning -> Response)"):
        try:
            response = requests.post(
                f"{API_URL}/ask-questions", json={"question": question}, timeout=300
            )
            if response.status_code == 200:
                result = response.json()
                st.markdown("### Answer")
                st.write(result["answer"])

                if result.get("sources"):
                    st.markdown("**Sources:** " + ", ".join(result["sources"]))

                validation = result.get("validation", {})
                if not validation.get("passed", True):
                    st.warning(f"⚠️ Guardrail flags: {validation.get('flags')}")

                with st.expander("Debug details"):
                    st.json(result)
            else:
                st.error(f"Request failed: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API.")
