"""
Central configuration for the app.
Keeping all the "knobs" in one file makes it easy to tune later
without hunting through every module.
"""

import os

# --- LLM settings (Ollama runs locally, no API key needed) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Embedding model (runs locally via sentence-transformers) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking settings ---
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 120     # overlap between chunks so context isn't cut off

# --- Retrieval settings ---
TOP_K = 4                # how many chunks to retrieve per query

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "enterprise_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
