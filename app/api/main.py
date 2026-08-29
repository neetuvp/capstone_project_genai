"""
FastAPI application.
Exposes the three core endpoints from the class notes:
  POST /upload-document  - ingest a file into the knowledge base
  POST /ask-questions     - run the agent pipeline against the knowledge base
  GET  /health-check       - liveness check, returns 200 when the API is up

Run from the project root with:
    python -m uvicorn app.api.main:app --reload
"""

import os
import shutil
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from config import UPLOAD_DIR
from app.services.ingestion import load_document, UnsupportedFileTypeError
from app.services.chunking import chunk_text
from app.services.vector_store import get_vector_store
from app.core.guardrails import validate_upload, validate_question, ValidationError
from app.agents.orchestrator import run_agent_query
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="GenAI Document Assistant",
    description="RAG + Agentic AI capstone project — query enterprise documents with autonomous agents.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/health-check")
def health_check():
    """Simple liveness probe. Returns 200 when the API is up."""
    return {"status": "ok"}


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a file (PDF, TXT, CSV, XLSX), validates it, extracts text,
    chunks it, embeds it, and stores it in the vector database.
    """
    dest_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save the upload to disk first so we can check its real size
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(dest_path)
        validate_upload(dest_path, file_size)

        text = load_document(dest_path)
        chunks = chunk_text(text, source_name=file.filename)

        vector_store = get_vector_store()
        added = vector_store.add_chunks(chunks)

        logger.info(f"Ingested '{file.filename}': {added} chunks added.")
        return {
            "filename": file.filename,
            "chunks_added": added,
            "status": "success",
        }

    except (ValidationError, UnsupportedFileTypeError, ValueError) as e:
        logger.warning(f"Rejected upload '{file.filename}': {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error ingesting '{file.filename}': {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal error while processing the document.")

    finally:
        file.file.close()


@app.post("/ask-questions")
def ask_questions(request: QuestionRequest):
    """
    Runs the full agent pipeline (Plan -> Retrieve -> Reason -> Respond)
    against the knowledge base and returns a grounded answer with sources.
    """
    try:
        validate_question(request.question)
        result = run_agent_query(request.question)
        return result

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error answering question: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal error while generating an answer. Is Ollama running?",
        )
