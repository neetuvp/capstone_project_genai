# GenAI Document Assistant — Capstone Project

An agentic RAG application that lets users upload enterprise documents
(PDF, TXT, CSV, Excel) and ask natural language questions about them,
answered by autonomous AI agents grounded in the uploaded content.

## Architecture

```
Streamlit UI  --HTTP-->  FastAPI backend  -->  Agent pipeline (LangGraph)
                                                  Plan -> Retrieve -> Reason -> Respond
                                                            |
                                                     Chroma vector store
                                                     (local embeddings)
                                                            |
                                                     Ollama (local LLM)
```

- **UI layer:** Streamlit app (`streamlit_app.py`) — upload documents, ask questions, view answers + sources.
- **API layer:** FastAPI (`app/api/main.py`) — exposes `/upload-document`, `/ask-questions`, `/health-check`.
- **Ingestion:** `app/services/ingestion.py` — extracts text from PDF, TXT, CSV, XLSX.
- **Chunking:** `app/services/chunking.py` — splits text into overlapping chunks for embedding.
- **Vector store:** `app/services/vector_store.py` — Chroma DB with local `sentence-transformers` embeddings (no API key required).
- **RAG core:** `app/core/rag_pipeline.py` — standalone retrieve+generate function (used directly or via agents).
- **Agents:** `app/agents/` — four cooperating agents, each a single LangGraph node:
  - **Planner** (`planner.py`) — decides retrieval strategy (targeted vs. broad) based on the question.
  - **Retriever** (`retriever_agent.py`) — runs similarity search against the vector store.
  - **Reasoning** (`reasoning_agent.py`) — calls the local LLM to draft an answer grounded in retrieved context.
  - **Response** (`response_agent.py`) — validates the draft via guardrails and packages the final output with sources.
  - **Orchestrator** (`orchestrator.py`) — wires the four agents into a LangGraph state graph.
- **Guardrails:** `app/core/guardrails.py` — input validation (file type/size, question length) and output validation (flags empty or likely-ungrounded answers).
- **Logging:** `app/utils/logging_config.py` — centralized logging for debugging ingestion and query failures.

## Tech stack (all free / local — no API keys required)

| Component | Tool |
|---|---|
| LLM | Ollama (`llama3.1:8b`, runs locally) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local) |
| Vector DB | ChromaDB (embedded, local) |
| Orchestration | LangChain + LangGraph |
| API | FastAPI |
| UI | Streamlit |

> Note: the class notes referenced `langchain-openai`. This project substitutes
> `langchain-ollama` to keep the whole stack free and local — the RAG/agent
> logic is otherwise identical to what an OpenAI-backed version would use.

## Setup

1. **Install Ollama** (one-time): download from [ollama.com](https://ollama.com), then:
   ```
   ollama pull llama3.1:8b
   ```
2. **Create a virtual environment and install dependencies:**
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   pip install -r requirements.txt
   ```
3. **Start the API** (from the project root):
   ```
   python -m uvicorn app.api.main:app --reload
   ```
4. **Start the UI** (in a second terminal):
   ```
   streamlit run streamlit_app.py
   ```
5. Open the Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## Deployment (Docker)

```
docker-compose up --build
```

This starts the API on port 8000 and the UI on port 8501. Ollama itself
runs on the host machine (not containerized) since it needs direct
access to system resources for the LLM — see the note in
`docker-compose.yml` for platform-specific networking.

## Workflow

1. User uploads a document via the UI.
2. The API validates the file, extracts text, chunks it, embeds the
   chunks locally, and stores them in Chroma.
3. User asks a question.
4. The agent pipeline runs: Planner decides retrieval breadth ->
   Retriever fetches relevant chunks -> Reasoning agent drafts an
   answer grounded in those chunks -> Response agent validates the
   draft against guardrails and returns the final answer with cited
   sources.

## Limitations

- Answer quality depends on the local LLM (`llama3.1:8b` by default) —
  smaller local models are less capable than large hosted models.
- Guardrails are rule-based (keyword/heuristic checks), not a second
  LLM-based verification pass, so they catch obvious failure patterns
  but not subtle hallucinations.
- No authentication/multi-user isolation — all uploaded documents share
  a single Chroma collection.
- Excel/CSV files are converted to flattened text tables, which works
  well for small-to-medium sheets but loses some structure on very
  large or complex spreadsheets.
- Ollama must be running locally for `/ask-questions` to work; the API
  will return a 500 error with a hint if it can't reach Ollama.

## Project structure

```
genai_capstone/
├── app/
│   ├── api/main.py            # FastAPI endpoints
│   ├── services/               # ingestion, chunking, vector store
│   ├── core/                    # RAG pipeline, guardrails
│   ├── agents/                   # Planner, Retriever, Reasoning, Response, orchestrator
│   └── utils/                     # logging
├── data/
│   ├── uploads/                    # uploaded files
│   └── chroma_db/                   # persisted vector store
├── streamlit_app.py
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
