"""
Vector store service.
Wraps Chroma so the rest of the app never touches embedding or
similarity-search details directly. Embeddings run locally via
sentence-transformers — no API key, no per-call cost.
"""

import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        chunks: list of {"text": ..., "metadata": {...}}
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        # IDs must be unique and stable; combine source + chunk index
        ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks]

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embeds the query and returns the top_k most similar chunks,
        each with its text, metadata, and distance score.
        """
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
        )

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def document_count(self) -> int:
        return self.collection.count()


# Single shared instance — Chroma's PersistentClient is safe to reuse
_vector_store_instance = None


def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
