"""
tools/rag_memory.py
Local vector-store "data dictionary" memory: index company schemas, KPI
definitions, and metric formulas, then retrieve relevant entries by
semantic search. Uses a persistent local ChromaDB collection with a local
sentence-transformers embedding model, so no external API key is needed.
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from ..config import settings

_COLLECTION_NAME = "jarvis_data_dictionary"


class DataDictionary:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME, embedding_function=self.embedder
        )

    def add_entry(self, term: str, definition: str, source: str = "manual") -> str:
        """Index a single schema/KPI/metric definition, upserting by term."""
        if not term.strip() or not definition.strip():
            return "Both a term and a definition are required."
        doc_id = term.strip().lower().replace(" ", "_")
        self.collection.upsert(
            ids=[doc_id],
            documents=[f"{term}: {definition}"],
            metadatas=[{"term": term, "source": source}],
        )
        return f"Indexed '{term}' into the data dictionary."

    def add_file(self, path: str) -> str:
        """Bulk-index a text/markdown/CSV file of 'term: definition' lines."""
        p = Path(path)
        if not p.exists():
            p = settings.data_dir / path
        if not p.exists():
            return f"File not found: {path}"
        added = 0
        for line in p.read_text(errors="ignore").splitlines():
            if not line.strip() or ":" not in line:
                continue
            term, _, definition = line.partition(":")
            if term.strip() and definition.strip():
                self.add_entry(term.strip(), definition.strip(), source=p.name)
                added += 1
        return f"Indexed {added} entries from {p.name}"

    def search(self, query: str, k: int = 3) -> str:
        """Semantic search over indexed schema/KPI/metric definitions."""
        count = self.collection.count()
        if count == 0:
            return "The data dictionary is empty. Add entries first, e.g. 'remember DAU: Daily Active Users, counted as...'."
        results = self.collection.query(query_texts=[query], n_results=min(k, count))
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No matching definitions found."
        return "\n".join(f"- {d}" for d in docs)


# Module-level singleton shared by the agent's data-dictionary tools.
data_dictionary = DataDictionary()
