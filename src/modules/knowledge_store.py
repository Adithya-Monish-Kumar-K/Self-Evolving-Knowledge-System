"""
Knowledge Store – ChromaDB-backed vector store for semantic search
over the knowledge base. Also provides the dataset ingestion pipeline.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

import config
from utils.embeddings import embed_text, embed_batch
from utils.database import (
    init_db, insert_papers_batch, insert_knowledge_entry, count_papers,
)
from utils.knowledge_graph import get_knowledge_graph

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Persistent ChromaDB-backed vector store with metadata."""

    def __init__(self, persist_dir: str = config.CHROMA_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents)",
            config.CHROMA_COLLECTION_NAME,
            self.collection.count(),
        )

    # ── Search ───────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = config.CHROMA_SEARCH_TOP_K,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search. Returns list of dicts with id, content, metadata, distance."""
        query_emb = embed_text(query)
        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)

        docs = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 1.0,
                    "similarity": 1.0 - (results["distances"][0][i] if results["distances"] else 1.0),
                })
        return docs

    # ── CRUD ─────────────────────────────────────────────────────────
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None,
    ):
        """Add or update a single document."""
        emb = embedding or embed_text(content)
        self.collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[emb],
            metadatas=[metadata or {}],
        )

    def add_documents_batch(
        self,
        ids: List[str],
        contents: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Batch add/upsert documents."""
        embs = embeddings or embed_batch(contents)
        metas = metadatas or [{} for _ in ids]
        self.collection.upsert(
            ids=ids,
            documents=contents,
            embeddings=embs,
            metadatas=metas,
        )

    def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """Update an existing document (re-embed + upsert)."""
        self.add_document(doc_id, content, metadata)

    def delete_document(self, doc_id: str):
        try:
            self.collection.delete(ids=[doc_id])
        except Exception as e:
            logger.warning("Could not delete doc %s: %s", doc_id, e)

    def count(self) -> int:
        return self.collection.count()

    def get_document(self, doc_id: str) -> Optional[Dict]:
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0] if result["documents"] else "",
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
        except Exception:
            pass
        return None


# ── Dataset Ingestion ────────────────────────────────────────────────

def ingest_arxiv_dataset(filepath: str, max_papers: int = None):
    """
    Load arXiv JSON dataset (one JSON object per line) and populate:
      - SQLite papers table
      - ChromaDB vector collection
      - Knowledge Graph
    """
    init_db()
    store = get_knowledge_store()
    kg = get_knowledge_graph()

    logger.info("Starting ingestion from %s", filepath)

    # Read all papers
    papers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
                papers.append(paper)
            except json.JSONDecodeError:
                continue
            if max_papers and len(papers) >= max_papers:
                break

    total = len(papers)
    logger.info("Loaded %d papers, starting batch ingestion…", total)

    batch_size = config.INGESTION_BATCH_SIZE

    for start in tqdm(range(0, total, batch_size), desc="Ingesting"):
        batch = papers[start : start + batch_size]

        # 1. SQLite
        insert_papers_batch(batch)

        # 2. Prepare text for embedding
        ids = []
        texts = []
        metas = []
        for p in batch:
            pid = p.get("id", "")
            title = p.get("title", "").strip()
            abstract = p.get("abstract", "").strip()
            combined = f"{title}\n{abstract}"
            if not combined.strip():
                continue
            ids.append(f"paper:{pid}")
            texts.append(combined[:2000])  # cap at 2000 chars
            metas.append({
                "paper_id": pid,
                "title": title[:200],
                "categories": p.get("categories", ""),
                "source_type": "ingested",
                "update_date": p.get("update_date", ""),
            })

        # 3. Embed + insert into ChromaDB
        if texts:
            embeddings = embed_batch(texts)
            store.add_documents_batch(ids, texts, metas, embeddings)

        # 4. Knowledge Graph
        for p in batch:
            kg.add_paper(p)

    # Save the graph
    kg.save()
    logger.info(
        "Ingestion complete: %d papers in SQLite, %d docs in ChromaDB, graph: %s",
        count_papers(), store.count(), kg.stats(),
    )


# ── Singleton ────────────────────────────────────────────────────────
_store: Optional[KnowledgeStore] = None


def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store
