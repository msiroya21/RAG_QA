from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DB_DIR, EMBED_MODEL

logger = logging.getLogger(__name__)

# Cache embeddings to avoid reloading on each add_documents call
_embeddings_cache: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create cached embeddings model."""
    global _embeddings_cache
    if _embeddings_cache is None:
        logger.info("Loading embeddings model: %s", EMBED_MODEL)
        _embeddings_cache = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings_cache


def get_vector_store(collection_name: str = "documents") -> Chroma:
    """Return a persistent Chroma vector store."""
    Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )


def add_documents(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    collection_name: str = "documents",
    persist_immediately: bool = False,
) -> None:
    """
    Add documents to the Chroma collection.
    
    Args:
        texts: List of document texts
        metadatas: List of metadata dicts (one per text)
        collection_name: Name of the collection
        persist_immediately: If False, persist is deferred (faster for batch ops).
                           Call persist_vector_store() manually when done.
    """
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas length must match texts length")

    store = get_vector_store(collection_name)
    store.add_texts(texts=texts, metadatas=metadatas)
    
    if persist_immediately:
        store.persist()


def persist_vector_store(collection_name: str = "documents") -> None:
    """Manually persist the vector store (call after batch operations)."""
    store = get_vector_store(collection_name)
    store.persist()
    logger.info("Vector store persisted")


