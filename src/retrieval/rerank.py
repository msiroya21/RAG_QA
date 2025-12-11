from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL


def rerank(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
    """Rerank a list of Documents using a cross-encoder."""
    if not docs:
        return []
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]

