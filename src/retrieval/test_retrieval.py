from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DB_DIR, EMBED_MODEL
from src.retrieval.rerank import rerank


def _boost_keyword_matches(query: str, docs, boost_factor: float = 1.5):
    """
    Boost documents that contain exact query keywords.
    Also boost high-priority content (boxes).
    Returns reordered list with keyword matches prioritized.
    """
    query_lower = query.lower()
    # Extract key terms (remove common words)
    stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
    key_terms = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    
    scored = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        score = 1.0
        
        # Boost high-priority content (boxes)
        priority = doc.metadata.get("priority", "normal")
        if priority == "high":
            score *= boost_factor * 1.5
        
        # Boost if exact phrase matches
        if query_lower in content_lower:
            score *= boost_factor * 2
        
        # Boost if key terms match
        term_matches = sum(1 for term in key_terms if term in content_lower)
        if term_matches > 0:
            score *= 1 + (term_matches / len(key_terms)) * (boost_factor - 1)
        
        # Extra boost for box modality
        if doc.metadata.get("modality") == "box":
            score *= boost_factor * 1.3
        
        scored.append((score, doc))
    
    # Sort by score (descending) but preserve original order for ties
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]


def main():
    # Setup embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=str(Path(CHROMA_DB_DIR)),
    )

    query = "What is Named Entity Recognition?"
    initial = store.similarity_search(query, k=15)  # Get more candidates
    # Boost keyword matches before reranking
    boosted = _boost_keyword_matches(query, initial)
    results = rerank(query, boosted[:10], top_k=3)  # Rerank top 10, return top 3

    print(f"Top 3 results for: {query!r}\n")
    for i, doc in enumerate(results, start=1):
        meta = doc.metadata or {}
        page = meta.get("page", "N/A")
        source = meta.get("source", "N/A")
        modality = meta.get("modality", "unknown")
        text_preview = (doc.page_content or "")[:200]
        print(f"Result {i}:")
        print(f"  Source: {source}")
        print(f"  Page: {page}")
        print(f"  Modality: {modality}")
        print(f"  Text: {text_preview}\n")


if __name__ == "__main__":
    main()