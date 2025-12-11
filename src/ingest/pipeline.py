from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_RAW_DIR,
)
from src.embeddings.vector_store import add_documents
from src.ingest.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)


def _collect_page_content(elements: List[Dict]) -> Dict[int, Dict[str, List[str]]]:
    """
    Collect page content by type: text, table, box.
    Returns dict mapping page_num -> {"text": [...], "table": [...], "box": [...]}
    """
    pages: Dict[int, Dict[str, List[str]]] = {}
    for el in elements:
        page = el.get("page_number")
        content = el.get("content", "")
        el_type = el.get("type", "text")
        if not page or not content:
            continue
        if page not in pages:
            pages[page] = {"text": [], "table": [], "box": []}
        pages[page].setdefault(el_type, []).append(content)
    return pages


def _chunk_text(text: str) -> List[str]:
    """
    Chunk text with semantic awareness.
    Try to preserve boxes/cards as complete units by splitting on double newlines first.
    """
    # First, try to split by double newlines (often separates boxes/sections)
    sections = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # If adding this section would exceed chunk size, save current and start new
        if current_chunk and len(current_chunk) + len(section) + 2 > CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk = f"{current_chunk}\n\n{section}".strip() if current_chunk else section
        
        # If a single section is too large, use recursive splitter on it
        if len(current_chunk) > CHUNK_SIZE * 1.5:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            sub_chunks = splitter.split_text(current_chunk)
            chunks.extend(sub_chunks[:-1])  # Add all but last
            current_chunk = sub_chunks[-1] if sub_chunks else ""
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Fallback to standard splitter if no sections found
    if not chunks:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(text)
    
    return chunks


def process_pdf(pdf_path: Path) -> None:
    """Process PDF and extract text, tables, and boxes."""
    logger.info("Processing PDF: %s", pdf_path)
    
    try:
        elements = parse_pdf(pdf_path)
    except Exception as e:
        logger.error("Failed to parse %s: %s. Skipping entirely.", pdf_path, e)
        return
    
    if not elements:
        logger.warning("No content extracted from %s", pdf_path.name)
        return
    
    page_content = _collect_page_content(elements)

    docs: List[str] = []
    metas: List[dict] = []

    for page_num, content_dict in page_content.items():
        # PRIORITY 1: Box content - store as complete units (don't chunk)
        boxes = content_dict.get("box", [])
        for idx, box_text in enumerate(boxes):
            if box_text and box_text.strip():
                docs.append(box_text.strip())
                metas.append(
                    {
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk": idx,
                        "modality": "box",
                        "priority": "high",
                    }
                )

        # PRIORITY 2: Regular text - chunk normally
        text_parts = content_dict.get("text", [])
        combined_text = "\n\n".join(text_parts) if text_parts else ""
        if combined_text and combined_text.strip():
            text_chunks = _chunk_text(combined_text)
            for idx, chunk in enumerate(text_chunks):
                docs.append(chunk)
                metas.append(
                    {
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk": idx,
                        "modality": "text",
                        "priority": "normal",
                    }
                )

        # PRIORITY 3: Tables - store as complete units (don't chunk)
        table_parts = content_dict.get("table", [])
        for idx, table_text in enumerate(table_parts):
            if table_text and table_text.strip():
                docs.append(table_text.strip())
                metas.append(
                    {
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk": idx,
                        "modality": "table",
                        "priority": "normal",
                    }
                )

    if docs:
        add_documents(docs, metas, persist_immediately=False)
        logger.info("Added %s chunks from %s", len(docs), pdf_path.name)
    else:
        logger.warning("No content extracted from %s", pdf_path.name)


def run_pipeline() -> None:
    from src.embeddings.vector_store import persist_vector_store
    
    pdf_dir = Path(DATA_RAW_DIR)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s", pdf_dir)
        return

    total_docs = 0
    for pdf in pdf_files:
        try:
            process_pdf(pdf)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", pdf, exc)
    
    # Persist once at the end after all PDFs are processed
    logger.info("Persisting vector store...")
    persist_vector_store()
    logger.info("Pipeline complete!")