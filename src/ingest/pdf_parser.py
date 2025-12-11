from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pdfplumber

logger = logging.getLogger(__name__)


def _table_to_markdown(table: List[List[str]]) -> str:
    """Convert a pdfplumber table (list of rows) to a simple markdown string."""
    if not table:
        return ""
    # Normalize row lengths
    max_len = max(len(row) for row in table)
    norm_rows = []
    for row in table:
        padded = row + [""] * (max_len - len(row))
        norm_rows.append([cell.strip() if cell else "" for cell in padded])

    header = norm_rows[0]
    separator = ["---"] * max_len
    body = norm_rows[1:] if len(norm_rows) > 1 else []

    def row_to_md(row: List[str]) -> str:
        return "| " + " | ".join(cell.replace("\n", " ").strip() for cell in row) + " |"

    lines = [row_to_md(header), row_to_md(separator)]
    lines.extend(row_to_md(r) for r in body)
    return "\n".join(lines)


def _detect_box_regions(page) -> List[str]:
    """
    Detect potential box/card regions by grouping words that are close together.
    Returns list of text strings, each representing a potential box.
    """
    boxes = []
    
    try:
        words = page.words
        if not words or len(words) < 3:
            return boxes
        
        # Group words by proximity (simple clustering)
        processed = set()
        for i, word1 in enumerate(words):
            if i in processed:
                continue
            
            # Start a new box region
            box_words = [word1]
            x0_min = word1['x0']
            y0_min = word1['top']
            x1_max = word1['x1']
            y1_max = word1['bottom']
            
            # Find nearby words (within a threshold)
            for j, word2 in enumerate(words[i+1:], start=i+1):
                if j in processed:
                    continue
                
                # Check if word2 is near the current box region
                # Allow some margin for boxes
                margin = 30
                overlap_x = not (word2['x1'] < x0_min - margin or word2['x0'] > x1_max + margin)
                overlap_y = not (word2['bottom'] < y0_min - margin or word2['top'] > y1_max + margin)
                
                if overlap_x and overlap_y:
                    box_words.append(word2)
                    processed.add(j)
                    x0_min = min(x0_min, word2['x0'])
                    y0_min = min(y0_min, word2['top'])
                    x1_max = max(x1_max, word2['x1'])
                    y1_max = max(y1_max, word2['bottom'])
            
            # If we found a meaningful cluster (at least 3 words), extract text
            if len(box_words) >= 3:
                # Sort words by reading order (top to bottom, left to right)
                sorted_words = sorted(box_words, key=lambda w: (w['top'], w['x0']))
                box_text = " ".join(w['text'] for w in sorted_words if w.get('text', '').strip())
                
                if box_text.strip() and len(box_text) > 20:  # Minimum length to be meaningful
                    boxes.append(box_text.strip())
                    processed.add(i)
    except Exception as exc:
        logger.debug("Box detection encountered an issue: %s", exc)
    
    return boxes


def parse_pdf(pdf_path: Path) -> List[Dict]:
    """
    Parse a PDF into structured elements: text blocks, tables, and detected boxes.
    Includes aggressive error handling to skip problematic pages and avoid hangs.

    Returns a list of dicts with keys:
    - type: "text" | "table" | "box"
    - page_number: int (1-indexed)
    - content: str (text or markdown)
    """
    elements: List[Dict] = []
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return elements

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            logger.info("Parsing PDF with %d pages", total_pages)
            
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    # Try to detect boxes first (before general text extraction)
                    try:
                        detected_boxes = _detect_box_regions(page)
                        for box_text in detected_boxes:
                            if box_text.strip():
                                elements.append({
                                    "type": "box",
                                    "page_number": page_idx,
                                    "content": box_text.strip(),
                                })
                    except Exception as exc:
                        logger.debug("Box detection failed on page %s: %s", page_idx, exc)

                    # Extract general text
                    try:
                        text = page.extract_text() or ""
                    except Exception as exc:
                        logger.warning("Text extraction failed on page %s: %s", page_idx, exc)
                        text = ""

                    if text.strip():
                        elements.append(
                            {
                                "type": "text",
                                "page_number": page_idx,
                                "content": text.strip(),
                            }
                        )

                    # Extract tables
                    try:
                        tables = page.extract_tables() or []
                    except Exception as exc:
                        logger.warning("Table extraction failed on page %s: %s", page_idx, exc)
                        tables = []

                    for table in tables:
                        md = _table_to_markdown(table)
                        if md.strip():
                            elements.append(
                                {
                                    "type": "table",
                                    "page_number": page_idx,
                                    "content": md,
                                }
                            )
                
                except Exception as exc:
                    logger.warning("Skipping page %d due to error: %s", page_idx, exc)
                    # Continue to next page instead of failing entire PDF
                    continue
                    
    except Exception as exc:
        logger.error("Failed to open/parse PDF %s: %s", pdf_path, exc)
        if not elements:
            logger.error("No content extracted from %s - PDF may be corrupted", pdf_path)

    return elements

