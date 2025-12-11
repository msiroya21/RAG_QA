# RAG-QA - PDF Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system that ingests PDF documents, creates semantic embeddings, and enables accurate question-answering with grounded source citations.

## Overview

This system combines three advanced retrieval techniques to answer questions about PDF documents with precision:

1. **Semantic Search** – Uses HuggingFace embeddings (e5-large-v2) to find semantically relevant passages
2. **Keyword Boosting** – Amplifies exact keyword matches for improved recall
3. **Cross-Encoder Reranking** – Re-ranks results using a fine-tuned cross-encoder for optimal relevance

The LLM (Groq Llama 3.3-70b) receives documents with explicit page metadata, ensuring **accurate, grounded citations** with correct page numbers.

## Project Structure

```
src/
├── app/
│   └── ui.py                  # Streamlit interactive chat interface
├── embeddings/
│   └── vector_store.py        # ChromaDB management and HuggingFace embeddings
├── ingest/
│   ├── pipeline.py            # PDF processing orchestration
│   └── pdf_parser.py          # Text, table, and box extraction via pdfplumber
├── qa/
│   └── chain.py               # Custom retriever, 3-stage ranking, QA chain
└── retrieval/
    └── rerank.py              # Cross-encoder reranking

data/
├── raw/                       # Input PDF files
├── processed/                 # Processed document metadata
└── chroma_db/                 # Persistent vector store

run_pipeline.py                # CLI for batch PDF ingestion
requirements.txt               # Python dependencies
README.md                       # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Documents

Place PDF files in `data/raw/` directory.

### 2. Ingest PDFs

```bash
python run_pipeline.py
```

This processes all PDFs in `data/raw/`, extracts text/tables/boxes, creates embeddings, and stores them in ChromaDB.

### 3. Launch the App

```bash
streamlit run src/app/ui.py
```

## Features

### Chat Interface
- Ask natural language questions about your documents
- Receive answers powered by a 3-stage retrieval pipeline
- Get instant feedback with accurate page citations

### Source Citations
- Click **"📚 View Source Citations"** to see:
  - Exact page numbers
  - Source filenames
  - Retrieved text snippets (with modality labels for tables/boxes)

### Retrieval Pipeline
1. **Semantic Search** – Find top-K passages using embedding similarity
2. **Keyword Boosting** – Boost results containing exact query terms (2x multiplier)
3. **Cross-Encoder Reranking** – Final ranking using ms-marco-MiniLM-L-6-v2

### Metadata Grounding
- Page numbers, sources, and modality types (text/table/box) are embedded in the LLM context
- LLM sees explicitly: `[Source] filename.pdf | Page X` for each document
- Eliminates hallucinated citations

## Configuration

Edit `src/config.py` to customize:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CHUNK_SIZE` | 600 | Characters per text chunk |
| `CHUNK_OVERLAP` | 120 | Overlap between chunks |
| `TOP_K` | 5 | Number of semantic search results |
| `RERANK_TOP_K` | 10 | Documents sent to cross-encoder |
| `EMBEDDING_MODEL` | e5-large-v2 | HuggingFace embedding model |
| `LLM_MODEL` | llama-3.3-70b-versatile | Groq model to use |

## Architecture Highlights

### Three-Stage Retrieval

```
Query
  ↓
[1] Semantic Search (ChromaDB)
  ↓ (top-K results)
[2] Keyword Boosting (exact match amplification)
  ↓ (filtered results)
[3] Cross-Encoder Reranking
  ↓ (top-K ranked results)
LLM with Metadata Context
```

### Metadata-Grounded Chain

The custom chain in `src/qa/chain.py` ensures:
- Each document includes explicit source and page metadata
- LLM sees page numbers before generating answers
- Citations are grounded in actual document positions

### Graceful Degradation

- Per-page error handling prevents one corrupted PDF from failing the entire pipeline
- Missing embeddings, tables, or boxes are skipped with logging
- System continues processing remaining content

## Performance

- **Ingestion**: ~2-3 seconds per PDF (text-only, optimized)
- **Retrieval**: <1 second (semantic + keyword + reranking)
- **LLM Response**: 3-5 seconds (via Groq)
- **Total QA Latency**: 4-6 seconds end-to-end

## Deployment

### Development
```bash
streamlit run src/app/ui.py
```

### Production Considerations
- Use managed vector DB (Qdrant, Pinecone, Weaviate) for scaling
- Host cross-encoder on GPU for throughput
- Add LLM response caching for repeated questions
- Implement rate limiting and authentication
- Use async embeddings for batch ingestion

## Technologies

| Component | Technology |
|-----------|-----------|
| Embeddings | HuggingFace (e5-large-v2) |
| Vector DB | ChromaDB (local + persistent) |
| PDF Parsing | pdfplumber |
| LLM | Groq (Llama 3.3-70b) |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| UI | Streamlit |
| Orchestration | LangChain |

## Troubleshooting

**Q: ChromaDB not found after restart?**
- ChromaDB persists in `data/chroma_db/`. Delete this folder to reset.

**Q: Inaccurate citations?**
- Check `data/raw/` for document corruption. System logs page-level errors.
- Verify embeddings model is loaded (check terminal for HuggingFace downloads).

**Q: Slow ingestion?**
- Monitor CPU/memory in `run_pipeline.py` logs.
- Reduce `CHUNK_SIZE` or `TOP_K` in `src/config.py` if needed.

**Q: Groq API errors?**
- Ensure `GROQ_API_KEY` environment variable is set correctly.
- Check API rate limits at https://console.groq.com

## Environment Variables

Required:
- `GROQ_API_KEY` – API key for Groq LLM access

Optional:
- `HF_HOME` – Custom directory for HuggingFace model caching

## License

This project is provided as-is for educational and commercial use.
