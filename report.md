# Technical Report — Multi-Modal RAG System

## 1. Architecture (Multimodal Summarization)

This system follows a "Multimodal Summarization" approach: images (page renders) are processed with a generative vision model (Gemini 2.0 Flash) to produce concise text descriptions and scene summaries. Those image descriptions are treated equivalently to extracted text (paragraphs, table markdown, visual "boxes") and are embedded using the same embedding model. All embeddings (text + image descriptions) are stored together in ChromaDB, producing a unified multi-modal embedding space.

At query time, a retriever performs vector similarity search over this unified space, returning candidates across modalities. Candidates are then boosted by simple keyword- and metadata-based heuristics and finally reranked using a cross-encoder to improve snippet-level relevance before being passed into a LangChain-based QA chain powered by Gemini.

This pipeline converts visual content to text at ingestion time, enabling downstream components (embedding, retrieval, LLM reasoning) to operate on a single, uniform representation while preserving modality-specific metadata for citation and display.

## 2. Design Choices

- Why Gemini 2.0 Flash?
  - Speed: Flash variants prioritize low-latency responses, which is important both for describing images during ingestion and for interactive QA.
  - Cost/Rate trade-offs: Flash models generally provide a good balance of cost-to-performance for high-throughput preprocessing jobs (vision descriptions) and interactive chat.
  - Practicality: Using an LLM with strong vision-language capability simplifies building high-quality image descriptions versus handcrafted CV pipelines.

- Why ChromaDB?
  - Local-first: Chroma is easy to run locally, persists to disk, and is suitable for prototyping and mid-size datasets.
  - Simplicity: The codebase can directly embed and persist documents without external hosted services, lowering integration friction.

- Why Streamlit?
  - Rapid prototyping: Streamlit allows building an interactive demo quickly and focuses on UX without writing frontend code.
  - Iteration speed: Developers can test ingestion, question flows, and citation display rapidly.

## 3. Benchmarks & Evaluation (Accuracy & Faithfulness)

This section outlines how to measure system quality and a placeholder summary of expected behaviors.

- Accuracy: Evaluate whether answers are correct with respect to source documents. Create a labeled test set of questions with ground-truth answers and measure exact/partial match and F1.

- Faithfulness: Ensure the system cites the correct page(s) and does not hallucinate facts not present in the context. Use interventions where the retriever is restricted to ground-truth pages and check whether answers match perspectives in those pages.

- Citation-based evaluation: For each produced answer, verify that the content can be traced to cited snippets (source pages). Measure the fraction of produced facts that have supporting text within cited pages.

Placeholder summary: the architecture encourages faithfulness by (1) constraining the LLM with a context-only prompt, (2) providing page-level metadata for citations, and (3) re-ranking with a cross-encoder to prioritize snippets with high semantic overlap. The remaining risk is that the LLM may still synthesize or conflate details; robust evaluation and stricter prompt scaffolding (or answer verification steps) can mitigate this.

## 4. Future Improvements

- Cross-modal reranking: Use a multimodal cross-encoder that jointly inspects image pixels and text snippets for more accurate visual grounding.
- Hybrid Search: Combine sparse (BM25 / metadata filtering) and dense retrieval to improve recall for fact-heavy queries (e.g., tables and numeric facts).
- On-demand image describing: Allow runtime image uploads to be described and appended temporarily to retrieval results (good for user-supplied images during a session).
- Citation enforcement: Post-process answers to automatically append verified page citations or map answer spans to source snippets programmatically to reduce hallucination.
- Production scaling: Move to a managed vector DB (Qdrant, Pinecone) and deploy the cross-encoder on GPU-backed inference for lower latency.
