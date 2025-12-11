from __future__ import annotations

import os
import sys
# import tempfile  # Disabled: ingest-on-upload feature disabled for now
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.qa.chain import get_qa_chain
# from src.ingest.pipeline import process_pdf  # Disabled: ingest-on-upload feature disabled for now


@st.cache_resource
def load_chain():
    return get_qa_chain()


def extract_context_list(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize possible context formats into a list of dict-like items."""
    contexts = []
    if not result:
        return contexts

    # Check for new custom chain format (context_docs key)
    if "context_docs" in result and result["context_docs"]:
        contexts = result["context_docs"]
    # Check for legacy retrieval chain formats
    elif "context" in result and result["context"]:
        contexts = result["context"]
    elif "source_documents" in result and result["source_documents"]:
        contexts = result["source_documents"]
    elif "sources" in result and result["sources"]:
        contexts = result["sources"]

    normalized = []
    for c in contexts:
        if hasattr(c, "page_content"):
            normalized.append({"text": c.page_content, "metadata": dict(getattr(c, "metadata", {}) or {})})
        elif isinstance(c, dict):
            text = c.get("page_content") or c.get("text") or c.get("content") or ""
            metadata = c.get("metadata") or c.get("meta") or {}
            normalized.append({"text": text, "metadata": dict(metadata or {})})
        else:
            normalized.append({"text": str(c), "metadata": {}})

    return normalized


def render_sources(contexts: List[Dict[str, Any]]):
    if not contexts:
        st.info("No source documents available.")
        return

    for i, item in enumerate(contexts, start=1):
        metadata = item.get("metadata", {})
        page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_num")
        modality = metadata.get("modality", "text")
        source_file = metadata.get("source", "Unknown Source")
        snippet = (item.get("text") or "").strip()
        
        # Determine display text based on modality
        if str(modality).lower() in ("table", "table_extraction"):
            display_type = "Table"
            snippet_display = "(Table data - see document for full table)"
        elif str(modality).lower() in ("vision", "image", "image_description"):
            display_type = "Image Description"
            snippet_display = snippet if snippet else "(Image description not available)"
        else:
            display_type = "Text"
            # Truncate text snippets
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
            snippet_display = snippet if snippet else "(No text snippet available)"

        label = f" [{display_type}]"

        title = f"Source {i} — {source_file}"
        if page is not None:
            title += f" (Page {page})"
        title += label

        st.markdown(f"**{title}**")
        st.markdown(f"*Modality: {modality}*")

        # Show thumbnail if we have an image path in metadata
        img_path = metadata.get("image_path") or metadata.get("img_path") or metadata.get("image")
        if img_path and os.path.exists(img_path):
            try:
                st.image(img_path, width=220)
            except Exception:
                pass

        # Display snippet
        st.markdown("**Snippet:**")
        st.markdown(f"> {snippet_display}")
        
        st.divider()


def _render_message_bubble(role: str, content: str, refined: Optional[Dict] = None) -> None:
    # helper kept for future HTML-rich bubbles
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**Assistant:** {content}")


# def ingest_uploaded_pdf(uploaded_file, persist_immediately: bool = True) -> tuple[bool, str]:
#     """
#     Process an uploaded PDF file and add it to the RAG system.
#     
#     Args:
#         uploaded_file: Streamlit UploadedFile object
#         persist_immediately: If True, persist to disk after ingestion
#         
#     Returns:
#         (success: bool, message: str)
#     """
#     try:
#         # Create a temporary file to save the upload
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_file.getbuffer())
#             tmp_path = Path(tmp_file.name)
#         
#         # Process the PDF (skip vision for speed, can be re-enabled)
#         process_pdf(tmp_path, skip_vision=True)
#         
#         # Persist to vector store
#         if persist_immediately:
#             from src.embeddings.vector_store import persist_vector_store
#             persist_vector_store()
#         
#         # Clean up temp file
#         tmp_path.unlink()
#         
#         return True, f"✓ Successfully ingested '{uploaded_file.name}'"
#     except Exception as e:
#         return False, f"✗ Error ingesting '{uploaded_file.name}': {str(e)}"


def main():
    st.set_page_config(layout="wide", page_title="Multi-Modal RAG", initial_sidebar_state="expanded")

    # Inject minimal CSS for chat bubbles and layout
    st.markdown(
        """
        <style>
        .app-title {font-size:30px; font-weight:700; margin-bottom:10px}
        .msg {padding:8px; margin:10px 0; border-radius:10px;}
        .msg-content {margin-top:6px; white-space:pre-wrap}
        .user {background:#0b6cff14; border:1px solid #0b6cff44}
        .assistant {background:#111214; color:#fff; border:1px solid #2b2b2b}
        .sidebar-row {margin-bottom:12px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_processed_input" not in st.session_state:
        st.session_state.last_processed_input = None

    # Sidebar controls
    with st.sidebar:
        st.markdown("## Multi-Modal RAG")

        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.session_state.last_results = []
            st.session_state.last_processed_input = None

        st.markdown("---")
        # Uncomment the section below to enable PDF upload functionality
        # st.markdown("**📤 Upload & Ingest**")
        # uploaded_files = st.file_uploader(
        #     "Upload PDF files",
        #     type=["pdf"],
        #     accept_multiple_files=True,
        #     key="pdf_uploader"
        # )
        # 
        # if uploaded_files:
        #     st.info(f"Processing {len(uploaded_files)} file(s)...")
        #     ingest_progress = st.progress(0)
        #     ingest_status = st.empty()
        #     
        #     for idx, file in enumerate(uploaded_files):
        #         success, message = ingest_uploaded_pdf(file)
        #         
        #         if success:
        #             ingest_status.success(message)
        #         else:
        #             ingest_status.error(message)
        #         
        #         ingest_progress.progress((idx + 1) / len(uploaded_files))
        #     
        #     # Show completion message without trying to clear session state
        #     ingest_status.success("✓ All files processed! You can now ask questions about them.")
        # 
        # st.markdown("---")

        st.markdown("**Settings**")
        top_k = st.slider("Top K (results)", 1, 10, 5)
        show_brief = st.checkbox("Show Briefing (summary)", value=False)
        st.markdown("---")
        st.markdown("**Diagnostics**")
        if st.button("Show Last Results JSON"):
            st.json(st.session_state.get("last_results", {}))

    # Load chain
    try:
        qa_chain = load_chain()
    except Exception as e:
        qa_chain = None
        st.sidebar.error(f"LLM chain init failed: {e}")

    # Main layout
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("<div class='app-title'>Multi-Modal RAG — Chat</div>", unsafe_allow_html=True)
    with header_col2:
        st.empty()

    chat_col, info_col = st.columns([3, 1])

    with chat_col:
        # Render history
        result_idx = 0  # Track index for assistant responses only
        for msg in st.session_state.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")
                # Get the corresponding result for this assistant message
                if result_idx < len(st.session_state.last_results):
                    last_result = st.session_state.last_results[result_idx]
                    if last_result:
                        with st.expander("📚 View Source Citations"):
                            contexts = extract_context_list(last_result)
                            if contexts:
                                render_sources(contexts)
                            else:
                                st.write("No sources available for this response.")
                result_idx += 1

        # Input
        user_input = st.text_input("Ask a question", placeholder="Type your question and press Enter")

        if user_input and user_input != st.session_state.last_processed_input:
            # Mark this input as processed to avoid duplicate runs
            st.session_state.last_processed_input = user_input
            
            st.session_state.messages.append({"role": "user", "content": user_input})

            if qa_chain is None:
                st.session_state.messages.append({"role": "assistant", "content": "QA chain is not available."})
                st.session_state.last_results.append(None)
            else:
                with st.spinner("Thinking..."):
                    try:
                        # keep API of previous chain: input param
                        result = qa_chain.invoke({"input": user_input})
                    except Exception as exc:
                        err = f"Error running QA chain: {exc}"
                        st.session_state.messages.append({"role": "assistant", "content": err})
                        st.session_state.last_results.append(None)
                    else:
                        answer = None
                        if isinstance(result, dict):
                            answer = result.get("answer") or result.get("output_text") or result.get("text") or result.get("result")
                            if isinstance(answer, dict):
                                answer = answer.get("output_text") or str(answer)
                        else:
                            answer = str(result)

                        if not answer:
                            answer = str(result)

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.last_results.append(result if isinstance(result, dict) else {"context": []})
            
            # Force rerun to display the new message
            st.rerun()

            

    with info_col:
        st.markdown("### Controls")
        st.write("- Use the chat to ask questions about ingested documents.")
        st.write("- Click the citation expander under answers to inspect source snippets and pages.")
        st.markdown("---")



if __name__ == "__main__":
    main()
