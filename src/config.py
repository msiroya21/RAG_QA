import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Model configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # Legacy (kept for reference)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Groq Llama API key

EMBED_MODEL = "intfloat/e5-large-v2"  # retrieval-optimized embedding model
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq Llama model for QA chain

# Tesseract configuration (legacy OCR - not used)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)
if TESSERACT_CMD:
    os.environ["TESSDATA_PREFIX"] = str(Path(TESSERACT_CMD).parent)

# Vector DB configuration
USE_QDRANT = bool(os.getenv("QDRANT_URL"))
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Chunking defaults
CHUNK_SIZE = 600  # characters
CHUNK_OVERLAP = 120  # characters
TABLE_MAX_COLS = 20
TABLE_MAX_ROWS = 200

# Retrieval defaults
TOP_K = 5
RERANK_TOP_K = 10

# Reranker (cross-encoder)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def ensure_dirs() -> None:
    """Ensure expected directories exist."""
    for path in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        CHROMA_DB_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


