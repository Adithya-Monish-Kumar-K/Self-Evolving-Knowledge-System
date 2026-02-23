"""
Configuration for the Self-Evolving Knowledge System.
Single source of truth for all settings.
Loads overrides from .env when present.
"""

import os
from dotenv import load_dotenv

# Load .env from same directory as this file
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot.json")
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_DIR = os.path.join(KB_DIR, "chroma_db")
SQLITE_PATH = os.path.join(KB_DIR, "knowledge.db")
GRAPH_DIR = os.path.join(KB_DIR, "graph")
GRAPH_PATH = os.path.join(GRAPH_DIR, "knowledge_graph.graphml")

# ─── LLM (Ollama) ───────────────────────────────────────────────────
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_FALLBACK_MODEL = _env("OLLAMA_FALLBACK_MODEL", "phi3:mini")
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.1)
OLLAMA_REQUEST_TIMEOUT = _env_int("OLLAMA_REQUEST_TIMEOUT", 120)

# ─── Embedding Model ────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"               # Keep VRAM free for Ollama
EMBEDDING_DIMENSION = 384              # all-MiniLM-L6-v2 output dim

# ─── ChromaDB ────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "knowledge_base"
CHROMA_SEARCH_TOP_K = 5

# ─── Epistemic Assessment ───────────────────────────────────────────
CONFIDENCE_THRESHOLD = _env_float("CONFIDENCE_THRESHOLD", 0.65)
RETRIEVAL_WEIGHT = 0.35
COVERAGE_WEIGHT = 0.25
LLM_ASSESSMENT_WEIGHT = 0.40

# ─── Gap Detection ──────────────────────────────────────────────────
GAP_TYPE_MISSING = "missing_topic"
GAP_TYPE_SHALLOW = "shallow_coverage"
GAP_TYPE_OUTDATED = "outdated_information"
GAP_TYPE_CONTRADICTORY = "contradictory_information"
GAP_TYPE_NONE = "no_gap"

STALENESS_DAYS = _env_int("STALENESS_DAYS", 365 * 2)

# ─── Validation & Trust ─────────────────────────────────────────────
TRUST_SOURCE_WEIGHT = 0.40
TRUST_CONSISTENCY_WEIGHT = 0.30
TRUST_RELEVANCE_WEIGHT = 0.30

TRUST_REJECT_THRESHOLD = 0.30          # Below → reject
TRUST_LOW_CONFIDENCE_THRESHOLD = 0.60  # Below → accept with low confidence
RELEVANCE_MIN_THRESHOLD = 0.40         # Min similarity to accept

# ─── Source Defaults ─────────────────────────────────────────────────
SOURCE_TRUST_ARXIV = 0.80
SOURCE_TRUST_WIKIPEDIA = 0.60
SOURCE_TRUST_SEMANTIC_SCHOLAR = 0.75

# ─── External APIs ──────────────────────────────────────────────────
ARXIV_ENABLED = _env_bool("ARXIV_ENABLED", True)
ARXIV_API_URL = _env("ARXIV_API_URL", "http://export.arxiv.org/api/query")
ARXIV_MAX_RESULTS = _env_int("ARXIV_MAX_RESULTS", 10)
ARXIV_RATE_LIMIT_SECONDS = _env_float("ARXIV_RATE_LIMIT_SECONDS", 3.0)

WIKIPEDIA_ENABLED = _env_bool("WIKIPEDIA_ENABLED", True)
WIKIPEDIA_API_URL = _env("WIKIPEDIA_API_URL", "https://en.wikipedia.org/w/api.php")

SEMANTIC_SCHOLAR_ENABLED = _env_bool("SEMANTIC_SCHOLAR_ENABLED", True)
SEMANTIC_SCHOLAR_API_URL = _env("SEMANTIC_SCHOLAR_API_URL", "https://api.semanticscholar.org/graph/v1")
SEMANTIC_SCHOLAR_API_KEY = _env("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_RATE_LIMIT = _env_int("SEMANTIC_SCHOLAR_RATE_LIMIT", 100)

ACQUISITION_MAX_RESULTS = _env_int("ACQUISITION_MAX_RESULTS", 5)
ACQUISITION_TIMEOUT = _env_int("ACQUISITION_TIMEOUT", 30)

# ─── Belief Revision ────────────────────────────────────────────────
CONFLICT_TRUST_MARGIN = 0.10           # If trust diff < this → keep both

# ─── Dataset Ingestion ───────────────────────────────────────────────
INGESTION_BATCH_SIZE = 500

# ─── UI ──────────────────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "Self-Evolving Knowledge System"
STREAMLIT_PAGE_ICON = "🧠"
