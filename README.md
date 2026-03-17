# Self-Evolving Knowledge System (Autonomous Learning AI)

An AI system that **knows what it doesn't know**. It detects knowledge gaps, autonomously retrieves and validates new information from external sources, updates its knowledge base with version control, and reasons over evolved knowledge — all running locally.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Input Interface    │  ← Preprocessing, key-term extraction
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Epistemic Analyzer   │  ← Multi-signal confidence assessment
│ (Retrieval+Graph+LLM)│     (retrieval, coverage, LLM self-assessment)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│   Gap Detector       │  ← Classifies: Missing / Shallow / Outdated / Contradictory / No Gap
└────────┬────────────┘
         │
    ┌────┴────┐
    │ No Gap  │ Gap Detected
    │         ▼
    │  ┌──────────────────┐
    │  │ Knowledge         │  ← arXiv API, Wikipedia, Semantic Scholar
    │  │ Acquisition       │
    │  └────────┬─────────┘
    │           ▼
    │  ┌──────────────────┐
    │  │ Validation &      │  ← Source reliability, consistency, relevance
    │  │ Trust Assessment  │
    │  └────────┬─────────┘
    │           ▼
    │  ┌──────────────────┐
    │  │ Belief Revision   │  ← Version control, conflict resolution
    │  │ & Evolution       │
    │  └────────┬─────────┘
    │           │
    └─────┬─────┘
          ▼
┌─────────────────────┐
│ Reasoning Engine     │  ← Context assembly + structured LLM reasoning
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Output Interface     │  ← Answer + confidence + sources + reasoning chain
└─────────────────────┘
```

## Tech Stack

| Component           | Technology                |
|---------------------|---------------------------|
| LLM                 | Ollama (qwen2.5:7b Q4)   |
| Embeddings          | all-MiniLM-L6-v2 (CPU)   |
| Vector Store        | ChromaDB                  |
| Knowledge Graph     | NetworkX (GraphML)        |
| Relational DB       | SQLite                    |
| Orchestration       | LangChain                 |
| Web UI              | Streamlit                 |
| External Sources    | arXiv API, Wikipedia, Semantic Scholar |

## Hardware Requirements

- **GPU:** NVIDIA GPU with ≥ 6GB VRAM (e.g., RTX 4050)
- **RAM:** ≥ 16GB
- **Disk:** ~10GB free space
- Embeddings run on **CPU** to keep VRAM free for the LLM

## Quick Start

### 1. Setup

```powershell
cd src
python setup.py
```

This creates a virtual environment, installs all dependencies (CPU-only PyTorch), and initialises the database.

### 2. Install & Start Ollama

Download from [ollama.com](https://ollama.com), then:

```powershell
ollama pull qwen2.5:7b
ollama serve
```

### 3. Run the App

```powershell
cd src
.venv\Scripts\activate
streamlit run app.py
```

### 4. Ingest Dataset

In the Streamlit app, go to **⚙️ Settings → Dataset Ingestion** and upload your arXiv JSON dataset.

Or from the terminal:

```python
from modules.knowledge_store import ingest_arxiv_dataset
ingest_arxiv_dataset("data/arxiv-metadata.json", max_papers=10000)
```

### 5. Run Tests

```powershell
pytest tests/ -v
```

## Project Structure

```
src/
├── app.py                  # Streamlit dashboard (4 pages)
├── config.py               # All configuration settings
├── pipeline.py             # Main pipeline orchestrator
├── setup.py                # One-click setup script
├── requirements.txt        # Python dependencies
├── data/                   # Dataset files
├── knowledge_base/
│   ├── chroma_db/          # ChromaDB persistent store
│   ├── knowledge.db        # SQLite database
│   └── graph/              # NetworkX graph files
├── modules/
│   ├── input_interface.py          # Query preprocessing
│   ├── epistemic_analyzer.py       # Confidence assessment
│   ├── gap_detector.py             # Gap classification & routing
│   ├── knowledge_acquisition.py    # External source retrieval
│   ├── validation_engine.py        # Trust & consistency validation
│   ├── knowledge_store.py          # ChromaDB + ingestion pipeline
│   ├── belief_revision.py          # Version control & conflict resolution
│   ├── reasoning_engine.py         # Context assembly & LLM reasoning
│   └── output_interface.py         # Structured response builder
├── utils/
│   ├── llm.py              # Ollama LLM wrapper
│   ├── embeddings.py       # Sentence-transformers wrapper (CPU)
│   ├── database.py         # SQLite schema & CRUD
│   └── knowledge_graph.py  # NetworkX graph management
└── tests/
    ├── test_epistemic.py           # Epistemic awareness tests
    ├── test_acquisition.py         # Acquisition & validation tests
    ├── test_belief_revision.py     # Belief revision tests
    └── test_e2e.py                 # End-to-end & latency tests
```

## Evaluation Experiments

| Test | Objective | Metrics |
|------|-----------|---------|
| Baseline vs. Epistemic | Gap detection vs. hallucination | Hallucination Rate, Confidence Calibration Error |
| Autonomous Acquisition | Learning new concepts | Retrieval Precision/Recall, Validation Accuracy |
| Belief Revision | Updating outdated knowledge | Update Success Rate, Retention Rate |
| End-to-End Latency | Pipeline performance | Time-to-Resolution (TTR) |
