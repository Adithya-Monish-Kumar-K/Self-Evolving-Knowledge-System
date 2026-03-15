"""
Self-Evolving Knowledge System — Streamlit Dashboard

Pages:
  1. Query Interface (main)
  2. Knowledge Base Explorer
  3. System Logs & Metrics
  4. Settings & Ingestion
"""

import sys
import os
import json
import logging
import time

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go

import config
from utils.database import init_db, get_stats, get_gaps, get_revisions, get_epistemic_logs, search_papers, count_papers, get_gap_with_entries
from utils.llm import get_llm

# ─── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Initialise DB on first run ──────────────────────────────────────
init_db()

# ─── Sidebar Navigation ─────────────────────────────────────────────
st.sidebar.title("🧠 SEKS")
st.sidebar.caption("Self-Evolving Knowledge System")
page = st.sidebar.radio(
    "Navigate",
    ["💬 Query", "📚 Knowledge Base", "📊 Logs & Metrics", "⚙️ Settings"],
)

# =====================================================================
# PAGE 1: QUERY INTERFACE
# =====================================================================
if page == "💬 Query":
    st.title("🧠 Self-Evolving Knowledge System")
    st.markdown("Ask a question. The system will assess its knowledge, detect gaps, learn autonomously, and reason.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area("Enter your query:", height=100, placeholder="e.g., What are recent advances in quantum error correction?")
    with col2:
        st.markdown("**Options**")
        baseline_mode = st.checkbox("Baseline mode (no gap detection)", value=False)
        show_trace = st.checkbox("Show full pipeline trace", value=True)

    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            # Check Ollama connectivity
            with st.spinner("Checking Ollama connection..."):
                try:
                    llm = get_llm()
                    if not llm.ping():
                        st.error(
                            f"❌ Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
                            f"Please ensure Ollama is running with model `{config.OLLAMA_MODEL}`.\n\n"
                            f"Run: `ollama run {config.OLLAMA_MODEL}`"
                        )
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Ollama error: {e}")
                    st.stop()

            # Run the pipeline
            with st.spinner("Running the Self-Evolving Knowledge pipeline..."):
                from pipeline import run_query
                start = time.time()
                response = run_query(query.strip(), baseline_mode=baseline_mode)
                elapsed = time.time() - start

            # ── Display Results ──────────────────────────────────────
            st.divider()

            # Confidence gauge
            col_ans, col_gauge = st.columns([3, 1])
            with col_gauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=response.confidence * 100,
                    title={"text": "Confidence"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "#ff4444"},
                            {"range": [30, 65], "color": "#ffaa00"},
                            {"range": [65, 100], "color": "#44bb44"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.8,
                            "value": config.CONFIDENCE_THRESHOLD * 100,
                        },
                    },
                ))
                fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)

            with col_ans:
                st.subheader("Answer")
                st.markdown(response.answer)

                if response.limitations:
                    st.caption(f"⚠️ Limitations: {response.limitations}")

            # Sources
            if response.sources:
                with st.expander(f"📎 Sources ({len(response.sources)})", expanded=False):
                    for src in response.sources:
                        st.markdown(
                            f"- **[Source {src.get('index', '?')}]** {src.get('title', 'Unknown')} "
                            f"| Type: `{src.get('source_type', '')}` "
                            f"| Trust: `{src.get('trust_score', 'N/A')}` "
                            f"| Similarity: `{src.get('similarity', 0):.3f}`"
                        )

            # Reasoning chain
            if response.reasoning_chain:
                with st.expander("🔗 Reasoning Chain", expanded=False):
                    for step in response.reasoning_chain:
                        st.markdown(f"- {step}")

            # Knowledge gaps addressed
            if response.knowledge_gaps_addressed:
                with st.expander("🎓 Knowledge Gaps Addressed", expanded=True):
                    for gap in response.knowledge_gaps_addressed:
                        st.info(
                            f"**Gap Type:** {gap.get('gap_type', 'unknown')}\n\n"
                            f"**Description:** {gap.get('description', '')}\n\n"
                            f"**New entries added:** {gap.get('entries_added', 0)}"
                        )

            # Pipeline trace
            if show_trace and response.pipeline_trace:
                with st.expander("🔧 Pipeline Trace", expanded=False):
                    for step in response.pipeline_trace.steps:
                        status_icon = {"completed": "✅", "skipped": "⏭️", "failed": "❌"}.get(step["status"], "⬜")
                        st.markdown(f"{status_icon} **{step['module']}** — {step['summary']}")
                        if step.get("data"):
                            st.json(step["data"])

            st.caption(f"⏱️ Total time: {elapsed:.1f}s")


# =====================================================================
# PAGE 2: KNOWLEDGE BASE EXPLORER
# =====================================================================
elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base Explorer")

    tab1, tab2, tab3 = st.tabs(["Papers", "Knowledge Graph", "Statistics"])

    # ── Papers tab ───────────────────────────────────────────────────
    with tab1:
        search_query = st.text_input("Search papers by title or abstract:")
        if search_query:
            papers = search_papers(search_query, limit=50)
            if papers:
                st.success(f"Found {len(papers)} papers")
                for p in papers:
                    with st.expander(f"📄 {p.get('title', 'Untitled')[:100]}"):
                        st.markdown(f"**ID:** {p.get('id', '')}")
                        st.markdown(f"**Authors:** {p.get('authors', '')[:200]}")
                        st.markdown(f"**Categories:** {p.get('categories', '')}")
                        st.markdown(f"**Abstract:** {p.get('abstract', '')[:500]}")
                        if p.get("doi"):
                            st.markdown(f"**DOI:** {p['doi']}")
                        if p.get("journal_ref"):
                            st.markdown(f"**Journal:** {p['journal_ref']}")
                        st.caption(f"Updated: {p.get('update_date', 'N/A')} | Ingested: {p.get('ingested_at', 'N/A')}")
            else:
                st.info("No papers found matching your query.")
        else:
            total = count_papers()
            st.info(f"Total papers in database: **{total}**. Enter a search term to explore.")

    # ── Knowledge Graph tab ──────────────────────────────────────────
    with tab2:
        st.markdown("### Knowledge Graph")
        try:
            from utils.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            stats = kg.stats()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Nodes", stats.get("total_nodes", 0))
            col2.metric("Total Edges", stats.get("total_edges", 0))
            col3.metric("Papers", stats.get("paper", 0))
            col4.metric("Categories", stats.get("category", 0))

            # Search the graph
            graph_query = st.text_input("Search concepts in the graph:")
            if graph_query:
                terms = graph_query.lower().split()
                related = kg.find_related_concepts(terms, top_k=20)
                if related:
                    for node in related:
                        st.markdown(
                            f"- **[{node.get('node_type', '')}]** {node.get('name', node.get('title', node.get('id', '')))[:100]} "
                            f"(relevance: {node.get('score', 0)})"
                        )
                else:
                    st.info("No matching concepts found.")
        except Exception as e:
            st.warning(f"Could not load knowledge graph: {e}")

    # ── Statistics tab ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Database Statistics")
        try:
            stats = get_stats()
            col1, col2, col3 = st.columns(3)
            col1.metric("Papers", stats.get("papers", 0))
            col1.metric("Knowledge Entries", stats.get("knowledge_entries", 0))
            col2.metric("Active Entries", stats.get("active_entries", 0))
            col2.metric("Logical Rules", stats.get("logical_rules", 0))
            col3.metric("Revision History", stats.get("revision_history", 0))
            col3.metric("Gaps Resolved", f"{stats.get('gaps_resolved', 0)}/{stats.get('gap_log', 0)}")

            # ChromaDB
            try:
                from modules.knowledge_store import get_knowledge_store
                store = get_knowledge_store()
                st.metric("ChromaDB Documents", store.count())
            except Exception:
                pass
        except Exception as e:
            st.error(f"Could not load stats: {e}")


# =====================================================================
# PAGE 3: LOGS & METRICS
# =====================================================================
elif page == "📊 Logs & Metrics":
    st.title("📊 System Logs & Metrics")

    tab1, tab2, tab3 = st.tabs(["Epistemic Log", "Gap Log", "Revision History"])

    # ── Epistemic Log ────────────────────────────────────────────────
    with tab1:
        st.markdown("### Epistemic Assessment History")
        logs = get_epistemic_logs(limit=50)
        if logs:
            import pandas as pd
            df = pd.DataFrame(logs)
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp", ascending=False)

            # Metrics
            if len(df) > 0:
                avg_composite = df["composite_score"].mean() if "composite_score" in df.columns else 0
                gap_rate = df["gap_detected"].mean() * 100 if "gap_detected" in df.columns else 0
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Assessments", len(df))
                col2.metric("Avg Confidence", f"{avg_composite:.2f}")
                col3.metric("Gap Detection Rate", f"{gap_rate:.1f}%")

            st.dataframe(df, use_container_width=True)

            # Confidence over time chart
            if "composite_score" in df.columns and len(df) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=df["composite_score"].tolist(),
                    mode="lines+markers",
                    name="Composite Confidence",
                ))
                fig.add_hline(y=config.CONFIDENCE_THRESHOLD, line_dash="dash", line_color="red",
                              annotation_text="Threshold")
                fig.update_layout(title="Confidence Scores Over Time", yaxis_title="Score", height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No epistemic assessments logged yet. Ask a question to see data here.")

    # ── Gap Log ──────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Knowledge Gap History")
        show_resolved = st.selectbox("Filter:", ["All", "Unresolved", "Resolved"])
        resolved_filter = None
        if show_resolved == "Unresolved":
            resolved_filter = 0
        elif show_resolved == "Resolved":
            resolved_filter = 1
        gaps = get_gaps(resolved=resolved_filter, limit=50)
        if gaps:
            for g in gaps:
                icon = "✅" if g.get("resolved") else "❓"
                with st.expander(f"{icon} [{g.get('gap_type', '')}] {g.get('query', '')[:80]}"):
                    st.markdown(f"**Type:** {g.get('gap_type', '')}")
                    st.markdown(f"**Description:** {g.get('description', '')}")
                    st.markdown(f"**Resolved:** {'Yes' if g.get('resolved') else 'No'}")
                    if g.get("resolution_info"):
                        st.markdown(f"**Resolution:** {g['resolution_info']}")
                    st.caption(f"Timestamp: {g.get('timestamp', '')}")

                    # Show resolved knowledge entries
                    if g.get("resolved"):
                        gap_detail = get_gap_with_entries(g["id"])
                        if gap_detail and gap_detail.get("resolved_entries"):
                            st.divider()
                            st.markdown("**📄 Knowledge Entries Added to Resolve This Gap:**")
                            for entry in gap_detail["resolved_entries"]:
                                entry_title = entry.get("title", "Untitled")
                                entry_source = entry.get("source_type", "unknown")
                                entry_trust = entry.get("trust_score", "N/A")
                                entry_conf = entry.get("confidence", "N/A")
                                content_preview = (entry.get("content") or "")[:400]

                                st.markdown(
                                    f"**🔹 {entry_title}**\n\n"
                                    f"- Source: `{entry_source}` | Trust: `{entry_trust}` | Confidence: `{entry_conf}`\n"
                                    f"- Entry ID: `{entry.get('id', '?')}` | Source ID: `{entry.get('source_id', 'N/A')}`"
                                )
                                if content_preview:
                                    st.text_area(
                                        f"Content preview (entry #{entry.get('id', '?')})",
                                        value=content_preview,
                                        height=100,
                                        disabled=True,
                                        key=f"gap_{g['id']}_entry_{entry.get('id', 0)}",
                                    )
        else:
            st.info("No gaps logged yet.")

    # ── Revision History ─────────────────────────────────────────────
    with tab3:
        st.markdown("### Belief Revision Timeline")
        revisions = get_revisions(limit=50)
        if revisions:
            for r in revisions:
                with st.expander(f"🔄 Entry #{r.get('entry_id', '?')} — {r.get('timestamp', '')}"):
                    st.markdown(f"**Reason:** {r.get('reason', '')}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Old Content:**")
                        st.text(r.get("old_content", "")[:500])
                    with col2:
                        st.markdown("**New Content:**")
                        st.text(r.get("new_content", "")[:500])
        else:
            st.info("No revisions recorded yet.")


# =====================================================================
# PAGE 4: SETTINGS & INGESTION
# =====================================================================
elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Data Ingestion")

    tab1, tab2, tab3 = st.tabs(["Configuration", "API Sources", "Dataset Ingestion"])

    # ── Configuration ────────────────────────────────────────────────
    with tab1:
        st.markdown("### System Configuration")
        st.markdown("These settings are the current runtime defaults from `config.py`.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**LLM Settings**")
            st.text(f"Model: {config.OLLAMA_MODEL}")
            st.text(f"Fallback Model: {config.OLLAMA_FALLBACK_MODEL}")
            st.text(f"Temperature: {config.OLLAMA_TEMPERATURE}")
            st.text(f"Base URL: {config.OLLAMA_BASE_URL}")

            st.markdown("**Embedding**")
            st.text(f"Model: {config.EMBEDDING_MODEL}")
            st.text(f"Device: {config.EMBEDDING_DEVICE}")
            st.text(f"Dimension: {config.EMBEDDING_DIMENSION}")

        with col2:
            st.markdown("**Epistemic Thresholds**")
            st.text(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
            st.text(f"Retrieval Weight: {config.RETRIEVAL_WEIGHT}")
            st.text(f"Coverage Weight: {config.COVERAGE_WEIGHT}")
            st.text(f"LLM Assessment Weight: {config.LLM_ASSESSMENT_WEIGHT}")

            st.markdown("**Trust Settings**")
            st.text(f"Reject Threshold: {config.TRUST_REJECT_THRESHOLD}")
            st.text(f"Low Confidence Threshold: {config.TRUST_LOW_CONFIDENCE_THRESHOLD}")
            st.text(f"Relevance Min: {config.RELEVANCE_MIN_THRESHOLD}")

        # Ollama health check
        st.divider()
        if st.button("🏥 Check Ollama Health"):
            try:
                llm = get_llm()
                if llm.ping():
                    st.success(f"✅ Ollama is running and model `{config.OLLAMA_MODEL}` is responsive.")
                else:
                    st.error(f"❌ Ollama responded but model `{config.OLLAMA_MODEL}` did not.")
            except Exception as e:
                st.error(f"❌ Cannot connect to Ollama: {e}")

    # ── Dataset Ingestion ────────────────────────────────────────────
    with tab2:
        st.markdown("### External API Sources")
        st.markdown(
            "Configure which external sources the system uses to fill knowledge gaps. "
            "Changes are saved to the `.env` file and take effect immediately."
        )

        # Load current .env values
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        env_values = {}
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        env_values[key.strip()] = val.strip()

        st.markdown("#### Source Toggles")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            arxiv_on = st.toggle(
                "🔬 arXiv",
                value=env_values.get("ARXIV_ENABLED", "true").lower() == "true",
                help="Search arXiv for academic papers",
            )
        with col_b:
            wiki_on = st.toggle(
                "📖 Wikipedia",
                value=env_values.get("WIKIPEDIA_ENABLED", "true").lower() == "true",
                help="Search Wikipedia for general knowledge",
            )
        with col_c:
            scholar_on = st.toggle(
                "🎓 Semantic Scholar",
                value=env_values.get("SEMANTIC_SCHOLAR_ENABLED", "true").lower() == "true",
                help="Search Semantic Scholar for papers with citation data",
            )

        st.markdown("#### API Configuration")
        arxiv_url = st.text_input("arXiv API URL:", value=env_values.get("ARXIV_API_URL", "http://export.arxiv.org/api/query"))
        wiki_url = st.text_input("Wikipedia API URL:", value=env_values.get("WIKIPEDIA_API_URL", "https://en.wikipedia.org/w/api.php"))
        scholar_url = st.text_input("Semantic Scholar API URL:", value=env_values.get("SEMANTIC_SCHOLAR_API_URL", "https://api.semanticscholar.org/graph/v1"))
        scholar_key = st.text_input("Semantic Scholar API Key (optional):", value=env_values.get("SEMANTIC_SCHOLAR_API_KEY", ""), type="password")

        st.markdown("#### Acquisition Settings")
        col_d, col_e = st.columns(2)
        with col_d:
            max_results = st.number_input("Max results per source:", min_value=1, max_value=50, value=int(env_values.get("ACQUISITION_MAX_RESULTS", "5")))
        with col_e:
            timeout = st.number_input("API timeout (seconds):", min_value=5, max_value=120, value=int(env_values.get("ACQUISITION_TIMEOUT", "30")))

        if st.button("💾 Save API Settings", type="primary"):
            new_env = {
                "ARXIV_ENABLED": str(arxiv_on).lower(),
                "WIKIPEDIA_ENABLED": str(wiki_on).lower(),
                "SEMANTIC_SCHOLAR_ENABLED": str(scholar_on).lower(),
                "ARXIV_API_URL": arxiv_url,
                "WIKIPEDIA_API_URL": wiki_url,
                "SEMANTIC_SCHOLAR_API_URL": scholar_url,
                "SEMANTIC_SCHOLAR_API_KEY": scholar_key,
                "ACQUISITION_MAX_RESULTS": str(max_results),
                "ACQUISITION_TIMEOUT": str(timeout),
            }
            # Merge with existing env values (preserve keys not shown here)
            merged = {**env_values, **new_env}
            # Write .env file
            try:
                with open(env_path, "w") as f:
                    f.write("# Self-Evolving Knowledge System — Environment Configuration\n")
                    f.write("# Auto-saved from UI\n\n")
                    for k, v in merged.items():
                        f.write(f"{k}={v}\n")
                # Reload config values in-process
                config.ARXIV_ENABLED = arxiv_on
                config.WIKIPEDIA_ENABLED = wiki_on
                config.SEMANTIC_SCHOLAR_ENABLED = scholar_on
                config.ACQUISITION_MAX_RESULTS = max_results
                config.ACQUISITION_TIMEOUT = timeout
                st.success("✅ API settings saved to `.env`. Changes are active now.")
            except Exception as e:
                st.error(f"❌ Failed to save settings: {e}")

        # Show current status summary
        st.divider()
        st.markdown("#### Current Source Status")
        status_data = {
            "Source": ["arXiv", "Wikipedia", "Semantic Scholar"],
            "Enabled": [
                "✅" if config.ARXIV_ENABLED else "❌",
                "✅" if config.WIKIPEDIA_ENABLED else "❌",
                "✅" if config.SEMANTIC_SCHOLAR_ENABLED else "❌",
            ],
            "API URL": [
                getattr(config, "ARXIV_API_URL", "default"),
                getattr(config, "WIKIPEDIA_API_URL", "default"),
                getattr(config, "SEMANTIC_SCHOLAR_API_URL", "default"),
            ],
        }
        st.table(status_data)

    # ── Dataset Ingestion ────────────────────────────────────────────
    with tab3:
        st.markdown("### Ingest arXiv Dataset")
        st.markdown(
            "Upload your arXiv dataset (JSON, one record per line) or specify a file path. "
            "The system will populate the SQLite database, ChromaDB vector store, and knowledge graph."
        )

        method = st.radio("Method:", ["Upload file", "File path"])

        max_papers = st.number_input("Max papers to ingest (0 = all):", min_value=0, value=0, step=1000)

        if method == "Upload file":
            uploaded = st.file_uploader("Upload JSON dataset:", type=["json"])
            if uploaded and st.button("🚀 Start Ingestion", type="primary"):
                # Save uploaded file
                save_path = os.path.join(config.DATA_DIR, uploaded.name)
                os.makedirs(config.DATA_DIR, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.info(f"File saved to `{save_path}`. Starting ingestion...")

                with st.spinner("Ingesting dataset... This may take several minutes."):
                    from modules.knowledge_store import ingest_arxiv_dataset
                    ingest_arxiv_dataset(
                        save_path,
                        max_papers=max_papers if max_papers > 0 else None,
                    )
                st.success("✅ Ingestion complete!")
                st.rerun()

        else:
            filepath = st.text_input("Path to JSON file:", value=config.DATASET_PATH)
            if st.button("🚀 Start Ingestion", type="primary"):
                if not os.path.exists(filepath):
                    st.error(f"File not found: {filepath}")
                else:
                    with st.spinner("Ingesting dataset... This may take several minutes."):
                        from modules.knowledge_store import ingest_arxiv_dataset
                        ingest_arxiv_dataset(
                            filepath,
                            max_papers=max_papers if max_papers > 0 else None,
                        )
                    st.success("✅ Ingestion complete!")
                    st.rerun()

        # Show current ingestion status
        st.divider()
        st.markdown("### Current Status")
        try:
            stats = get_stats()
            col1, col2, col3 = st.columns(3)
            col1.metric("Papers in DB", stats.get("papers", 0))
            col2.metric("Knowledge Entries", stats.get("knowledge_entries", 0))
            col3.metric("Logical Rules", stats.get("logical_rules", 0))
        except Exception:
            st.info("Database not yet initialised.")
