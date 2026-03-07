"""
Epistemic State Analyzer – Evaluates the system's confidence about
a query using three signals: retrieval similarity, knowledge graph
coverage, and LLM self-assessment. Produces a composite confidence
score and logs every assessment for calibration over time.
"""

import logging
from typing import Any, Dict, List, Tuple

import config
from modules.input_interface import Query
from modules.knowledge_store import get_knowledge_store
from utils.knowledge_graph import get_knowledge_graph
from utils.llm import get_llm
from utils.database import insert_epistemic_log

logger = logging.getLogger(__name__)

# ── Structured prompt for LLM self-assessment ────────────────────────
_LLM_ASSESSMENT_PROMPT = """You are an epistemic analysis engine. Given a user query and the top retrieved context passages from our knowledge base, assess how well our existing knowledge can answer the query.

USER QUERY:
{query}

RETRIEVED CONTEXT (top results from knowledge base):
{context}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "confidence": <float 0.0 to 1.0 – how confidently our context can answer this query>,
  "coverage": "<brief description of what aspects are covered>",
  "missing": "<brief description of what information is missing or weak>",
  "contradictions": "<any contradictions detected, or 'none'>"
}}
"""


class EpistemicAnalyzer:
    """Multi-signal confidence assessment for queries."""

    def __init__(self):
        self.store = get_knowledge_store()
        self.kg = get_knowledge_graph()
        self.llm = get_llm()

    def assess(self, query: Query) -> Dict[str, Any]:
        """
        Run a full epistemic assessment on the query.

        Returns a dict with:
            retrieval_score, coverage_score, llm_score, composite_score,
            retrieved_docs, llm_analysis, gap_detected
        """
        # 1. Retrieval confidence
        retrieved_docs = self.store.search(query.normalised_text, top_k=config.CHROMA_SEARCH_TOP_K)
        retrieval_score = self._compute_retrieval_score(retrieved_docs)

        # 2. Knowledge graph coverage
        coverage_score = self._compute_coverage_score(query.key_terms)

        # 3. LLM self-assessment
        llm_score, llm_analysis = self._compute_llm_score(query.normalised_text, retrieved_docs)

        # 4. Composite
        composite = (
            config.RETRIEVAL_WEIGHT * retrieval_score
            + config.COVERAGE_WEIGHT * coverage_score
            + config.LLM_ASSESSMENT_WEIGHT * llm_score
        )
        composite = round(composite, 4)

        gap_detected = composite < config.CONFIDENCE_THRESHOLD

        # 5. Log to SQLite
        try:
            insert_epistemic_log(
                query=query.normalised_text,
                retrieval=retrieval_score,
                coverage=coverage_score,
                llm_score=llm_score,
                composite=composite,
                gap_detected=gap_detected,
            )
        except Exception as e:
            logger.warning("Could not log epistemic assessment: %s", e)

        result = {
            "retrieval_score": round(retrieval_score, 4),
            "coverage_score": round(coverage_score, 4),
            "llm_score": round(llm_score, 4),
            "composite_score": composite,
            "gap_detected": gap_detected,
            "retrieved_docs": retrieved_docs,
            "llm_analysis": llm_analysis,
        }
        logger.info(
            "Epistemic assessment — composite=%.3f gap=%s (ret=%.2f cov=%.2f llm=%.2f)",
            composite, gap_detected, retrieval_score, coverage_score, llm_score,
        )
        return result

    # ── Signal 1: Retrieval confidence ───────────────────────────────
    def _compute_retrieval_score(self, docs: List[Dict]) -> float:
        """Score based on cosine similarities of top-k retrieved docs."""
        if not docs:
            return 0.0
        similarities = [d.get("similarity", 0.0) for d in docs]
        # Weighted: best match contributes more
        if len(similarities) == 1:
            return max(0.0, min(1.0, similarities[0]))
        best = max(similarities)
        avg = sum(similarities) / len(similarities)
        score = 0.6 * best + 0.4 * avg
        return max(0.0, min(1.0, score))

    # ── Signal 2: Knowledge graph coverage ───────────────────────────
    def _compute_coverage_score(self, key_terms: List[str]) -> float:
        """Check how well-represented the query terms are in the graph."""
        if not key_terms:
            return 0.0

        scores = []
        for term in key_terms:
            # Check category density
            density = self.kg.get_category_density(term)
            if density > 0:
                scores.append(density)
            else:
                # Check if any related concepts exist
                related = self.kg.find_related_concepts([term], top_k=3)
                if related:
                    scores.append(min(len(related) / 3.0, 1.0) * 0.5)
                else:
                    scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    # ── Signal 3: LLM self-assessment ────────────────────────────────
    def _compute_llm_score(
        self, query_text: str, docs: List[Dict]
    ) -> Tuple[float, Dict]:
        """Ask the LLM to rate how well the context answers the query."""
        if not docs:
            context_str = "(No relevant documents found in the knowledge base.)"
        else:
            context_parts = []
            for i, d in enumerate(docs[:5], 1):
                sim = d.get("similarity", 0)
                content = d.get("content", "")[:500]
                meta = d.get("metadata", {})
                title = meta.get("title", "Untitled")
                context_parts.append(
                    f"[{i}] (similarity={sim:.2f}) {title}\n{content}"
                )
            context_str = "\n\n".join(context_parts)

        prompt = _LLM_ASSESSMENT_PROMPT.format(query=query_text, context=context_str)

        try:
            analysis = self.llm.generate_json(prompt)
            score = float(analysis.get("confidence", 0.5))
            score = max(0.0, min(1.0, score))
            return score, analysis
        except Exception as e:
            logger.warning("LLM assessment failed: %s. Defaulting to 0.5", e)
            return 0.5, {"confidence": 0.5, "error": str(e)}


# ── Singleton ────────────────────────────────────────────────────────
_analyzer = None


def get_epistemic_analyzer() -> EpistemicAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = EpistemicAnalyzer()
    return _analyzer
