"""
Validation & Trust Assessment Engine – Evaluates the reliability,
consistency, and relevance of each acquired piece of information
before it enters the knowledge base.
"""

import logging
from typing import Any, Dict, List, Tuple

import config
from modules.knowledge_acquisition import AcquisitionResult
from modules.knowledge_store import get_knowledge_store
from utils.embeddings import embed_text
from utils.llm import get_llm

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of validating a single acquisition."""

    def __init__(
        self,
        acquisition: AcquisitionResult,
        source_reliability: float,
        consistency_score: float,
        relevance_score: float,
        trust_score: float,
        accepted: bool,
        low_confidence: bool,
        rejection_reason: str = "",
    ):
        self.acquisition = acquisition
        self.source_reliability = source_reliability
        self.consistency_score = consistency_score
        self.relevance_score = relevance_score
        self.trust_score = trust_score
        self.accepted = accepted
        self.low_confidence = low_confidence
        self.rejection_reason = rejection_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.acquisition.title,
            "source_name": self.acquisition.source_name,
            "source_reliability": round(self.source_reliability, 3),
            "consistency_score": round(self.consistency_score, 3),
            "relevance_score": round(self.relevance_score, 3),
            "trust_score": round(self.trust_score, 3),
            "accepted": self.accepted,
            "low_confidence": self.low_confidence,
            "rejection_reason": self.rejection_reason,
        }


class ValidationEngine:
    """Evaluates acquired information for trust, consistency, relevance."""

    def __init__(self):
        self.store = get_knowledge_store()
        self.llm = get_llm()

    def validate_batch(
        self, acquisitions: List[AcquisitionResult], original_query: str
    ) -> List[ValidationResult]:
        """Validate a batch of acquisition results. Returns validated list."""
        results = []
        for acq in acquisitions:
            vr = self.validate(acq, original_query)
            results.append(vr)

        accepted = sum(1 for r in results if r.accepted)
        logger.info(
            "Validation complete: %d/%d accepted (%d low-confidence)",
            accepted,
            len(results),
            sum(1 for r in results if r.low_confidence),
        )
        return results

    def validate(self, acq: AcquisitionResult, original_query: str) -> ValidationResult:
        """Validate a single acquisition result."""

        # 1. Source reliability
        source_reliability = self._assess_source_reliability(acq)

        # 2. Relevance to original query
        relevance_score = self._assess_relevance(acq, original_query)

        # 3. Consistency with existing knowledge
        consistency_score = self._assess_consistency(acq)

        # 4. Compute final trust score
        trust_score = (
            config.TRUST_SOURCE_WEIGHT * source_reliability
            + config.TRUST_CONSISTENCY_WEIGHT * consistency_score
            + config.TRUST_RELEVANCE_WEIGHT * relevance_score
        )
        trust_score = round(trust_score, 4)

        # 5. Make acceptance decision
        accepted = True
        low_confidence = False
        rejection_reason = ""

        if relevance_score < config.RELEVANCE_MIN_THRESHOLD:
            accepted = False
            rejection_reason = f"Relevance score ({relevance_score:.2f}) below threshold ({config.RELEVANCE_MIN_THRESHOLD})"
        elif trust_score < config.TRUST_REJECT_THRESHOLD:
            accepted = False
            rejection_reason = f"Trust score ({trust_score:.2f}) below rejection threshold ({config.TRUST_REJECT_THRESHOLD})"
        elif trust_score < config.TRUST_LOW_CONFIDENCE_THRESHOLD:
            low_confidence = True

        return ValidationResult(
            acquisition=acq,
            source_reliability=source_reliability,
            consistency_score=consistency_score,
            relevance_score=relevance_score,
            trust_score=trust_score,
            accepted=accepted,
            low_confidence=low_confidence,
            rejection_reason=rejection_reason,
        )

    # ── Source Reliability ────────────────────────────────────────────
    def _assess_source_reliability(self, acq: AcquisitionResult) -> float:
        """Score based on source type, journal presence, citations, recency."""
        # Base score by source type
        base_scores = {
            "arxiv": config.SOURCE_TRUST_ARXIV,
            "wikipedia": config.SOURCE_TRUST_WIKIPEDIA,
            "semantic_scholar": config.SOURCE_TRUST_SEMANTIC_SCHOLAR,
        }
        score = base_scores.get(acq.source_name, 0.5)

        # Bonus for journal publication
        if acq.journal_ref and acq.journal_ref.strip():
            score = min(1.0, score + 0.1)

        # Bonus for citation count (Semantic Scholar)
        if acq.citation_count > 100:
            score = min(1.0, score + 0.1)
        elif acq.citation_count > 20:
            score = min(1.0, score + 0.05)

        # Recency bonus (simple check: if year > 2022)
        try:
            year = int(acq.published_date[:4])
            if year >= 2024:
                score = min(1.0, score + 0.05)
            elif year < 2015:
                score = max(0.0, score - 0.05)
        except (ValueError, IndexError):
            pass

        return score

    # ── Relevance ────────────────────────────────────────────────────
    def _assess_relevance(self, acq: AcquisitionResult, original_query: str) -> float:
        """Semantic similarity between the acquisition content and the query."""
        try:
            query_emb = embed_text(original_query)
            # Use title + content for better relevance matching
            eval_text = f"{acq.title}\n{acq.content[:1500]}"
            content_emb = embed_text(eval_text)

            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_emb, content_emb))
            norm_q = sum(a ** 2 for a in query_emb) ** 0.5
            norm_c = sum(b ** 2 for b in content_emb) ** 0.5
            if norm_q == 0 or norm_c == 0:
                return 0.0
            similarity = dot / (norm_q * norm_c)

            # Also check if query terms appear directly in the title (strong relevance signal)
            query_lower = original_query.lower()
            title_lower = acq.title.lower()
            query_terms = [t.strip() for t in query_lower.split() if len(t.strip()) > 2]
            term_hits = sum(1 for t in query_terms if t in title_lower)
            if query_terms and term_hits / len(query_terms) >= 0.5:
                # At least half the query terms appear in the title — boost relevance
                similarity = max(similarity, 0.60)

            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning("Relevance computation failed: %s", e)
            return 0.5  # neutral fallback

    # ── Consistency ──────────────────────────────────────────────────
    def _assess_consistency(self, acq: AcquisitionResult) -> float:
        """Check if the new information is consistent with existing knowledge."""
        try:
            # Find existing docs similar to this acquisition
            existing = self.store.search(acq.content[:500], top_k=3)

            if not existing:
                # No existing knowledge to compare → neutral
                return 0.7

            # If the most similar existing doc is very similar, that's consistent
            best_sim = max(d.get("similarity", 0) for d in existing)

            if best_sim > 0.8:
                # Very similar content exists — consistent (or redundant)
                return 0.9
            elif best_sim > 0.5:
                # Moderately related — use LLM for deeper check
                return self._llm_consistency_check(acq, existing)
            else:
                # Not much overlap — neutral
                return 0.7

        except Exception as e:
            logger.warning("Consistency check failed: %s", e)
            return 0.6

    def _llm_consistency_check(
        self, acq: AcquisitionResult, existing: List[Dict]
    ) -> float:
        """Ask LLM whether new info contradicts existing knowledge."""
        existing_text = "\n---\n".join(
            d.get("content", "")[:300] for d in existing[:3]
        )
        prompt = f"""Compare the following NEW information with EXISTING knowledge in our database.

NEW INFORMATION:
{acq.content[:500]}

EXISTING KNOWLEDGE:
{existing_text}

Respond with ONLY a JSON object:
{{
  "consistent": <true or false>,
  "confidence": <float 0.0 to 1.0>,
  "explanation": "<brief explanation>"
}}"""

        try:
            result = self.llm.generate_json(prompt)
            consistent = result.get("consistent", True)
            conf = float(result.get("confidence", 0.7))
            if consistent:
                return max(0.6, conf)
            else:
                return max(0.1, 1.0 - conf)
        except Exception:
            return 0.6


# ── Singleton ────────────────────────────────────────────────────────
_engine = None


def get_validation_engine() -> ValidationEngine:
    global _engine
    if _engine is None:
        _engine = ValidationEngine()
    return _engine
