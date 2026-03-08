"""
Knowledge Gap Detector – Classifies the type of knowledge gap
detected by the Epistemic Analyzer and makes routing decisions.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import config
from modules.input_interface import Query
from utils.database import insert_gap

logger = logging.getLogger(__name__)


class GapDetectionResult:
    """Structured result from gap detection."""

    def __init__(
        self,
        gap_type: str,
        description: str,
        query: str,
        epistemic_data: Dict[str, Any],
        gap_id: Optional[int] = None,
    ):
        self.gap_type = gap_type
        self.description = description
        self.query = query
        self.epistemic_data = epistemic_data
        self.gap_id = gap_id  # set after logging to DB
        self.requires_learning = gap_type != config.GAP_TYPE_NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_type": self.gap_type,
            "description": self.description,
            "requires_learning": self.requires_learning,
            "query": self.query,
            "gap_id": self.gap_id,
            "epistemic_scores": {
                "retrieval": self.epistemic_data.get("retrieval_score"),
                "coverage": self.epistemic_data.get("coverage_score"),
                "llm": self.epistemic_data.get("llm_score"),
                "composite": self.epistemic_data.get("composite_score"),
            },
        }


class GapDetector:
    """Classifies knowledge gaps and routes queries."""

    def detect(self, query: Query, epistemic_data: Dict[str, Any]) -> GapDetectionResult:
        """
        Analyse the epistemic assessment and classify the gap.

        Returns a GapDetectionResult with the gap type and routing info.
        """
        if not epistemic_data.get("gap_detected", False):
            return GapDetectionResult(
                gap_type=config.GAP_TYPE_NONE,
                description="Knowledge base has sufficient coverage for this query.",
                query=query.normalised_text,
                epistemic_data=epistemic_data,
            )

        # Determine the specific gap type
        gap_type, description = self._classify_gap(query, epistemic_data)

        # Log the gap
        try:
            gap_id = insert_gap(
                query=query.normalised_text,
                gap_type=gap_type,
                description=description,
            )
        except Exception as e:
            logger.warning("Could not log gap: %s", e)
            gap_id = None

        result = GapDetectionResult(
            gap_type=gap_type,
            description=description,
            query=query.normalised_text,
            epistemic_data=epistemic_data,
            gap_id=gap_id,
        )

        logger.info("Gap detected: type=%s — %s", gap_type, description)
        return result

    def _classify_gap(
        self, query: Query, data: Dict[str, Any]
    ) -> tuple:
        """Return (gap_type, description) based on epistemic signals."""

        retrieval = data.get("retrieval_score", 0)
        coverage = data.get("coverage_score", 0)
        llm_score = data.get("llm_score", 0)
        llm_analysis = data.get("llm_analysis", {})
        docs = data.get("retrieved_docs", [])

        # 1. Missing Topic — knowledge base has nothing relevant
        #    Retrieval can be misleadingly high if irrelevant docs float up with
        #    marginal similarity. Coverage and LLM assessment are better signals.
        is_missing = (
            (retrieval < 0.15 and coverage < 0.1)                  # nothing at all
            or (coverage < 0.15 and llm_score < 0.3)              # coverage + LLM both say no
            or (retrieval < 0.50 and coverage < 0.10 and llm_score < 0.35)  # low across the board
        )
        if is_missing:
            return (
                config.GAP_TYPE_MISSING,
                f"No relevant information found in the knowledge base for: '{query.normalised_text}'. "
                f"Retrieval score ({retrieval:.2f}), coverage ({coverage:.2f}), LLM ({llm_score:.2f}) are all low.",
            )

        # 2. Check for contradictions (from LLM analysis)
        contradictions = llm_analysis.get("contradictions", "none")
        if isinstance(contradictions, str) and contradictions.lower() not in ("none", "no", "n/a", ""):
            return (
                config.GAP_TYPE_CONTRADICTORY,
                f"Contradictory information detected: {contradictions}",
            )

        # 3. Outdated information — check dates of retrieved docs
        if docs and self._check_staleness(docs):
            return (
                config.GAP_TYPE_OUTDATED,
                f"Retrieved documents appear outdated (older than {config.STALENESS_DAYS} days). "
                "Information may no longer be current.",
            )

        # 4. Shallow coverage — some info exists but insufficient
        if retrieval >= 0.15 and llm_score < 0.5:
            missing = llm_analysis.get("missing", "details not identified")
            return (
                config.GAP_TYPE_SHALLOW,
                f"Partial information exists but with significant gaps. "
                f"Missing: {missing}",
            )

        # Default to shallow if we reach here (gap was detected but no specific category)
        return (
            config.GAP_TYPE_SHALLOW,
            f"Composite confidence ({data.get('composite_score', 0):.2f}) is below threshold "
            f"({config.CONFIDENCE_THRESHOLD}). Knowledge base coverage is insufficient.",
        )

    def _check_staleness(self, docs: list) -> bool:
        """Check if the majority of retrieved docs are old."""
        stale_count = 0
        checked = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=config.STALENESS_DAYS)

        for doc in docs:
            meta = doc.get("metadata", {})
            update_str = meta.get("update_date", "")
            if not update_str:
                continue
            try:
                update_date = datetime.strptime(update_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                checked += 1
                if update_date < cutoff:
                    stale_count += 1
            except ValueError:
                continue

        if checked == 0:
            return False
        return (stale_count / checked) > 0.6


# ── Singleton ────────────────────────────────────────────────────────
_detector = None


def get_gap_detector() -> GapDetector:
    global _detector
    if _detector is None:
        _detector = GapDetector()
    return _detector
