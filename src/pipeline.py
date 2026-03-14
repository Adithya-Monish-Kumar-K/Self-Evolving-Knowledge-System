"""
Main Pipeline Orchestrator – Connects all modules into the
Self-Evolving Knowledge System pipeline.

Flow:
  Input → Epistemic Assessment → Gap Detection
    ├─ No Gap  → Reasoning → Output
    └─ Gap     → Acquisition → Validation → Belief Revision → Reasoning → Output
"""

import logging
import time
from typing import Optional

from modules.input_interface import preprocess_query, Query
from modules.epistemic_analyzer import get_epistemic_analyzer
from modules.gap_detector import get_gap_detector, GapDetectionResult
from modules.knowledge_acquisition import get_knowledge_acquisition
from modules.validation_engine import get_validation_engine
from modules.belief_revision import get_belief_revision_engine
from modules.reasoning_engine import get_reasoning_engine
from modules.output_interface import (
    SystemResponse,
    PipelineTrace,
    build_response,
)
from utils.database import init_db

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full Self-Evolving Knowledge System pipeline."""

    def __init__(self):
        # Ensure database is initialised
        init_db()

        self.epistemic = get_epistemic_analyzer()
        self.gap_detector = get_gap_detector()
        self.acquisition = get_knowledge_acquisition()
        self.validation = get_validation_engine()
        self.belief_revision = get_belief_revision_engine()
        self.reasoning = get_reasoning_engine()

    def run(self, raw_query: str, baseline_mode: bool = False) -> SystemResponse:
        """
        Execute the full pipeline for a user query.

        Args:
            raw_query: The raw user question.
            baseline_mode: If True, skip gap detection (for experiment comparison).

        Returns:
            SystemResponse with full traceability.
        """
        trace = PipelineTrace()
        start = time.time()
        gaps_addressed = []

        # ── Step 1: Input preprocessing ──────────────────────────────
        query = preprocess_query(raw_query)
        trace.add_step(
            "Input Interface",
            "completed",
            f"Query preprocessed. Key terms: {query.key_terms[:10]}",
        )

        # ── Step 2: Epistemic Assessment ─────────────────────────────
        t0 = time.time()
        epistemic_data = self.epistemic.assess(query)
        trace.add_step(
            "Epistemic State Analyzer",
            "completed",
            f"Composite confidence: {epistemic_data['composite_score']:.3f} "
            f"(ret={epistemic_data['retrieval_score']:.2f}, "
            f"cov={epistemic_data['coverage_score']:.2f}, "
            f"llm={epistemic_data['llm_score']:.2f})",
            data={
                "retrieval_score": epistemic_data["retrieval_score"],
                "coverage_score": epistemic_data["coverage_score"],
                "llm_score": epistemic_data["llm_score"],
                "composite_score": epistemic_data["composite_score"],
                "elapsed_s": round(time.time() - t0, 2),
            },
        )

        # ── Step 3: Gap Detection ────────────────────────────────────
        gap_result: Optional[GapDetectionResult] = None

        if baseline_mode:
            trace.add_step(
                "Gap Detector",
                "skipped",
                "Baseline mode — gap detection disabled.",
            )
        else:
            gap_result = self.gap_detector.detect(query, epistemic_data)
            trace.add_step(
                "Knowledge Gap Detection",
                "completed",
                f"Gap type: {gap_result.gap_type} — Requires learning: {gap_result.requires_learning}",
                data=gap_result.to_dict(),
            )

        # ── Step 4–6: Learning Pathway (if gap detected) ────────────
        if gap_result and gap_result.requires_learning:
            # Step 4: Autonomous Acquisition
            t0 = time.time()
            acquisitions = self.acquisition.acquire(gap_result)
            trace.add_step(
                "Autonomous Knowledge Acquisition",
                "completed",
                f"Retrieved {len(acquisitions)} results from external sources.",
                data={
                    "sources": list(set(a.source_name for a in acquisitions)),
                    "count": len(acquisitions),
                    "elapsed_s": round(time.time() - t0, 2),
                },
            )

            # Step 5: Validation
            t0 = time.time()
            validated = self.validation.validate_batch(acquisitions, query.normalised_text)
            accepted = [v for v in validated if v.accepted]
            trace.add_step(
                "Validation & Trust Assessment",
                "completed",
                f"Validated: {len(accepted)}/{len(validated)} accepted.",
                data={
                    "total": len(validated),
                    "accepted": len(accepted),
                    "rejected": len(validated) - len(accepted),
                    "elapsed_s": round(time.time() - t0, 2),
                },
            )

            # Step 6: Belief Revision
            t0 = time.time()
            revision_result = self.belief_revision.integrate(validated, gap_id=gap_result.gap_id)
            trace.add_step(
                "Belief Revision & Evolution",
                "completed",
                f"Added {revision_result.to_dict()['entries_added']}, "
                f"updated {revision_result.to_dict()['entries_updated']}, "
                f"conflicts {revision_result.to_dict()['conflicts_flagged']}.",
                data={
                    **revision_result.to_dict(),
                    "elapsed_s": round(time.time() - t0, 2),
                },
            )

            gaps_addressed.append({
                "gap_type": gap_result.gap_type,
                "description": gap_result.description,
                "entries_added": revision_result.to_dict()["entries_added"],
            })

        # ── Step 7: Reasoning ────────────────────────────────────────
        t0 = time.time()
        # Re-retrieve after potential knowledge base update
        reasoning_result = self.reasoning.reason(query.normalised_text)
        trace.add_step(
            "Reasoning & Inference Engine",
            "completed",
            f"Answer generated with confidence {reasoning_result.confidence:.2f}",
            data={
                "confidence": reasoning_result.confidence,
                "sources_count": len(reasoning_result.sources),
                "elapsed_s": round(time.time() - t0, 2),
            },
        )

        # ── Step 8: Build output ─────────────────────────────────────
        total_time = round(time.time() - start, 2)
        trace.add_step(
            "Output Interface",
            "completed",
            f"Response assembled in {total_time}s total.",
            data={"total_elapsed_s": total_time},
        )

        return build_response(
            answer=reasoning_result.answer,
            confidence=reasoning_result.confidence,
            sources=reasoning_result.sources,
            reasoning_chain=reasoning_result.reasoning_chain,
            gaps_addressed=gaps_addressed,
            trace=trace,
            limitations=reasoning_result.limitations,
        )


# ── Module-level quick access ────────────────────────────────────────
_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def run_query(raw_query: str, baseline_mode: bool = False) -> SystemResponse:
    """Convenience function to run a query through the pipeline."""
    return get_pipeline().run(raw_query, baseline_mode=baseline_mode)
