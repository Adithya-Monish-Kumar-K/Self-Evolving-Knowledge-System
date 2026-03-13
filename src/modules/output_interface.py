"""
Output & Explanation Interface – Structures the final response
with full pipeline traceability.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    """Records which modules were activated and their outputs."""
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def add_step(self, module: str, status: str, summary: str, data: Optional[Dict] = None):
        self.steps.append({
            "module": module,
            "status": status,
            "summary": summary,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


@dataclass
class SystemResponse:
    """Complete structured response from the system."""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning_chain: List[str]
    knowledge_gaps_addressed: List[Dict[str, Any]]
    pipeline_trace: PipelineTrace
    limitations: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "reasoning_chain": self.reasoning_chain,
            "knowledge_gaps_addressed": self.knowledge_gaps_addressed,
            "pipeline_trace": [s for s in self.pipeline_trace.steps],
            "limitations": self.limitations,
            "timestamp": self.timestamp,
        }


def build_response(
    answer: str,
    confidence: float,
    sources: List[Dict],
    reasoning_chain: List[str],
    gaps_addressed: List[Dict],
    trace: PipelineTrace,
    limitations: str = "",
) -> SystemResponse:
    """Convenience builder for SystemResponse."""
    return SystemResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        reasoning_chain=reasoning_chain,
        knowledge_gaps_addressed=gaps_addressed,
        pipeline_trace=trace,
        limitations=limitations,
    )
