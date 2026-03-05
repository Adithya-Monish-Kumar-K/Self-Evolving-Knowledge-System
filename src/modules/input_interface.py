"""
Input Interface – normalises and preprocesses user queries
before they enter the pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Normalised query object that flows through the pipeline."""
    raw_text: str
    normalised_text: str = ""
    key_terms: List[str] = field(default_factory=list)
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)


def preprocess_query(raw: str) -> Query:
    """Clean and extract key information from a user query."""
    normalised = raw.strip()
    normalised = re.sub(r"\s+", " ", normalised)

    # Simple keyword extraction: remove stop-words, keep significant tokens
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "about",
        "between", "under", "above", "and", "but", "or", "nor", "not",
        "so", "yet", "both", "either", "neither", "each", "every",
        "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just",
        "because", "if", "when", "where", "how", "what", "which",
        "who", "whom", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "him", "his", "she",
        "her", "it", "its", "they", "them", "their", "tell", "explain",
        "describe", "show", "find", "give", "get", "know", "think",
    }
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]+\b", normalised.lower())
    key_terms = [t for t in tokens if t not in stop_words and len(t) > 2]

    return Query(
        raw_text=raw,
        normalised_text=normalised,
        key_terms=key_terms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
