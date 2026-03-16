"""
Test 1: Baseline vs. Epistemic Awareness
Tests that the Epistemic State Analyzer correctly identifies knowledge gaps
rather than hallucinating answers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from modules.input_interface import preprocess_query
from modules.gap_detector import GapDetector
import config


class TestEpistemicAwareness:
    """Test the epistemic analyzer's ability to detect unknowns."""

    def test_preprocess_query_extracts_key_terms(self):
        q = preprocess_query("What are the latest advances in quantum computing?")
        assert "quantum" in q.key_terms
        assert "computing" in q.key_terms
        assert "latest" in q.key_terms
        assert "advances" in q.key_terms
        # Stop words should be removed
        assert "what" not in q.key_terms
        assert "the" not in q.key_terms
        assert "are" not in q.key_terms

    def test_preprocess_query_handles_empty(self):
        q = preprocess_query("")
        assert q.normalised_text == ""
        assert q.key_terms == []

    def test_preprocess_query_normalises_whitespace(self):
        q = preprocess_query("  too   many    spaces  ")
        assert q.normalised_text == "too many spaces"

    def test_gap_detector_no_gap_when_high_confidence(self):
        detector = GapDetector()
        query = preprocess_query("test query")
        epistemic_data = {
            "retrieval_score": 0.8,
            "coverage_score": 0.7,
            "llm_score": 0.9,
            "composite_score": 0.80,
            "gap_detected": False,
            "retrieved_docs": [],
            "llm_analysis": {},
        }
        result = detector.detect(query, epistemic_data)
        assert result.gap_type == config.GAP_TYPE_NONE
        assert not result.requires_learning

    def test_gap_detector_missing_topic(self):
        detector = GapDetector()
        query = preprocess_query("quantum teleportation through wormholes")
        epistemic_data = {
            "retrieval_score": 0.05,
            "coverage_score": 0.02,
            "llm_score": 0.1,
            "composite_score": 0.05,
            "gap_detected": True,
            "retrieved_docs": [],
            "llm_analysis": {"contradictions": "none"},
        }
        result = detector.detect(query, epistemic_data)
        assert result.gap_type == config.GAP_TYPE_MISSING
        assert result.requires_learning

    def test_gap_detector_shallow_coverage(self):
        detector = GapDetector()
        query = preprocess_query("details about higgs boson decay channels")
        epistemic_data = {
            "retrieval_score": 0.4,
            "coverage_score": 0.3,
            "llm_score": 0.3,
            "composite_score": 0.33,
            "gap_detected": True,
            "retrieved_docs": [{"metadata": {"update_date": "2026-02-01"}}],
            "llm_analysis": {"contradictions": "none", "missing": "specific decay channel branching ratios"},
        }
        result = detector.detect(query, epistemic_data)
        assert result.gap_type == config.GAP_TYPE_SHALLOW
        assert result.requires_learning

    def test_gap_detector_contradictory(self):
        detector = GapDetector()
        query = preprocess_query("speed of light in vacuum")
        epistemic_data = {
            "retrieval_score": 0.5,
            "coverage_score": 0.5,
            "llm_score": 0.4,
            "composite_score": 0.45,
            "gap_detected": True,
            "retrieved_docs": [],
            "llm_analysis": {"contradictions": "Two sources disagree on the measured value"},
        }
        result = detector.detect(query, epistemic_data)
        assert result.gap_type == config.GAP_TYPE_CONTRADICTORY
        assert result.requires_learning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
