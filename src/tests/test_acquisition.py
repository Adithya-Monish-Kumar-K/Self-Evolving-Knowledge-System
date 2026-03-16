"""
Test 2: Autonomous Acquisition and Integration
Tests the knowledge acquisition module's ability to formulate queries
and retrieve from external sources, plus validation accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from modules.knowledge_acquisition import KnowledgeAcquisition, AcquisitionResult
from modules.validation_engine import ValidationEngine, ValidationResult
from modules.gap_detector import GapDetectionResult
import config


class TestKnowledgeAcquisition:
    """Test autonomous acquisition from external sources."""

    def _make_gap(self, query="quantum computing", gap_type=config.GAP_TYPE_MISSING):
        return GapDetectionResult(
            gap_type=gap_type,
            description="No relevant information found",
            query=query,
            epistemic_data={"retrieval_score": 0.05, "coverage_score": 0.02,
                            "llm_score": 0.1, "composite_score": 0.05},
        )

    def test_acquisition_result_dataclass(self):
        ar = AcquisitionResult(
            content="Test paper about quantum computing",
            title="Quantum Computing Survey",
            source_name="arxiv",
            source_url="https://arxiv.org/abs/1234.5678",
            source_id="1234.5678",
        )
        assert ar.source_name == "arxiv"
        assert ar.title == "Quantum Computing Survey"
        assert ar.citation_count == 0

    @patch.object(KnowledgeAcquisition, '_search_arxiv')
    @patch.object(KnowledgeAcquisition, '_search_wikipedia')
    @patch.object(KnowledgeAcquisition, '_search_semantic_scholar')
    @patch.object(KnowledgeAcquisition, '_formulate_queries')
    def test_acquire_calls_all_sources(self, mock_formulate, mock_ss, mock_wiki, mock_arxiv):
        mock_formulate.return_value = ["quantum computing advances"]
        mock_arxiv.return_value = [AcquisitionResult(content="arxiv paper", title="Paper 1", source_name="arxiv")]
        mock_wiki.return_value = [AcquisitionResult(content="wiki article", title="QC Wiki", source_name="wikipedia")]
        mock_ss.return_value = []

        acq = KnowledgeAcquisition()
        gap = self._make_gap()
        results = acq.acquire(gap)

        assert len(results) == 2
        mock_arxiv.assert_called_once()
        mock_wiki.assert_called_once()
        mock_ss.assert_called_once()


class TestValidationEngine:
    """Test the validation and trust assessment."""

    def _make_acquisition(self, source="arxiv", journal="", citations=0, year="2024"):
        return AcquisitionResult(
            content="Test content about machine learning",
            title="ML Paper",
            source_name=source,
            source_id="test123",
            journal_ref=journal,
            citation_count=citations,
            published_date=f"{year}-01-01",
            retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def test_source_reliability_arxiv_base(self):
        ve = ValidationEngine.__new__(ValidationEngine)
        ve.store = MagicMock()
        ve.llm = MagicMock()

        acq = self._make_acquisition(source="arxiv")
        score = ve._assess_source_reliability(acq)
        assert score >= config.SOURCE_TRUST_ARXIV

    def test_source_reliability_journal_bonus(self):
        ve = ValidationEngine.__new__(ValidationEngine)
        ve.store = MagicMock()
        ve.llm = MagicMock()

        acq = self._make_acquisition(source="arxiv", journal="Nature Physics")
        score = ve._assess_source_reliability(acq)
        assert score >= config.SOURCE_TRUST_ARXIV + 0.1

    def test_source_reliability_citation_bonus(self):
        ve = ValidationEngine.__new__(ValidationEngine)
        ve.store = MagicMock()
        ve.llm = MagicMock()

        acq = self._make_acquisition(source="semantic_scholar", citations=150)
        score = ve._assess_source_reliability(acq)
        assert score > config.SOURCE_TRUST_SEMANTIC_SCHOLAR

    def test_relevance_scoring_returns_bounded_value(self):
        ve = ValidationEngine.__new__(ValidationEngine)
        ve.store = MagicMock()
        ve.llm = MagicMock()

        acq = self._make_acquisition()
        # Mock embed_text
        with patch("modules.validation_engine.embed_text", return_value=[1.0] * 384):
            score = ve._assess_relevance(acq, "machine learning")
        assert 0.0 <= score <= 1.0

    def test_acceptance_rejected_low_relevance(self):
        ve = ValidationEngine.__new__(ValidationEngine)
        ve.store = MagicMock()
        ve.llm = MagicMock()

        acq = self._make_acquisition()

        with patch.object(ve, '_assess_source_reliability', return_value=0.8):
            with patch.object(ve, '_assess_relevance', return_value=0.1):  # below threshold
                with patch.object(ve, '_assess_consistency', return_value=0.7):
                    result = ve.validate(acq, "something totally unrelated")

        assert not result.accepted
        assert "Relevance" in result.rejection_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
