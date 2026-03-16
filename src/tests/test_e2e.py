"""
Test 4: End-to-End Pipeline & Latency
Tests the full pipeline integration and measures time-to-resolution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
from unittest.mock import patch, MagicMock

from modules.input_interface import preprocess_query, Query
from modules.output_interface import PipelineTrace, SystemResponse, build_response


class TestEndToEnd:
    """Test pipeline integration and output structure."""

    def test_pipeline_trace_records_steps(self):
        trace = PipelineTrace()
        trace.add_step("Input Interface", "completed", "Query preprocessed")
        trace.add_step("Epistemic Analyzer", "completed", "Confidence: 0.85")

        assert len(trace.steps) == 2
        assert trace.steps[0]["module"] == "Input Interface"
        assert trace.steps[1]["status"] == "completed"

    def test_system_response_structure(self):
        trace = PipelineTrace()
        trace.add_step("Test", "completed", "Test step")

        response = build_response(
            answer="Test answer",
            confidence=0.8,
            sources=[{"index": 1, "title": "Source 1"}],
            reasoning_chain=["Step 1", "Step 2"],
            gaps_addressed=[],
            trace=trace,
            limitations="None",
        )

        assert isinstance(response, SystemResponse)
        assert response.answer == "Test answer"
        assert response.confidence == 0.8
        assert len(response.sources) == 1
        assert len(response.reasoning_chain) == 2
        assert response.timestamp  # should be auto-set

    def test_response_to_dict(self):
        trace = PipelineTrace()
        response = build_response(
            answer="Answer",
            confidence=0.5,
            sources=[],
            reasoning_chain=[],
            gaps_addressed=[],
            trace=trace,
        )
        d = response.to_dict()
        assert "answer" in d
        assert "confidence" in d
        assert "pipeline_trace" in d
        assert "timestamp" in d

    def test_input_preprocessing_speed(self):
        """Input preprocessing should be near-instant (< 50ms)."""
        start = time.time()
        for _ in range(100):
            preprocess_query("What is the Higgs boson mass measurement from ATLAS experiment?")
        elapsed = time.time() - start
        # 100 queries in < 5 seconds
        assert elapsed < 5.0, f"Input preprocessing too slow: {elapsed:.2f}s for 100 queries"


class TestLatencyMetrics:
    """Placeholder tests for measuring TTR in real environment.

    These tests are designed to be run manually when Ollama is available.
    They print timing information rather than asserting strict bounds.
    """

    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
    )
    def test_full_pipeline_latency(self):
        """Measure full pipeline time-to-resolution."""
        from pipeline import run_query

        queries = [
            "What is the Higgs boson?",
            "Explain quantum entanglement",
            "What are recent advances in large language models?",
        ]

        for q in queries:
            start = time.time()
            response = run_query(q)
            elapsed = time.time() - start
            print(f"\nQuery: {q}")
            print(f"  TTR: {elapsed:.2f}s")
            print(f"  Confidence: {response.confidence:.2f}")
            print(f"  Gap detected: {bool(response.knowledge_gaps_addressed)}")
            print(f"  Sources: {len(response.sources)}")
            for step in response.pipeline_trace.steps:
                data = step.get("data", {})
                step_time = data.get("elapsed_s", "?")
                print(f"  [{step['module']}] {step_time}s — {step['summary']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
