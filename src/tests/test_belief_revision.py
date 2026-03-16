"""
Test 3: Belief Revision and Conflict Resolution
Tests the system's ability to update knowledge, handle conflicts,
and prevent catastrophic forgetting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from utils.database import (
    init_db, insert_knowledge_entry, get_knowledge_entry,
    update_knowledge_entry, insert_revision, get_revisions,
)
import config


class TestBeliefRevision:
    """Test knowledge entry updates, versioning, and conflict handling."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        """Use a temporary database for each test."""
        self.db_path = str(tmp_path / "test.db")
        init_db(self.db_path)

    def test_insert_and_retrieve_entry(self):
        entry_id = insert_knowledge_entry(
            source_type="arxiv",
            content="Quantum computing enables exponential speedup",
            title="QC Paper",
            source_id="qc001",
            confidence=0.8,
            trust=0.85,
            db_path=self.db_path,
        )
        assert entry_id is not None

        entry = get_knowledge_entry(entry_id, db_path=self.db_path)
        assert entry is not None
        assert entry["source_type"] == "arxiv"
        assert entry["version"] == 1
        assert entry["is_active"] == 1

    def test_version_increment_on_update(self):
        entry_id = insert_knowledge_entry(
            source_type="arxiv",
            content="Original content",
            title="Test",
            db_path=self.db_path,
        )

        update_knowledge_entry(
            entry_id,
            new_content="Updated content",
            new_trust=0.9,
            db_path=self.db_path,
        )

        entry = get_knowledge_entry(entry_id, db_path=self.db_path)
        assert entry["version"] == 2
        assert entry["content"] == "Updated content"
        assert entry["trust_score"] == 0.9

    def test_supersede_entry(self):
        old_id = insert_knowledge_entry(
            source_type="arxiv",
            content="Old information",
            title="Old",
            trust=0.5,
            db_path=self.db_path,
        )
        new_id = insert_knowledge_entry(
            source_type="arxiv",
            content="New information",
            title="New",
            trust=0.9,
            db_path=self.db_path,
        )

        # Supersede old with new
        update_knowledge_entry(
            old_id,
            is_active=0,
            superseded_by=new_id,
            db_path=self.db_path,
        )

        old_entry = get_knowledge_entry(old_id, db_path=self.db_path)
        assert old_entry["is_active"] == 0
        assert old_entry["superseded_by"] == new_id

        # New entry still active
        new_entry = get_knowledge_entry(new_id, db_path=self.db_path)
        assert new_entry["is_active"] == 1

    def test_revision_history_preserved(self):
        entry_id = insert_knowledge_entry(
            source_type="wikipedia",
            content="Speed of light is 300000 km/s",
            title="Speed of Light",
            db_path=self.db_path,
        )

        insert_revision(
            entry_id=entry_id,
            old_content="Speed of light is 300000 km/s",
            new_content="Speed of light is 299792.458 km/s",
            reason="More precise value from peer-reviewed source",
            db_path=self.db_path,
        )

        revisions = get_revisions(entry_id=entry_id, db_path=self.db_path)
        assert len(revisions) == 1
        assert "299792.458" in revisions[0]["new_content"]
        assert "More precise" in revisions[0]["reason"]

    def test_conflict_flag(self):
        entry_id = insert_knowledge_entry(
            source_type="arxiv",
            content="Conflicting claim",
            title="Conflict",
            db_path=self.db_path,
        )

        update_knowledge_entry(
            entry_id,
            conflict_flag=1,
            db_path=self.db_path,
        )

        entry = get_knowledge_entry(entry_id, db_path=self.db_path)
        assert entry["conflict_flag"] == 1

    def test_retention_after_supersession(self):
        """Old entries should still be retrievable (no catastrophic forgetting)."""
        ids = []
        for i in range(3):
            eid = insert_knowledge_entry(
                source_type="arxiv",
                content=f"Content version {i}",
                title=f"Entry {i}",
                db_path=self.db_path,
            )
            ids.append(eid)

        # Supersede first two
        for i in range(2):
            update_knowledge_entry(ids[i], is_active=0, superseded_by=ids[2], db_path=self.db_path)

        # All entries should still exist
        for eid in ids:
            entry = get_knowledge_entry(eid, db_path=self.db_path)
            assert entry is not None  # not deleted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
