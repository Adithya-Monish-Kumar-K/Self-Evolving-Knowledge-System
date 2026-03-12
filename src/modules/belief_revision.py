"""
Belief Revision & Knowledge Evolution Engine – Manages version-controlled
updates to the knowledge base, conflict resolution, and knowledge graph
evolution. Ensures no catastrophic forgetting.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import config
from modules.knowledge_store import get_knowledge_store
from modules.validation_engine import ValidationResult
from utils.database import (
    insert_knowledge_entry,
    get_active_entries,
    get_knowledge_entry,
    update_knowledge_entry,
    insert_revision,
    insert_rule,
    resolve_gap,
)
from utils.knowledge_graph import get_knowledge_graph
from utils.llm import get_llm

logger = logging.getLogger(__name__)


class BeliefRevisionResult:
    """Summary of what was integrated / revised."""

    def __init__(self):
        self.entries_added: List[Dict] = []
        self.entries_updated: List[Dict] = []
        self.entries_superseded: List[Dict] = []
        self.conflicts_flagged: List[Dict] = []
        self.rules_added: List[Dict] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_added": len(self.entries_added),
            "entries_updated": len(self.entries_updated),
            "entries_superseded": len(self.entries_superseded),
            "conflicts_flagged": len(self.conflicts_flagged),
            "rules_added": len(self.rules_added),
            "details": {
                "added": self.entries_added,
                "updated": self.entries_updated,
                "superseded": self.entries_superseded,
                "conflicts": self.conflicts_flagged,
                "rules": self.rules_added,
            },
        }


class BeliefRevisionEngine:
    """Integrates validated knowledge and resolves conflicts."""

    def __init__(self):
        self.store = get_knowledge_store()
        self.kg = get_knowledge_graph()
        self.llm = get_llm()

    def integrate(
        self,
        validated: List[ValidationResult],
        gap_id: Optional[int] = None,
    ) -> BeliefRevisionResult:
        """
        Integrate validated acquisition results into the knowledge base.
        Handles conflict detection, version control, and graph updates.
        """
        result = BeliefRevisionResult()

        accepted = [v for v in validated if v.accepted]
        if not accepted:
            logger.info("No accepted results to integrate.")
            return result

        for vr in accepted:
            acq = vr.acquisition
            try:
                self._integrate_single(vr, result)
            except Exception as e:
                logger.error("Failed to integrate '%s': %s", acq.title[:50], e)

        # Save the knowledge graph
        try:
            self.kg.save()
        except Exception as e:
            logger.warning("Could not save knowledge graph: %s", e)

        # Mark gap as resolved
        if gap_id is not None:
            try:
                added_ids = [e["entry_id"] for e in result.entries_added if "entry_id" in e]
                updated_ids = [e.get("entry_id") for e in result.entries_updated if e.get("entry_id")]
                all_ids = added_ids + updated_ids
                resolve_gap(
                    gap_id,
                    resolution_info=f"Integrated {len(result.entries_added)} entries, "
                    f"updated {len(result.entries_updated)}, "
                    f"flagged {len(result.conflicts_flagged)} conflicts.",
                    entry_ids=all_ids,
                )
            except Exception as e:
                logger.warning("Could not resolve gap: %s", e)

        logger.info(
            "Belief revision complete: added=%d, updated=%d, superseded=%d, conflicts=%d",
            len(result.entries_added),
            len(result.entries_updated),
            len(result.entries_superseded),
            len(result.conflicts_flagged),
        )
        return result

    def _integrate_single(self, vr: ValidationResult, result: BeliefRevisionResult):
        """Integrate a single validated result."""
        acq = vr.acquisition

        # Check for existing entries with the same source_id
        existing = self._find_existing(acq)

        if existing:
            self._handle_existing(existing, vr, result)
        else:
            self._add_new(vr, result)

    def _find_existing(self, acq) -> Optional[Dict]:
        """Find existing knowledge entry matching this acquisition."""
        if acq.source_id:
            # Check ChromaDB first
            doc_id = f"{acq.source_name}:{acq.source_id}"
            existing = self.store.get_document(doc_id)
            if existing:
                return existing

        # Semantic search for very similar content
        similar = self.store.search(acq.content[:500], top_k=1)
        if similar and similar[0].get("similarity", 0) > 0.9:
            return similar[0]

        return None

    def _add_new(self, vr: ValidationResult, result: BeliefRevisionResult):
        """Add a brand new knowledge entry."""
        acq = vr.acquisition
        doc_id = f"{acq.source_name}:{acq.source_id or acq.title[:50]}"

        # SQLite entry
        confidence = vr.trust_score if not vr.low_confidence else vr.trust_score * 0.7
        metadata = {
            "source_url": acq.source_url,
            "authors": acq.authors,
            "categories": acq.categories,
            "published_date": acq.published_date,
            "journal_ref": acq.journal_ref,
            "citation_count": acq.citation_count,
        }

        entry_id = insert_knowledge_entry(
            source_type=acq.source_name,
            content=acq.content[:5000],
            title=acq.title,
            source_id=acq.source_id,
            confidence=confidence,
            trust=vr.trust_score,
            metadata_json=json.dumps(metadata),
        )

        # ChromaDB document
        self.store.add_document(
            doc_id=doc_id,
            content=acq.content[:2000],
            metadata={
                "entry_id": str(entry_id),
                "title": acq.title[:200],
                "source_type": acq.source_name,
                "categories": acq.categories,
                "trust_score": str(round(vr.trust_score, 3)),
                "update_date": acq.published_date[:10] if acq.published_date else "",
            },
        )

        # Knowledge Graph
        paper_data = {
            "id": acq.source_id or doc_id,
            "title": acq.title,
            "categories": acq.categories,
            "update_date": acq.published_date,
            "authors_parsed": [[a.strip()] for a in acq.authors.split(",") if a.strip()],
        }
        self.kg.add_paper(paper_data)

        # Try to extract logical rules
        self._extract_rules(acq, entry_id, result)

        result.entries_added.append({
            "entry_id": entry_id,
            "doc_id": doc_id,
            "title": acq.title,
            "trust_score": vr.trust_score,
        })

    def _handle_existing(self, existing: Dict, vr: ValidationResult, result: BeliefRevisionResult):
        """Handle the case where similar/same content already exists."""
        acq = vr.acquisition
        existing_id = existing.get("id", "")

        # Compare trust scores to decide action
        existing_meta = existing.get("metadata", {})
        existing_trust = float(existing_meta.get("trust_score", 0.5))
        new_trust = vr.trust_score

        trust_diff = abs(new_trust - existing_trust)

        if trust_diff < config.CONFLICT_TRUST_MARGIN:
            # Trust scores too close — flag as conflict, keep both
            new_doc_id = f"{acq.source_name}:{acq.source_id or acq.title[:50]}_new"
            entry_id = insert_knowledge_entry(
                source_type=acq.source_name,
                content=acq.content[:5000],
                title=acq.title,
                source_id=acq.source_id,
                confidence=new_trust * 0.8,
                trust=new_trust,
                metadata_json=json.dumps({"conflict_with": existing_id}),
            )
            update_knowledge_entry(entry_id, conflict_flag=1)

            self.store.add_document(
                doc_id=new_doc_id,
                content=acq.content[:2000],
                metadata={
                    "entry_id": str(entry_id),
                    "title": acq.title[:200],
                    "source_type": acq.source_name,
                    "trust_score": str(round(new_trust, 3)),
                    "conflict": "true",
                },
            )
            result.conflicts_flagged.append({
                "existing_id": existing_id,
                "new_entry_id": entry_id,
                "title": acq.title,
            })
        elif new_trust > existing_trust:
            # New info is more trustworthy — supersede old
            old_content = existing.get("content", "")

            # Update ChromaDB doc
            self.store.update_document(
                doc_id=existing_id,
                content=acq.content[:2000],
                metadata={
                    "title": acq.title[:200],
                    "source_type": acq.source_name,
                    "trust_score": str(round(new_trust, 3)),
                    "update_date": acq.published_date[:10] if acq.published_date else "",
                },
            )

            # Log revision in SQLite
            existing_entry_id = int(existing_meta.get("entry_id", 0)) if existing_meta.get("entry_id") else None
            if existing_entry_id:
                insert_revision(
                    entry_id=existing_entry_id,
                    old_content=old_content[:2000],
                    new_content=acq.content[:2000],
                    reason=f"Superseded by higher-trust source ({acq.source_name}, trust={new_trust:.2f} vs {existing_trust:.2f})",
                )
                update_knowledge_entry(
                    existing_entry_id,
                    new_content=acq.content[:5000],
                    new_trust=new_trust,
                )

            result.entries_updated.append({
                "existing_id": existing_id,
                "title": acq.title,
                "old_trust": existing_trust,
                "new_trust": new_trust,
            })
        else:
            # Existing info is more trustworthy — skip (but log)
            logger.info(
                "Existing knowledge has higher trust (%.2f > %.2f), skipping: %s",
                existing_trust, new_trust, acq.title[:50],
            )

    def _extract_rules(self, acq, entry_id: int, result: BeliefRevisionResult):
        """Try to extract logical rules from the acquired content using LLM."""
        prompt = f"""Analyse the following academic content and extract 0-3 logical rules (if-then statements) that represent key findings or relationships.

CONTENT:
{acq.content[:1500]}

Respond with ONLY a JSON object:
{{
  "rules": [
    {{"antecedent": "IF condition/premise", "consequent": "THEN conclusion/result", "confidence": 0.8}}
  ]
}}
If no clear rules can be extracted, return: {{"rules": []}}"""

        try:
            result_json = self.llm.generate_json(prompt)
            rules = result_json.get("rules", [])
            for rule in rules[:3]:
                ant = rule.get("antecedent", "")
                con = rule.get("consequent", "")
                conf = float(rule.get("confidence", 0.5))
                if ant and con:
                    rule_id = insert_rule(ant, con, conf, entry_id)
                    result.rules_added.append({
                        "rule_id": rule_id,
                        "antecedent": ant,
                        "consequent": con,
                    })
        except Exception as e:
            logger.debug("Rule extraction failed: %s", e)


# ── Singleton ────────────────────────────────────────────────────────
_engine = None


def get_belief_revision_engine() -> BeliefRevisionEngine:
    global _engine
    if _engine is None:
        _engine = BeliefRevisionEngine()
    return _engine
