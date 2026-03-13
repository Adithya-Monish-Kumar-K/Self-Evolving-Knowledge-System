"""
Reasoning & Inference Engine – Assembles context from the knowledge base,
knowledge graph, and logical rules, then generates a reasoned answer
with citations via the LLM.
"""

import logging
from typing import Any, Dict, List, Optional

import config
from modules.knowledge_store import get_knowledge_store
from utils.knowledge_graph import get_knowledge_graph
from utils.database import get_rules
from utils.llm import get_llm

logger = logging.getLogger(__name__)

_REASONING_SYSTEM_PROMPT = """You are a Self-Evolving Knowledge System. Your role is to provide accurate, well-reasoned answers based EXCLUSIVELY on the provided context.

RULES:
1. Base your answer ONLY on the provided context (documents, graph connections, and logical rules).
2. Cite your sources using [Source N] notation corresponding to the numbered context passages.
3. If the context is insufficient, clearly state what you cannot answer and why.
4. Provide step-by-step reasoning before your final answer.
5. Rate your confidence (0.0-1.0) in the answer based on the quality and completeness of the context."""

_REASONING_PROMPT = """USER QUERY:
{query}

─── RETRIEVED DOCUMENTS ───
{documents}

─── KNOWLEDGE GRAPH CONNECTIONS ───
{graph_info}

─── LOGICAL RULES ───
{rules}

─── TASK ───
Using the above context, answer the user's query step by step.

Respond with ONLY a JSON object:
{{
  "reasoning_steps": [
    "Step 1: <reasoning>",
    "Step 2: <reasoning>"
  ],
  "answer": "<your comprehensive answer>",
  "confidence": <float 0.0-1.0>,
  "sources_used": [1, 2, 3],
  "limitations": "<any limitations or caveats>"
}}"""


class ReasoningResult:
    """Structured output from the reasoning engine."""

    def __init__(
        self,
        answer: str,
        confidence: float,
        reasoning_chain: List[str],
        sources: List[Dict],
        limitations: str = "",
    ):
        self.answer = answer
        self.confidence = confidence
        self.reasoning_chain = reasoning_chain
        self.sources = sources
        self.limitations = limitations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain,
            "sources": self.sources,
            "limitations": self.limitations,
        }


class ReasoningEngine:
    """Assembles context and generates reasoned answers."""

    def __init__(self):
        self.store = get_knowledge_store()
        self.kg = get_knowledge_graph()
        self.llm = get_llm()

    def reason(self, query_text: str, retrieved_docs: List[Dict] = None) -> ReasoningResult:
        """
        Assemble context and generate a reasoned answer.

        If retrieved_docs is None, performs its own retrieval.
        """
        # 1. Retrieve documents if not provided
        if retrieved_docs is None:
            retrieved_docs = self.store.search(query_text, top_k=config.CHROMA_SEARCH_TOP_K)

        # 2. Format document context
        doc_context = self._format_documents(retrieved_docs)

        # 3. Get knowledge graph context
        graph_context = self._format_graph_info(query_text)

        # 4. Get relevant logical rules
        rules_context = self._format_rules(query_text)

        # 5. Build prompt and send to LLM
        prompt = _REASONING_PROMPT.format(
            query=query_text,
            documents=doc_context,
            graph_info=graph_context,
            rules=rules_context,
        )

        try:
            result = self.llm.generate_json(prompt, system_prompt=_REASONING_SYSTEM_PROMPT)

            reasoning_steps = result.get("reasoning_steps", [])
            answer = result.get("answer", "I could not generate an answer from the available context.")
            confidence = float(result.get("confidence", 0.5))
            sources_used = result.get("sources_used", [])
            limitations = result.get("limitations", "")

            # Map source indices to actual documents
            sources = []
            for idx in sources_used:
                if isinstance(idx, int) and 1 <= idx <= len(retrieved_docs):
                    doc = retrieved_docs[idx - 1]
                    sources.append({
                        "index": idx,
                        "title": doc.get("metadata", {}).get("title", "Unknown"),
                        "source_type": doc.get("metadata", {}).get("source_type", ""),
                        "trust_score": doc.get("metadata", {}).get("trust_score", "N/A"),
                        "similarity": round(doc.get("similarity", 0), 3),
                    })

            return ReasoningResult(
                answer=answer,
                confidence=confidence,
                reasoning_chain=reasoning_steps,
                sources=sources,
                limitations=limitations,
            )

        except Exception as e:
            logger.error("Reasoning failed: %s", e)
            return ReasoningResult(
                answer=f"Reasoning failed due to an error: {str(e)}",
                confidence=0.0,
                reasoning_chain=["Error occurred during reasoning."],
                sources=[],
                limitations="Complete reasoning failure.",
            )

    # ── Context Formatting ───────────────────────────────────────────
    def _format_documents(self, docs: List[Dict]) -> str:
        if not docs:
            return "(No documents retrieved.)"

        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            title = meta.get("title", "Untitled")
            source_type = meta.get("source_type", "unknown")
            trust = meta.get("trust_score", "N/A")
            sim = doc.get("similarity", 0)
            content = doc.get("content", "")[:800]

            parts.append(
                f"[Source {i}] ({source_type}, trust={trust}, similarity={sim:.2f})\n"
                f"Title: {title}\n"
                f"Content: {content}\n"
            )
        return "\n---\n".join(parts)

    def _format_graph_info(self, query_text: str) -> str:
        """Get relevant graph neighbors for context."""
        terms = query_text.lower().split()
        # Remove common words
        terms = [t for t in terms if len(t) > 3]

        related = self.kg.find_related_concepts(terms, top_k=5)
        if not related:
            return "(No relevant graph connections found.)"

        parts = []
        for node in related:
            node_type = node.get("node_type", "")
            name = node.get("name", node.get("title", node.get("id", "")))
            parts.append(f"- [{node_type}] {name}")

            # Get immediate neighbors
            neighbors = self.kg.get_neighbors(node["id"], depth=1)
            for nb in neighbors[:3]:
                nb_type = nb.get("node_type", "")
                nb_name = nb.get("name", nb.get("title", nb.get("id", "")))
                parts.append(f"  → [{nb_type}] {nb_name}")

        return "\n".join(parts[:20])  # cap output

    def _format_rules(self, query_text: str) -> str:
        """Get logical rules relevant to the query."""
        # Try to find rules matching query terms
        terms = [t for t in query_text.lower().split() if len(t) > 3]
        all_rules = []
        for term in terms[:3]:
            rules = get_rules(category=term, limit=5)
            all_rules.extend(rules)

        # Deduplicate by id
        seen = set()
        unique_rules = []
        for r in all_rules:
            rid = r.get("id")
            if rid not in seen:
                seen.add(rid)
                unique_rules.append(r)

        if not unique_rules:
            return "(No logical rules found for this domain.)"

        parts = []
        for r in unique_rules[:5]:
            parts.append(
                f"- {r.get('antecedent', '?')} → {r.get('consequent', '?')} "
                f"(confidence: {r.get('confidence', 0):.2f})"
            )
        return "\n".join(parts)


# ── Singleton ────────────────────────────────────────────────────────
_engine = None


def get_reasoning_engine() -> ReasoningEngine:
    global _engine
    if _engine is None:
        _engine = ReasoningEngine()
    return _engine
