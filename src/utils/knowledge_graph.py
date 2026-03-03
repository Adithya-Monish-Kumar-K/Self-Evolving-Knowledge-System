"""
Knowledge Graph utility – builds and queries a NetworkX directed graph
of papers, authors, and categories.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

import config

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Manages a directed knowledge graph persisted as GraphML."""

    def __init__(self, graph_path: str = config.GRAPH_PATH):
        self.graph_path = graph_path
        if os.path.exists(graph_path):
            logger.info("Loading existing knowledge graph from %s", graph_path)
            self.G = nx.read_graphml(graph_path)
        else:
            logger.info("Creating new knowledge graph")
            self.G = nx.DiGraph()

    # ── Persistence ──────────────────────────────────────────────────
    def save(self):
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        nx.write_graphml(self.G, self.graph_path)
        logger.info("Knowledge graph saved (%d nodes, %d edges)", self.G.number_of_nodes(), self.G.number_of_edges())

    # ── Add nodes ────────────────────────────────────────────────────
    def add_paper(self, paper: Dict[str, Any]):
        """Add a paper node and connect it to its authors and categories."""
        paper_id = f"paper:{paper.get('id', '')}"
        self.G.add_node(
            paper_id,
            node_type="paper",
            title=paper.get("title", "")[:200],
            update_date=paper.get("update_date", ""),
        )

        # Authors
        authors_raw = paper.get("authors_parsed", [])
        if isinstance(authors_raw, list):
            for author_parts in authors_raw:
                if isinstance(author_parts, list) and len(author_parts) >= 2:
                    name = f"{author_parts[1].strip()} {author_parts[0].strip()}".strip()
                elif isinstance(author_parts, str):
                    name = author_parts.strip()
                else:
                    continue
                if not name:
                    continue
                author_id = f"author:{name}"
                self.G.add_node(author_id, node_type="author", name=name)
                self.G.add_edge(paper_id, author_id, relation="authored_by")
                self.G.add_edge(author_id, paper_id, relation="wrote")

        # Categories
        categories = paper.get("categories", "")
        if isinstance(categories, str):
            for cat in categories.split():
                cat = cat.strip()
                if not cat:
                    continue
                cat_id = f"category:{cat}"
                self.G.add_node(cat_id, node_type="category", name=cat)
                self.G.add_edge(paper_id, cat_id, relation="belongs_to")
                self.G.add_edge(cat_id, paper_id, relation="contains")

    def add_concept(self, concept: str, related_to: Optional[str] = None, relation: str = "related"):
        """Add a free-form concept node, optionally linked to another node."""
        cid = f"concept:{concept}"
        self.G.add_node(cid, node_type="concept", name=concept)
        if related_to and related_to in self.G:
            self.G.add_edge(cid, related_to, relation=relation)
            self.G.add_edge(related_to, cid, relation=relation)

    # ── Query ────────────────────────────────────────────────────────
    def get_neighbors(self, node_id: str, depth: int = 1) -> List[Dict]:
        """Return nodes within *depth* hops of the given node."""
        if node_id not in self.G:
            return []
        visited: Set[str] = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for n in frontier:
                for neighbor in self.G.neighbors(n):
                    if neighbor not in visited and neighbor != node_id:
                        next_frontier.add(neighbor)
            visited |= next_frontier
            frontier = next_frontier
        return [
            {"id": n, **dict(self.G.nodes[n])}
            for n in visited
        ]

    def find_related_concepts(self, query_terms: List[str], top_k: int = 10) -> List[Dict]:
        """Find nodes whose name/title contains any of the query terms."""
        results = []
        query_lower = [t.lower() for t in query_terms]
        for node_id, data in self.G.nodes(data=True):
            text = (data.get("title", "") + " " + data.get("name", "")).lower()
            score = sum(1 for t in query_lower if t in text)
            if score > 0:
                results.append({"id": node_id, "score": score, **data})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_category_density(self, category: str) -> float:
        """Return a 0-1 density score for a category based on edge count."""
        cat_id = f"category:{category}"
        if cat_id not in self.G:
            # Try partial match
            for nid in self.G.nodes:
                if nid.startswith("category:") and category.lower() in nid.lower():
                    cat_id = nid
                    break
            else:
                return 0.0
        degree = self.G.degree(cat_id)
        # Normalise: cap at 200 edges → score 1.0
        return min(degree / 200.0, 1.0)

    def get_subgraph(self, center_node: str, depth: int = 1) -> nx.DiGraph:
        """Return a shallow subgraph around a node."""
        if center_node not in self.G:
            return nx.DiGraph()
        nodes = {center_node}
        frontier = {center_node}
        for _ in range(depth):
            next_f: Set[str] = set()
            for n in frontier:
                next_f |= set(self.G.neighbors(n))
            nodes |= next_f
            frontier = next_f
        return self.G.subgraph(nodes).copy()

    # ── Stats ────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, int]:
        type_counts: Dict[str, int] = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("node_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            **type_counts,
        }


# ── Singleton ────────────────────────────────────────────────────────
_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg
