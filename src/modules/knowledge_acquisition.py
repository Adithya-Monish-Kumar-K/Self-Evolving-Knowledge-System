"""
Autonomous Knowledge Acquisition Module – Formulates learning objectives,
queries multiple external sources (arXiv, Wikipedia, Semantic Scholar),
and returns normalised acquisition results.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import arxiv
import requests

import config
from modules.gap_detector import GapDetectionResult
from utils.llm import get_llm

logger = logging.getLogger(__name__)

# Shared session with proper User-Agent (required by Wikipedia/MediaWiki API)
_session = requests.Session()
_session.headers.update({
    "User-Agent": "SelfEvolvingKnowledgeSystem/1.0 (Academic Project; Python/requests)",
    "Accept": "application/json",
})


@dataclass
class AcquisitionResult:
    """Normalised result from any external source."""
    content: str
    title: str
    source_name: str              # 'arxiv', 'wikipedia', 'semantic_scholar'
    source_url: str = ""
    source_id: str = ""
    authors: str = ""
    categories: str = ""
    published_date: str = ""
    journal_ref: str = ""
    citation_count: int = 0
    retrieval_timestamp: str = ""
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeAcquisition:
    """Formulates search queries and retrieves from multiple sources."""

    def __init__(self):
        self.llm = get_llm()
        self._last_arxiv_call = 0.0
        self._semantic_scholar_calls = []

    # ── Main entry point ─────────────────────────────────────────────
    def acquire(self, gap: GapDetectionResult) -> List[AcquisitionResult]:
        """
        Given a detected gap, formulate learning objectives and
        retrieve relevant information from external sources.
        """
        # 1. Formulate search queries using the LLM
        search_queries = self._formulate_queries(gap)
        logger.info("Formulated %d search queries: %s", len(search_queries), search_queries)

        all_results: List[AcquisitionResult] = []

        for sq in search_queries:
            # 2a. arXiv API
            if config.ARXIV_ENABLED:
                arxiv_results = self._search_arxiv(sq)
                all_results.extend(arxiv_results)

            # 2b. Wikipedia API
            if config.WIKIPEDIA_ENABLED:
                wiki_results = self._search_wikipedia(sq)
                all_results.extend(wiki_results)

            # 2c. Semantic Scholar API
            if config.SEMANTIC_SCHOLAR_ENABLED:
                ss_results = self._search_semantic_scholar(sq)
                all_results.extend(ss_results)

        # 3. Fallback: if no Wikipedia results were found, try with the raw query
        wiki_count = sum(1 for r in all_results if r.source_name == "wikipedia")
        if wiki_count == 0 and config.WIKIPEDIA_ENABLED:
            logger.info("No Wikipedia results from formulated queries — trying raw query: '%s'", gap.query)
            fallback_wiki = self._search_wikipedia(gap.query)
            all_results.extend(fallback_wiki)

        logger.info("Acquired %d total results across all sources", len(all_results))
        return all_results

    # ── Query Formulation ────────────────────────────────────────────
    def _formulate_queries(self, gap: GapDetectionResult) -> List[str]:
        """Use the LLM to generate targeted search queries for the gap."""
        prompt = f"""You are a research assistant. A knowledge gap has been detected.

GAP TYPE: {gap.gap_type}
ORIGINAL QUERY: {gap.query}
GAP DESCRIPTION: {gap.description}

Generate 2-3 specific search queries that would help fill this knowledge gap.
Important: The queries will be used across MULTIPLE sources — academic databases (arXiv), Wikipedia, and Semantic Scholar.
So generate queries that are concise and work well as general search terms, NOT overly academic.
At least one query should be a simple, direct version of what the user is asking about.

Respond with ONLY a JSON object:
{{
  "queries": ["query1", "query2", "query3"]
}}"""

        try:
            result = self.llm.generate_json(prompt)
            queries = result.get("queries", [])
            if isinstance(queries, list) and queries:
                return queries[:3]
        except Exception as e:
            logger.warning("Query formulation failed: %s", e)

        # Fallback: use the original query terms
        return [gap.query]

    # ── arXiv API ────────────────────────────────────────────────────
    def _search_arxiv(self, query: str) -> List[AcquisitionResult]:
        """Search arXiv for papers matching the query."""
        # Rate limiting
        elapsed = time.time() - self._last_arxiv_call
        if elapsed < config.ARXIV_RATE_LIMIT_SECONDS:
            time.sleep(config.ARXIV_RATE_LIMIT_SECONDS - elapsed)

        results = []
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=config.ARXIV_MAX_RESULTS,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for paper in client.results(search):
                authors_str = ", ".join(a.name for a in paper.authors)
                categories_str = " ".join(paper.categories) if paper.categories else ""

                results.append(AcquisitionResult(
                    content=f"{paper.title}\n\n{paper.summary}",
                    title=paper.title,
                    source_name="arxiv",
                    source_url=paper.entry_id,
                    source_id=paper.entry_id.split("/")[-1] if paper.entry_id else "",
                    authors=authors_str,
                    categories=categories_str,
                    published_date=paper.published.isoformat() if paper.published else "",
                    journal_ref=paper.journal_ref or "",
                    retrieval_timestamp=datetime.utcnow().isoformat(),
                    raw_metadata={
                        "doi": paper.doi,
                        "primary_category": paper.primary_category,
                        "comment": paper.comment,
                    },
                ))
            self._last_arxiv_call = time.time()
            logger.info("arXiv returned %d results for '%s'", len(results), query[:50])
        except Exception as e:
            logger.error("arXiv search failed: %s", e)

        return results

    # ── Wikipedia API ────────────────────────────────────────────────
    def _search_wikipedia(self, query: str) -> List[AcquisitionResult]:
        """Search Wikipedia for summary information."""
        results = []
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "utf8": 1,
            }
            resp = _session.get(config.WIKIPEDIA_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                # Fetch the actual extract
                extract = self._get_wikipedia_extract(title)
                if extract:
                    results.append(AcquisitionResult(
                        content=f"{title}\n\n{extract}",
                        title=title,
                        source_name="wikipedia",
                        source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        source_id=str(item.get("pageid", "")),
                        retrieval_timestamp=datetime.utcnow().isoformat(),
                    ))
            logger.info("Wikipedia returned %d results for '%s'", len(results), query[:50])
        except Exception as e:
            logger.error("Wikipedia search failed for '%s': %s", query[:50], e)

        return results

    def _get_wikipedia_extract(self, title: str) -> Optional[str]:
        """Fetch the plain-text extract for a Wikipedia article."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exlimit": 1,
            }
            resp = _session.get(config.WIKIPEDIA_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
        except Exception:
            pass
        return None

    # ── Semantic Scholar API ─────────────────────────────────────────
    def _search_semantic_scholar(self, query: str) -> List[AcquisitionResult]:
        """Search Semantic Scholar for papers and citation data."""
        results = []
        try:
            url = f"{config.SEMANTIC_SCHOLAR_API_URL}/paper/search"
            params = {
                "query": query,
                "limit": 5,
                "fields": "title,abstract,authors,year,citationCount,journal,externalIds,url",
            }
            headers = {}
            if config.SEMANTIC_SCHOLAR_API_KEY:
                headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

            # Try up to 2 times with increasing backoff for rate limits
            resp = None
            for attempt in range(2):
                resp = _session.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code == 429:
                    wait = 3 * (attempt + 1)
                    logger.warning("Semantic Scholar rate limited (attempt %d), waiting %ds...", attempt + 1, wait)
                    time.sleep(wait)
                else:
                    break

            if resp is None or resp.status_code == 429:
                logger.warning("Semantic Scholar still rate limited after retries — skipping for '%s'", query[:50])
                return results

            resp.raise_for_status()
            data = resp.json()

            for paper in data.get("data", []):
                abstract = paper.get("abstract", "") or ""
                title = paper.get("title", "") or ""
                authors = ", ".join(
                    a.get("name", "") for a in (paper.get("authors") or [])
                )
                journal = paper.get("journal", {})
                journal_name = journal.get("name", "") if journal else ""
                citation_count = paper.get("citationCount", 0) or 0

                results.append(AcquisitionResult(
                    content=f"{title}\n\n{abstract}",
                    title=title,
                    source_name="semantic_scholar",
                    source_url=paper.get("url", ""),
                    source_id=paper.get("paperId", ""),
                    authors=authors,
                    published_date=str(paper.get("year", "")),
                    journal_ref=journal_name,
                    citation_count=citation_count,
                    retrieval_timestamp=datetime.utcnow().isoformat(),
                    raw_metadata={
                        "externalIds": paper.get("externalIds", {}),
                    },
                ))
            logger.info("Semantic Scholar returned %d results for '%s'", len(results), query[:50])
        except Exception as e:
            logger.error("Semantic Scholar search failed: %s", e)

        return results


# ── Singleton ────────────────────────────────────────────────────────
_acquisition = None


def get_knowledge_acquisition() -> KnowledgeAcquisition:
    global _acquisition
    if _acquisition is None:
        _acquisition = KnowledgeAcquisition()
    return _acquisition
