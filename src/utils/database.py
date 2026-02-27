"""
SQLite database helper – schema creation, CRUD operations, and queries
for papers, knowledge entries, logical rules, revision history, gap logs,
and epistemic assessment history.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# ─── Schema SQL ──────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         TEXT,
    abstract        TEXT,
    categories      TEXT,
    doi             TEXT,
    journal_ref     TEXT,
    submitter       TEXT,
    report_no       TEXT,
    comments        TEXT,
    license         TEXT,
    versions        TEXT,          -- JSON string
    created_date    TEXT,
    update_date     TEXT,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL,       -- 'arxiv', 'wikipedia', 'semantic_scholar', 'ingested'
    source_id       TEXT,                -- external id (e.g. arxiv paper id)
    content         TEXT NOT NULL,
    title           TEXT,
    confidence_score REAL DEFAULT 0.5,
    trust_score     REAL DEFAULT 0.5,
    version         INTEGER DEFAULT 1,
    is_active       INTEGER DEFAULT 1,   -- 1 = active, 0 = superseded
    superseded_by   INTEGER,             -- FK → knowledge_entries.id
    conflict_flag   INTEGER DEFAULT 0,   -- 1 = unresolved conflict
    metadata_json   TEXT,                -- arbitrary JSON
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS logical_rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    antecedent      TEXT NOT NULL,
    consequent      TEXT NOT NULL,
    confidence      REAL DEFAULT 0.5,
    source_entry_id INTEGER,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_entry_id) REFERENCES knowledge_entries(id)
);

CREATE TABLE IF NOT EXISTS revision_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id        INTEGER NOT NULL,
    old_content     TEXT,
    new_content     TEXT,
    reason          TEXT,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id)
);

CREATE TABLE IF NOT EXISTS gap_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query           TEXT NOT NULL,
    gap_type        TEXT NOT NULL,
    description     TEXT,
    resolved        INTEGER DEFAULT 0,
    resolution_info TEXT,
    resolved_entry_ids TEXT DEFAULT '[]',  -- JSON array of knowledge_entries.id
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS epistemic_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query           TEXT NOT NULL,
    retrieval_score REAL,
    coverage_score  REAL,
    llm_score       REAL,
    composite_score REAL,
    gap_detected    INTEGER DEFAULT 0,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers(categories);
CREATE INDEX IF NOT EXISTS idx_ke_source_type    ON knowledge_entries(source_type);
CREATE INDEX IF NOT EXISTS idx_ke_active         ON knowledge_entries(is_active);
CREATE INDEX IF NOT EXISTS idx_gap_resolved      ON gap_log(resolved);
"""


# ─── Connection helpers ──────────────────────────────────────────────

def get_connection(db_path: str = config.SQLITE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_cursor(db_path: str = config.SQLITE_PATH):
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = config.SQLITE_PATH):
    """Create all tables & indexes if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript(_SCHEMA)
    # Migrate: ensure resolved_entry_ids column exists for older databases
    try:
        cur = conn.execute("PRAGMA table_info(gap_log)")
        columns = {row[1] for row in cur.fetchall()}
        if "resolved_entry_ids" not in columns:
            conn.execute("ALTER TABLE gap_log ADD COLUMN resolved_entry_ids TEXT DEFAULT '[]'")
            conn.commit()
            logger.info("Migrated gap_log: added resolved_entry_ids column")
    except Exception as e:
        logger.warning("Migration check for gap_log failed: %s", e)
    conn.close()
    logger.info("Database initialised at %s", db_path)


# ─── Papers CRUD ─────────────────────────────────────────────────────

def insert_paper(paper: Dict[str, Any], db_path: str = config.SQLITE_PATH):
    """Insert a single arXiv paper record."""
    import json as _json

    sql = """
    INSERT OR IGNORE INTO papers
        (id, title, authors, abstract, categories, doi, journal_ref,
         submitter, report_no, comments, license, versions,
         created_date, update_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    versions_str = _json.dumps(paper.get("versions", []))
    # Determine created_date from first version
    created = None
    versions = paper.get("versions", [])
    if versions and isinstance(versions, list) and len(versions) > 0:
        created = versions[0].get("created", None)

    with get_cursor(db_path) as cur:
        cur.execute(sql, (
            paper.get("id", ""),
            paper.get("title", ""),
            paper.get("authors", ""),
            paper.get("abstract", ""),
            paper.get("categories", ""),
            paper.get("doi", None),
            paper.get("journal-ref", None),
            paper.get("submitter", None),
            paper.get("report-no", None),
            paper.get("comments", None),
            paper.get("license", None),
            versions_str,
            created,
            paper.get("update_date", None),
        ))


def insert_papers_batch(papers: List[Dict[str, Any]], db_path: str = config.SQLITE_PATH):
    """Bulk insert papers."""
    import json as _json

    sql = """
    INSERT OR IGNORE INTO papers
        (id, title, authors, abstract, categories, doi, journal_ref,
         submitter, report_no, comments, license, versions,
         created_date, update_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = []
    for p in papers:
        versions = p.get("versions", [])
        created = None
        if versions and isinstance(versions, list) and len(versions) > 0:
            created = versions[0].get("created", None)
        rows.append((
            p.get("id", ""),
            p.get("title", ""),
            p.get("authors", ""),
            p.get("abstract", ""),
            p.get("categories", ""),
            p.get("doi", None),
            p.get("journal-ref", None),
            p.get("submitter", None),
            p.get("report-no", None),
            p.get("comments", None),
            p.get("license", None),
            _json.dumps(versions),
            created,
            p.get("update_date", None),
        ))
    with get_cursor(db_path) as cur:
        cur.executemany(sql, rows)


def get_paper(paper_id: str, db_path: str = config.SQLITE_PATH) -> Optional[Dict]:
    with get_cursor(db_path) as cur:
        cur.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def search_papers(query: str, limit: int = 20, db_path: str = config.SQLITE_PATH) -> List[Dict]:
    with get_cursor(db_path) as cur:
        cur.execute(
            "SELECT * FROM papers WHERE title LIKE ? OR abstract LIKE ? LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        )
        return [dict(r) for r in cur.fetchall()]


def count_papers(db_path: str = config.SQLITE_PATH) -> int:
    with get_cursor(db_path) as cur:
        cur.execute("SELECT COUNT(*) FROM papers")
        return cur.fetchone()[0]


# ─── Knowledge Entries CRUD ──────────────────────────────────────────

def insert_knowledge_entry(
    source_type: str,
    content: str,
    title: str = "",
    source_id: str = None,
    confidence: float = 0.5,
    trust: float = 0.5,
    metadata_json: str = None,
    db_path: str = config.SQLITE_PATH,
) -> int:
    sql = """
    INSERT INTO knowledge_entries
        (source_type, source_id, content, title, confidence_score, trust_score, metadata_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_cursor(db_path) as cur:
        cur.execute(sql, (source_type, source_id, content, title, confidence, trust, metadata_json))
        return cur.lastrowid


def get_knowledge_entry(entry_id: int, db_path: str = config.SQLITE_PATH) -> Optional[Dict]:
    with get_cursor(db_path) as cur:
        cur.execute("SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_active_entries(source_type: str = None, limit: int = 100, db_path: str = config.SQLITE_PATH) -> List[Dict]:
    with get_cursor(db_path) as cur:
        if source_type:
            cur.execute(
                "SELECT * FROM knowledge_entries WHERE is_active = 1 AND source_type = ? ORDER BY updated_at DESC LIMIT ?",
                (source_type, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM knowledge_entries WHERE is_active = 1 ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]


def update_knowledge_entry(
    entry_id: int,
    new_content: str = None,
    new_trust: float = None,
    new_confidence: float = None,
    superseded_by: int = None,
    is_active: int = None,
    conflict_flag: int = None,
    db_path: str = config.SQLITE_PATH,
):
    """Update fields of a knowledge entry. Only supplied fields are changed."""
    updates = []
    params = []
    if new_content is not None:
        updates.append("content = ?")
        params.append(new_content)
    if new_trust is not None:
        updates.append("trust_score = ?")
        params.append(new_trust)
    if new_confidence is not None:
        updates.append("confidence_score = ?")
        params.append(new_confidence)
    if superseded_by is not None:
        updates.append("superseded_by = ?")
        params.append(superseded_by)
    if is_active is not None:
        updates.append("is_active = ?")
        params.append(is_active)
    if conflict_flag is not None:
        updates.append("conflict_flag = ?")
        params.append(conflict_flag)
    if not updates:
        return
    updates.append("version = version + 1")
    updates.append("updated_at = datetime('now')")
    params.append(entry_id)
    sql = f"UPDATE knowledge_entries SET {', '.join(updates)} WHERE id = ?"
    with get_cursor(db_path) as cur:
        cur.execute(sql, params)


# ─── Logical Rules ──────────────────────────────────────────────────

def insert_rule(antecedent: str, consequent: str, confidence: float = 0.5,
                source_entry_id: int = None, db_path: str = config.SQLITE_PATH) -> int:
    with get_cursor(db_path) as cur:
        cur.execute(
            "INSERT INTO logical_rules (antecedent, consequent, confidence, source_entry_id) VALUES (?,?,?,?)",
            (antecedent, consequent, confidence, source_entry_id),
        )
        return cur.lastrowid


def get_rules(category: str = None, limit: int = 50, db_path: str = config.SQLITE_PATH) -> List[Dict]:
    with get_cursor(db_path) as cur:
        if category:
            cur.execute(
                "SELECT * FROM logical_rules WHERE antecedent LIKE ? OR consequent LIKE ? LIMIT ?",
                (f"%{category}%", f"%{category}%", limit),
            )
        else:
            cur.execute("SELECT * FROM logical_rules ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# ─── Revision History ────────────────────────────────────────────────

def insert_revision(entry_id: int, old_content: str, new_content: str, reason: str,
                    db_path: str = config.SQLITE_PATH) -> int:
    with get_cursor(db_path) as cur:
        cur.execute(
            "INSERT INTO revision_history (entry_id, old_content, new_content, reason) VALUES (?,?,?,?)",
            (entry_id, old_content, new_content, reason),
        )
        return cur.lastrowid


def get_revisions(entry_id: int = None, limit: int = 50, db_path: str = config.SQLITE_PATH) -> List[Dict]:
    with get_cursor(db_path) as cur:
        if entry_id:
            cur.execute("SELECT * FROM revision_history WHERE entry_id = ? ORDER BY timestamp DESC LIMIT ?",
                        (entry_id, limit))
        else:
            cur.execute("SELECT * FROM revision_history ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# ─── Gap Log ─────────────────────────────────────────────────────────

def insert_gap(query: str, gap_type: str, description: str = "",
               db_path: str = config.SQLITE_PATH) -> int:
    with get_cursor(db_path) as cur:
        cur.execute(
            "INSERT INTO gap_log (query, gap_type, description) VALUES (?,?,?)",
            (query, gap_type, description),
        )
        return cur.lastrowid


def resolve_gap(gap_id: int, resolution_info: str = "", entry_ids: list = None, db_path: str = config.SQLITE_PATH):
    import json as _json
    ids_json = _json.dumps(entry_ids or [])
    with get_cursor(db_path) as cur:
        cur.execute(
            "UPDATE gap_log SET resolved = 1, resolution_info = ?, resolved_entry_ids = ? WHERE id = ?",
            (resolution_info, ids_json, gap_id),
        )


def get_gaps(resolved: int = None, limit: int = 50, db_path: str = config.SQLITE_PATH) -> List[Dict]:
    with get_cursor(db_path) as cur:
        if resolved is not None:
            cur.execute("SELECT * FROM gap_log WHERE resolved = ? ORDER BY timestamp DESC LIMIT ?",
                        (resolved, limit))
        else:
            cur.execute("SELECT * FROM gap_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


def get_entries_by_ids(entry_ids: List[int], db_path: str = config.SQLITE_PATH) -> List[Dict]:
    """Fetch knowledge entries by a list of IDs."""
    if not entry_ids:
        return []
    with get_cursor(db_path) as cur:
        placeholders = ",".join("?" for _ in entry_ids)
        cur.execute(f"SELECT * FROM knowledge_entries WHERE id IN ({placeholders})", entry_ids)
        return [dict(r) for r in cur.fetchall()]


def get_gap_with_entries(gap_id: int, db_path: str = config.SQLITE_PATH) -> Optional[Dict]:
    """Get a single gap log entry along with the knowledge entries that resolved it."""
    import json as _json
    with get_cursor(db_path) as cur:
        cur.execute("SELECT * FROM gap_log WHERE id = ?", (gap_id,))
        row = cur.fetchone()
        if not row:
            return None
        gap = dict(row)
        entry_ids_raw = gap.get("resolved_entry_ids", "[]")
        try:
            entry_ids = _json.loads(entry_ids_raw) if entry_ids_raw else []
        except (ValueError, TypeError):
            entry_ids = []
        gap["resolved_entries"] = get_entries_by_ids(entry_ids, db_path) if entry_ids else []
        return gap


# ─── Epistemic Log ───────────────────────────────────────────────────

def insert_epistemic_log(query: str, retrieval: float, coverage: float,
                         llm_score: float, composite: float, gap_detected: bool,
                         db_path: str = config.SQLITE_PATH) -> int:
    with get_cursor(db_path) as cur:
        cur.execute(
            "INSERT INTO epistemic_log (query, retrieval_score, coverage_score, llm_score, composite_score, gap_detected) VALUES (?,?,?,?,?,?)",
            (query, retrieval, coverage, llm_score, composite, int(gap_detected)),
        )
        return cur.lastrowid


def get_epistemic_logs(limit: int = 50, db_path: str = config.SQLITE_PATH) -> List[Dict]:
    with get_cursor(db_path) as cur:
        cur.execute("SELECT * FROM epistemic_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


# ─── Statistics ──────────────────────────────────────────────────────

def get_stats(db_path: str = config.SQLITE_PATH) -> Dict[str, Any]:
    """Return quick counts for the dashboard."""
    with get_cursor(db_path) as cur:
        stats = {}
        for table in ("papers", "knowledge_entries", "logical_rules", "revision_history", "gap_log", "epistemic_log"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM knowledge_entries WHERE is_active = 1")
        stats["active_entries"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM gap_log WHERE resolved = 1")
        stats["gaps_resolved"] = cur.fetchone()[0]
        return stats
