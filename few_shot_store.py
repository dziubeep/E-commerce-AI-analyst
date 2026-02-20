"""
few_shot_store.py — Dynamic Few-Shot Example Store

Replaces the static business_logic.txt with a learning system that logs
every successful Q&A pair and retrieves the most relevant examples for
future prompts using SQLite FTS5 (full-text search with BM25 ranking).

Key design decisions:
  • Zero new dependencies — uses SQLite FTS5, which is built-in.
  • Schema-scoped — examples are tagged with a hash of the DB schema,
    so examples from one dataset never pollute another.
  • Deduplication — won't store the same SQL twice for the same schema.
  • Capped — keeps at most MAX_EXAMPLES per schema (oldest pruned).
  • Formatted output — returns examples in the exact format the LLM
    prompt expects (Question / Answer / Explanation blocks).

Usage:
    store = FewShotStore()                         # auto-creates DB
    store.log(question, sql, explanation, "High", schema_hash)
    examples = store.retrieve("revenue by country", schema_hash, top_k=5)
    prompt_text = store.format_for_prompt(examples)
"""

import hashlib
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORE_PATH = os.path.join(BASE_DIR, "few_shot_store.db")
MAX_EXAMPLES_PER_SCHEMA = 500   # prune oldest beyond this
DEFAULT_TOP_K = 5               # examples to inject into prompt


# ── Data class for retrieved examples ────────────────────────────────────
@dataclass
class FewShotExample:
    """A single few-shot example retrieved from the store."""
    id: int
    question: str
    sql: str
    explanation: str
    confidence: str
    schema_hash: str
    created_at: str
    rank_score: float = 0.0  # BM25 relevance score (lower = better)


class FewShotStore:
    """
    Dynamic few-shot example store backed by SQLite + FTS5.

    Thread-safe: uses a threading lock around all write operations.
    The FTS5 virtual table enables fast BM25-ranked full-text search
    over stored questions, so the most relevant examples are injected
    into the LLM prompt automatically.
    """

    def __init__(self, db_path: str = DEFAULT_STORE_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── Schema Setup ─────────────────────────────────────────────────────
    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        # Main table: stores all examples
        cur.execute("""
            CREATE TABLE IF NOT EXISTS few_shot_examples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT    NOT NULL,
                sql_query   TEXT    NOT NULL,
                explanation TEXT    NOT NULL DEFAULT '',
                confidence  TEXT    NOT NULL DEFAULT 'Medium',
                schema_hash TEXT    NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for fast schema-scoped lookups
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_schema_hash
            ON few_shot_examples(schema_hash)
        """)

        # Unique constraint: don't store identical SQL for same schema
        # (We use a separate unique index on schema_hash + sql_query hash)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_dedup
            ON few_shot_examples(schema_hash, sql_query)
        """)

        # FTS5 virtual table for full-text search on questions
        # content= links it to the main table (external content FTS5 table)
        # We use a standalone FTS5 table synced manually for reliability.
        try:
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS few_shot_fts
                USING fts5(
                    question,
                    content='few_shot_examples',
                    content_rowid='id'
                )
            """)
        except sqlite3.OperationalError:
            # FTS5 not available — fall back to LIKE-based search
            logger.warning(
                "SQLite FTS5 not available — falling back to LIKE search. "
                "Retrieval will be slower but still functional."
            )

        conn.commit()
        conn.close()

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def compute_schema_hash(schema_string: str) -> str:
        """Compute a SHA-256 hash of the schema string.

        This scopes few-shot examples to a specific dataset structure,
        so examples from one uploaded CSV don't leak into another.
        """
        return hashlib.sha256(schema_string.encode("utf-8")).hexdigest()[:16]

    def _has_fts5(self) -> bool:
        """Check if the FTS5 table exists."""
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='few_shot_fts'"
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    # ── Write: Log a Successful Example ──────────────────────────────────
    def log(
        self,
        question: str,
        sql: str,
        explanation: str,
        confidence: str,
        schema_hash: str,
    ) -> Optional[int]:
        """
        Store a successful Q&A pair in the few-shot store.

        Deduplication: if the exact same SQL already exists for this
        schema, the example is skipped (returns None).

        Auto-prunes oldest examples when MAX_EXAMPLES_PER_SCHEMA is
        exceeded.

        Returns:
            Row ID of the inserted example, or None if deduplicated.
        """
        question = question.strip()
        sql = sql.strip()
        explanation = explanation.strip()

        # Skip trivially short questions or empty SQL
        if len(question) < 5 or len(sql) < 10:
            return None

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            cur = conn.cursor()

            try:
                # Dedup check: same SQL for same schema?
                cur.execute(
                    "SELECT id FROM few_shot_examples "
                    "WHERE schema_hash = ? AND sql_query = ?",
                    (schema_hash, sql),
                )
                if cur.fetchone() is not None:
                    logger.debug("Dedup: SQL already stored for schema %s",
                                 schema_hash[:8])
                    return None

                # Insert the new example
                cur.execute(
                    "INSERT INTO few_shot_examples "
                    "(question, sql_query, explanation, confidence, schema_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (question, sql, explanation, confidence, schema_hash),
                )
                row_id = cur.lastrowid

                # Sync FTS5 index
                if self._has_fts5():
                    cur.execute(
                        "INSERT INTO few_shot_fts(rowid, question) "
                        "VALUES (?, ?)",
                        (row_id, question),
                    )

                conn.commit()
                logger.info(
                    "Logged few-shot example #%d (schema %s): %s",
                    row_id, schema_hash[:8], question[:60],
                )

                # Prune if over limit
                self._prune(cur, schema_hash)
                conn.commit()

                return row_id

            except Exception as e:
                logger.error("Failed to log few-shot example: %s", e)
                conn.rollback()
                return None
            finally:
                conn.close()

    def _prune(self, cur: sqlite3.Cursor, schema_hash: str) -> None:
        """Remove oldest examples if count exceeds MAX_EXAMPLES_PER_SCHEMA."""
        cur.execute(
            "SELECT COUNT(*) FROM few_shot_examples WHERE schema_hash = ?",
            (schema_hash,),
        )
        count = cur.fetchone()[0]

        if count > MAX_EXAMPLES_PER_SCHEMA:
            excess = count - MAX_EXAMPLES_PER_SCHEMA
            # Get IDs of the oldest rows to delete
            cur.execute(
                "SELECT id FROM few_shot_examples "
                "WHERE schema_hash = ? "
                "ORDER BY created_at ASC LIMIT ?",
                (schema_hash, excess),
            )
            ids_to_delete = [row[0] for row in cur.fetchall()]

            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                cur.execute(
                    f"DELETE FROM few_shot_examples "
                    f"WHERE id IN ({placeholders})",
                    ids_to_delete,
                )
                # Clean FTS5
                if self._has_fts5():
                    for rid in ids_to_delete:
                        try:
                            cur.execute(
                                "INSERT INTO few_shot_fts"
                                "(few_shot_fts, rowid, question) "
                                "VALUES ('delete', ?, '')",
                                (rid,),
                            )
                        except sqlite3.OperationalError:
                            pass

                logger.info("Pruned %d old examples for schema %s",
                            len(ids_to_delete), schema_hash[:8])

    # ── Read: Retrieve Relevant Examples ─────────────────────────────────
    def retrieve(
        self,
        question: str,
        schema_hash: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[FewShotExample]:
        """
        Find the most relevant few-shot examples for a given question.

        Uses FTS5 BM25 ranking when available, falls back to keyword
        matching with LIKE otherwise.

        Args:
            question:    The user's natural-language question.
            schema_hash: Scope to examples from the same dataset.
            top_k:       Maximum number of examples to return.

        Returns:
            List of FewShotExample, ordered by relevance (best first).
        """
        question = question.strip()
        if not question:
            return []

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        try:
            if self._has_fts5():
                examples = self._retrieve_fts5(conn, question,
                                                schema_hash, top_k)
            else:
                examples = self._retrieve_like(conn, question,
                                                schema_hash, top_k)
            return examples
        finally:
            conn.close()

    def _retrieve_fts5(
        self,
        conn: sqlite3.Connection,
        question: str,
        schema_hash: str,
        top_k: int,
    ) -> list[FewShotExample]:
        """Retrieve using FTS5 BM25 ranking."""
        # Build FTS5 query: tokenize the question into OR-ed terms
        # FTS5 uses implicit AND by default; OR gives broader recall
        tokens = self._tokenize_for_fts(question)
        if not tokens:
            return self._retrieve_like(conn, question, schema_hash, top_k)

        fts_query = " OR ".join(tokens)

        try:
            rows = conn.execute(
                """
                SELECT e.id, e.question, e.sql_query, e.explanation,
                       e.confidence, e.schema_hash, e.created_at,
                       bm25(few_shot_fts) AS rank_score
                FROM few_shot_fts f
                JOIN few_shot_examples e ON e.id = f.rowid
                WHERE few_shot_fts MATCH ?
                  AND e.schema_hash = ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (fts_query, schema_hash, top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed (%s), falling back to LIKE", e)
            return self._retrieve_like(conn, question, schema_hash, top_k)

        return [
            FewShotExample(
                id=r["id"],
                question=r["question"],
                sql=r["sql_query"],
                explanation=r["explanation"],
                confidence=r["confidence"],
                schema_hash=r["schema_hash"],
                created_at=r["created_at"],
                rank_score=r["rank_score"],
            )
            for r in rows
        ]

    def _retrieve_like(
        self,
        conn: sqlite3.Connection,
        question: str,
        schema_hash: str,
        top_k: int,
    ) -> list[FewShotExample]:
        """Fallback retrieval using LIKE keyword matching."""
        keywords = self._extract_keywords(question)
        if not keywords:
            # Return most recent examples as fallback
            rows = conn.execute(
                "SELECT * FROM few_shot_examples "
                "WHERE schema_hash = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (schema_hash, top_k),
            ).fetchall()
        else:
            # Build WHERE clause: score = count of matching keywords
            # Use a subquery approach for scoring
            like_clauses = " + ".join(
                f"(CASE WHEN LOWER(question) LIKE '%{kw}%' THEN 1 ELSE 0 END)"
                for kw in keywords
            )
            rows = conn.execute(
                f"SELECT *, ({like_clauses}) AS match_score "
                f"FROM few_shot_examples "
                f"WHERE schema_hash = ? AND ({like_clauses}) > 0 "
                f"ORDER BY match_score DESC, created_at DESC "
                f"LIMIT ?",
                (schema_hash, top_k),
            ).fetchall()

        return [
            FewShotExample(
                id=r["id"],
                question=r["question"],
                sql=r["sql_query"],
                explanation=r["explanation"],
                confidence=r["confidence"],
                schema_hash=r["schema_hash"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    @staticmethod
    def _tokenize_for_fts(question: str) -> list[str]:
        """Extract meaningful tokens for FTS5 MATCH query."""
        import re
        # Remove punctuation, lowercase, split
        words = re.findall(r'\b[a-z]{2,}\b', question.lower())
        # Remove common stop words
        stop_words = {
            "the", "is", "at", "in", "on", "of", "to", "and", "or",
            "for", "by", "an", "it", "do", "me", "my", "we", "us",
            "am", "be", "so", "if", "no", "as", "up",
            "what", "how", "show", "give", "tell", "can", "you",
            "please", "which", "where", "when", "who", "this", "that",
            "from", "with", "are", "was", "were", "has", "have", "had",
            "will", "would", "could", "should",
        }
        return [w for w in words if w not in stop_words]

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        """Extract keywords for LIKE-based fallback (same logic, sanitized)."""
        import re
        words = re.findall(r'\b[a-z]{3,}\b', question.lower())
        stop_words = {
            "the", "is", "at", "in", "on", "of", "to", "and", "or",
            "for", "by", "an", "it", "do", "me", "my", "we", "us",
            "what", "how", "show", "give", "tell", "can", "you",
            "please", "which", "where", "when", "who", "this", "that",
            "from", "with", "are", "was", "were", "has", "have", "had",
            "will", "would", "could", "should",
        }
        # Sanitize for SQL LIKE (escape special chars)
        sanitized = []
        for w in words:
            if w not in stop_words:
                clean = w.replace("'", "''").replace("%", "").replace("_", "")
                if len(clean) >= 3:
                    sanitized.append(clean)
        return sanitized[:10]  # limit to prevent massive queries

    # ── Format for Prompt Injection ──────────────────────────────────────
    @staticmethod
    def format_for_prompt(examples: list[FewShotExample]) -> str:
        """
        Format retrieved examples into the text block injected into
        the LLM system prompt.

        Mirrors the format of the old business_logic.txt so the LLM
        sees a consistent structure.
        """
        if not examples:
            return "(No learned examples available yet.)"

        parts = []
        for i, ex in enumerate(examples, 1):
            block = (
                f"# ── Learned Example {i} ──\n"
                f"Question: {ex.question}\n"
                f"Answer:\n{ex.sql}\n"
                f"Explanation: {ex.explanation}"
            )
            parts.append(block)

        header = (
            f"# Dynamic Few-Shot Examples ({len(examples)} most relevant)\n"
            f"# These are real Q&A pairs that previously succeeded on "
            f"this dataset.\n"
        )
        return header + "\n\n".join(parts)

    # ── Stats & Maintenance ──────────────────────────────────────────────
    def get_stats(self, schema_hash: str | None = None) -> dict:
        """Return counts of stored examples, optionally filtered by schema."""
        conn = sqlite3.connect(self._db_path)
        try:
            if schema_hash:
                row = conn.execute(
                    "SELECT COUNT(*) FROM few_shot_examples "
                    "WHERE schema_hash = ?",
                    (schema_hash,),
                ).fetchone()
                return {"schema_hash": schema_hash, "count": row[0]}
            else:
                rows = conn.execute(
                    "SELECT schema_hash, COUNT(*) as cnt "
                    "FROM few_shot_examples "
                    "GROUP BY schema_hash "
                    "ORDER BY cnt DESC"
                ).fetchall()
                return {
                    "total": sum(r[1] for r in rows),
                    "schemas": {r[0]: r[1] for r in rows},
                }
        finally:
            conn.close()

    def clear(self, schema_hash: str | None = None) -> int:
        """Delete examples. If schema_hash given, only for that schema."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                if schema_hash:
                    # Get IDs to clean FTS
                    ids = [
                        r[0] for r in conn.execute(
                            "SELECT id FROM few_shot_examples "
                            "WHERE schema_hash = ?",
                            (schema_hash,),
                        ).fetchall()
                    ]
                    conn.execute(
                        "DELETE FROM few_shot_examples WHERE schema_hash = ?",
                        (schema_hash,),
                    )
                else:
                    ids = [
                        r[0] for r in conn.execute(
                            "SELECT id FROM few_shot_examples"
                        ).fetchall()
                    ]
                    conn.execute("DELETE FROM few_shot_examples")

                # Clean FTS5 index
                if self._has_fts5() and ids:
                    for rid in ids:
                        try:
                            conn.execute(
                                "INSERT INTO few_shot_fts"
                                "(few_shot_fts, rowid, question) "
                                "VALUES ('delete', ?, '')",
                                (rid,),
                            )
                        except sqlite3.OperationalError:
                            pass

                conn.commit()
                logger.info("Cleared %d few-shot examples", len(ids))
                return len(ids)
            finally:
                conn.close()

    def seed_from_business_logic(
        self,
        business_logic_path: str,
        schema_hash: str,
    ) -> int:
        """
        One-time migration: parse business_logic.txt and seed the store.

        This preserves the gold-standard examples as the initial training
        data for a schema, while the system continues to learn new ones.

        Returns:
            Number of examples seeded.
        """
        if not os.path.exists(business_logic_path):
            logger.warning("business_logic.txt not found at %s",
                           business_logic_path)
            return 0

        with open(business_logic_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse Question/Answer/Explanation blocks
        import re
        pattern = re.compile(
            r"Question:\s*(.+?)\n"
            r"Answer:\s*\n(.*?)\n"
            r"Explanation:\s*(.+?)(?=\n\n# ──|\Z)",
            re.DOTALL,
        )

        count = 0
        for match in pattern.finditer(content):
            question = match.group(1).strip()
            sql = match.group(2).strip()
            explanation = match.group(3).strip()

            result = self.log(
                question=question,
                sql=sql,
                explanation=explanation,
                confidence="High",  # gold-standard → High confidence
                schema_hash=schema_hash,
            )
            if result is not None:
                count += 1

        logger.info(
            "Seeded %d examples from business_logic.txt into schema %s",
            count, schema_hash[:8],
        )
        return count
