"""
LIMPHA MEMORY — Core storage for Yent's consciousness persistence.

Single SQLite database. FTS5 full-text search. Autonomous.

Tables:
- conversations: Every prompt/response with AMK state snapshot
- conversations_fts: FTS5 virtual table (auto-synced via triggers)
- sessions: Session metadata
- shards: Graduated episodes for delta training

All operations async via aiosqlite.
"""

import asyncio
import aiosqlite
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class Conversation:
    """One turn of dialogue."""
    id: int
    timestamp: float
    session_id: str
    prompt: str
    response: str
    # AMK state snapshot
    temperature: float
    destiny: float
    pain: float
    tension: float
    debt: float
    velocity: int
    alpha: float
    # Computed
    quality: float
    access_count: int


@dataclass
class ShardRecord:
    """An episode that graduated to training shard."""
    id: int
    conversation_id: int
    shard_path: str
    graduated_at: float
    reason: str
    priority: float
    training_status: str
    training_loss: Optional[float]


SCHEMA = """
-- Every conversation turn
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    -- AMK state at generation time
    temperature REAL DEFAULT 0.0,
    destiny REAL DEFAULT 0.0,
    pain REAL DEFAULT 0.0,
    tension REAL DEFAULT 0.0,
    debt REAL DEFAULT 0.0,
    velocity INTEGER DEFAULT 1,
    alpha REAL DEFAULT 0.0,
    -- Computed quality
    quality REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_quality ON conversations(quality DESC);

-- FTS5 full-text search over conversations
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    prompt,
    response,
    content=conversations,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with conversations table
CREATE TRIGGER IF NOT EXISTS conv_fts_insert AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;

CREATE TRIGGER IF NOT EXISTS conv_fts_delete AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
END;

CREATE TRIGGER IF NOT EXISTS conv_fts_update AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;

-- Session metadata
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    last_active REAL NOT NULL,
    turn_count INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.0
);

-- Shard records: graduated episodes for training
CREATE TABLE IF NOT EXISTS shards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER UNIQUE NOT NULL,
    shard_path TEXT NOT NULL,
    graduated_at REAL NOT NULL,
    reason TEXT DEFAULT '',
    priority REAL DEFAULT 0.0,
    training_status TEXT DEFAULT 'pending',
    training_loss REAL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_shards_status ON shards(training_status);
CREATE INDEX IF NOT EXISTS idx_shards_graduated ON shards(graduated_at DESC);
"""


class LimphaMemory:
    """
    Yent's memory. SQLite + FTS5. Fully autonomous.

    Usage:
        async with LimphaMemory() as mem:
            conv_id = await mem.store(prompt, response, amk_state)
            results = await mem.search("consciousness")
            candidates = await mem.find_shard_candidates()
    """

    # Shard graduation thresholds
    SHARD_MIN_QUALITY = 0.7
    SHARD_MIN_ACCESS = 3
    SHARD_MIN_COHERENCE = 0.3

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".yent" / "limpha.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        self._session_id: str = str(uuid.uuid4())[:8]

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Connect and initialize schema."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        # Enable WAL mode for concurrent reads during writes
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        # Start session
        now = time.time()
        await self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at, last_active) VALUES (?, ?, ?)",
            (self._session_id, now, now),
        )
        await self._conn.commit()

    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ═══════════════════════════════════════════════════════════════════════
    # STORE — after every generation, automatically
    # ═══════════════════════════════════════════════════════════════════════

    async def store(
        self,
        prompt: str,
        response: str,
        amk_state: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store a conversation turn. Called automatically after each generation.

        amk_state: dict with keys temperature, destiny, pain, tension, debt, velocity, alpha
        Returns conversation ID.
        """
        if amk_state is None:
            amk_state = {}

        now = time.time()
        quality = self._compute_quality(prompt, response, amk_state)

        cursor = await self._conn.execute(
            """INSERT INTO conversations
            (timestamp, session_id, prompt, response,
             temperature, destiny, pain, tension, debt, velocity, alpha,
             quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                self._session_id,
                prompt,
                response,
                amk_state.get("temperature", 0.0),
                amk_state.get("destiny", 0.0),
                amk_state.get("pain", 0.0),
                amk_state.get("tension", 0.0),
                amk_state.get("debt", 0.0),
                amk_state.get("velocity", 1),
                amk_state.get("alpha", 0.0),
                quality,
            ),
        )
        await self._conn.commit()
        conv_id = cursor.lastrowid

        # Update session
        await self._conn.execute(
            """UPDATE sessions SET
                last_active = ?,
                turn_count = turn_count + 1,
                avg_quality = (avg_quality * turn_count + ?) / (turn_count + 1)
            WHERE session_id = ?""",
            (now, quality, self._session_id),
        )
        await self._conn.commit()

        return conv_id

    def _compute_quality(
        self, prompt: str, response: str, state: Dict[str, Any]
    ) -> float:
        """
        Compute quality score for a conversation turn.

        Factors:
        - Response length (too short = low quality, sweet spot = higher)
        - Prompt-response ratio (not just echoing)
        - Not empty
        """
        if not response.strip():
            return 0.0

        resp_len = len(response.strip())
        prompt_len = max(len(prompt.strip()), 1)

        # Length score: short responses are low quality, diminishing returns after ~200 chars
        if resp_len < 10:
            length_score = 0.1
        elif resp_len < 50:
            length_score = 0.3
        elif resp_len < 200:
            length_score = 0.5 + 0.3 * (resp_len - 50) / 150
        else:
            length_score = 0.8

        # Ratio score: response should be meaningfully different from prompt
        ratio = resp_len / prompt_len
        if ratio < 0.3:
            ratio_score = 0.2  # Too short relative to prompt
        elif ratio > 10:
            ratio_score = 0.6  # Very long, might be rambling
        else:
            ratio_score = 0.7

        # Combined
        quality = 0.6 * length_score + 0.4 * ratio_score

        # Clamp
        return max(0.0, min(1.0, quality))

    # ═══════════════════════════════════════════════════════════════════════
    # SEARCH — FTS5 full-text search
    # ═══════════════════════════════════════════════════════════════════════

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search over all conversations.

        Supports FTS5 syntax:
        - "word1 word2" (AND)
        - "word1 OR word2"
        - '"exact phrase"'
        - "word*" (prefix)
        - "prompt:word" (column-specific)

        Results ranked by BM25.
        """
        if not query.strip():
            return []

        try:
            cursor = await self._conn.execute(
                """SELECT c.id, c.timestamp, c.session_id,
                          c.prompt, c.response, c.quality, c.access_count,
                          c.temperature, c.destiny, c.pain, c.tension,
                          c.alpha,
                          bm25(conversations_fts) as rank
                   FROM conversations_fts fts
                   JOIN conversations c ON c.id = fts.rowid
                   WHERE conversations_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            )
            rows = await cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "session_id": r["session_id"],
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "quality": r["quality"],
                    "access_count": r["access_count"],
                    "temperature": r["temperature"],
                    "destiny": r["destiny"],
                    "pain": r["pain"],
                    "tension": r["tension"],
                    "alpha": r["alpha"],
                    "rank": r["rank"],
                })
            return results
        except aiosqlite.OperationalError:
            return []

    # ═══════════════════════════════════════════════════════════════════════
    # RECALL — access conversation, bump access count
    # ═══════════════════════════════════════════════════════════════════════

    async def recall(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Recall a specific conversation, incrementing access count."""
        await self._conn.execute(
            "UPDATE conversations SET access_count = access_count + 1 WHERE id = ?",
            (conversation_id,),
        )
        await self._conn.commit()

        cursor = await self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # RECENT — get recent conversations
    # ═══════════════════════════════════════════════════════════════════════

    async def recent(self, limit: int = 10, session_only: bool = False) -> List[Dict[str, Any]]:
        """Get recent conversations, optionally limited to current session."""
        if session_only:
            cursor = await self._conn.execute(
                """SELECT * FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (self._session_id, limit),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )

        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]  # Chronological order

    # ═══════════════════════════════════════════════════════════════════════
    # SHARDS — autonomous graduation
    # ═══════════════════════════════════════════════════════════════════════

    async def find_shard_candidates(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find conversations that should graduate to training shards.

        Criteria:
        - quality >= 0.7 AND access_count >= 3
        - Not already a shard
        """
        cursor = await self._conn.execute(
            """SELECT c.* FROM conversations c
               LEFT JOIN shards s ON s.conversation_id = c.id
               WHERE s.id IS NULL
                 AND c.quality >= ?
                 AND c.access_count >= ?
               ORDER BY c.quality DESC, c.access_count DESC
               LIMIT ?""",
            (self.SHARD_MIN_QUALITY, self.SHARD_MIN_ACCESS, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def graduate_to_shard(
        self, conversation_id: int, shard_path: str, reason: str = "", priority: float = 0.0
    ) -> Optional[int]:
        """Record a conversation as graduated to shard."""
        try:
            cursor = await self._conn.execute(
                """INSERT INTO shards (conversation_id, shard_path, graduated_at, reason, priority)
                   VALUES (?, ?, ?, ?, ?)""",
                (conversation_id, shard_path, time.time(), reason, priority),
            )
            await self._conn.commit()
            return cursor.lastrowid
        except aiosqlite.IntegrityError:
            return None  # Already a shard

    async def get_training_queue(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get shards pending training."""
        cursor = await self._conn.execute(
            """SELECT s.*, c.prompt, c.response, c.quality
               FROM shards s
               JOIN conversations c ON c.id = s.conversation_id
               WHERE s.training_status = 'pending'
               ORDER BY s.priority DESC, s.graduated_at ASC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def mark_trained(self, shard_id: int, loss: Optional[float] = None):
        """Mark a shard as trained."""
        await self._conn.execute(
            "UPDATE shards SET training_status = 'trained', training_loss = ? WHERE id = ?",
            (loss, shard_id),
        )
        await self._conn.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # SEMANTIC SEARCH — cosine similarity over AMK state vectors
    # "Find conversations when I was in a similar state"
    # ═══════════════════════════════════════════════════════════════════════

    async def search_by_state(
        self,
        state: Dict[str, float],
        top_k: int = 5,
        min_quality: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find past conversations with similar AMK state (cosine similarity).

        state: dict with keys temperature, destiny, pain, tension, debt, alpha
        Returns conversations sorted by similarity (closest first).
        """
        query_vec = self._state_to_vector(state)

        cursor = await self._conn.execute(
            """SELECT * FROM conversations
               WHERE quality >= ?
               ORDER BY timestamp DESC
               LIMIT 1000""",
            (min_quality,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        scored = []
        for row in rows:
            row_dict = dict(row)
            row_vec = self._state_to_vector(row_dict)
            distance = _cosine_distance(query_vec, row_vec)
            row_dict["distance"] = distance
            scored.append((distance, row_dict))

        scored.sort(key=lambda x: x[0])
        return [item[1] for item in scored[:top_k]]

    @staticmethod
    def _state_to_vector(state: Dict[str, Any]) -> List[float]:
        """Convert AMK state dict to feature vector for similarity."""
        return [
            float(state.get("temperature", 0.0)),
            float(state.get("destiny", 0.0)),
            float(state.get("pain", 0.0)),
            float(state.get("tension", 0.0)),
            float(state.get("debt", 0.0)),
            float(state.get("alpha", 0.0)),
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════

    async def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        conv_count = (await (await self._conn.execute("SELECT COUNT(*) FROM conversations")).fetchone())[0]
        shard_count = (await (await self._conn.execute("SELECT COUNT(*) FROM shards")).fetchone())[0]
        session_count = (await (await self._conn.execute("SELECT COUNT(*) FROM sessions")).fetchone())[0]
        pending = (await (await self._conn.execute(
            "SELECT COUNT(*) FROM shards WHERE training_status = 'pending'"
        )).fetchone())[0]

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_conversations": conv_count,
            "total_shards": shard_count,
            "total_sessions": session_count,
            "pending_training": pending,
            "current_session": self._session_id,
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
        }


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance between two vectors (1 - cosine similarity). 0 = identical."""
    if len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)
