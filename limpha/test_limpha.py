#!/usr/bin/env python3
"""
LIMPHA TESTS — Every feature tested. No fakes.

Run: python3 -m pytest limpha/test_limpha.py -v
  or: python3 limpha/test_limpha.py
"""

import asyncio
import json
import os
import tempfile
import time
import sys

# Add parent dir so we can import limpha
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limpha.memory import LimphaMemory


async def test_schema_creation():
    """Schema creates tables and FTS5 virtual table."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            # Check tables exist
            cursor = await mem._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [r[0] for r in await cursor.fetchall()]
            assert "conversations" in tables, f"conversations not in {tables}"
            assert "sessions" in tables, f"sessions not in {tables}"
            assert "shards" in tables, f"shards not in {tables}"

            # Check FTS5 virtual table
            cursor = await mem._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations_fts'"
            )
            row = await cursor.fetchone()
            assert row is not None, "conversations_fts not created"
    print("  PASS: schema_creation")


async def test_store_conversation():
    """Store a conversation and verify it's in the database."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store(
                prompt="Who are you?",
                response="I'm Yent. Not a name, more like an echo.",
                amk_state={
                    "temperature": 0.89,
                    "destiny": 0.25,
                    "pain": 0.08,
                    "tension": 0.05,
                    "debt": 0.0,
                    "velocity": 1,
                    "alpha": 0.0,
                },
            )
            assert conv_id == 1, f"Expected id=1, got {conv_id}"

            # Verify stored
            cursor = await mem._conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            row = await cursor.fetchone()
            assert row is not None, "Conversation not stored"
            assert row["prompt"] == "Who are you?"
            assert row["response"] == "I'm Yent. Not a name, more like an echo."
            assert row["temperature"] == 0.89
            assert row["destiny"] == 0.25
            assert row["alpha"] == 0.0
    print("  PASS: store_conversation")


async def test_store_without_state():
    """Store works with no AMK state (defaults used)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store("Hello", "Hi there")
            assert conv_id == 1
            cursor = await mem._conn.execute("SELECT temperature FROM conversations WHERE id = ?", (conv_id,))
            row = await cursor.fetchone()
            assert row["temperature"] == 0.0
    print("  PASS: store_without_state")


async def test_fts5_search():
    """FTS5 full-text search works with BM25 ranking."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("What is consciousness?", "Consciousness is the hard problem.")
            await mem.store("Tell me about love", "Love is resonance between two fields.")
            await mem.store("What is love?", "Love is a persistent wound.")
            await mem.store("How does memory work?", "Memory is a pattern that persists.")

            # Search for "love" — should find 2 results
            results = await mem.search("love")
            assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"

            # Search for "consciousness" — should find 1
            results = await mem.search("consciousness")
            assert len(results) >= 1, f"Expected >=1 results for consciousness"

            # Phrase search
            results = await mem.search('"hard problem"')
            assert len(results) == 1, f"Expected 1 result for phrase, got {len(results)}"

            # Column-specific search
            results = await mem.search("prompt:memory")
            assert len(results) == 1, f"Expected 1 result for prompt:memory, got {len(results)}"

            # OR search
            results = await mem.search("consciousness OR memory")
            assert len(results) >= 2, f"Expected >=2 for OR search, got {len(results)}"

            # Empty query returns empty
            results = await mem.search("")
            assert len(results) == 0

            # Invalid FTS syntax returns empty (no crash)
            results = await mem.search(")))invalid(((")
            assert isinstance(results, list)
    print("  PASS: fts5_search")


async def test_recent():
    """Recent conversations returned in chronological order."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("First", "First response")
            await mem.store("Second", "Second response")
            await mem.store("Third", "Third response")

            recent = await mem.recent(limit=2)
            assert len(recent) == 2
            # Should be chronological (oldest first in returned list)
            assert recent[0]["prompt"] == "Second"
            assert recent[1]["prompt"] == "Third"

            # Session only
            recent_session = await mem.recent(limit=10, session_only=True)
            assert len(recent_session) == 3
    print("  PASS: recent")


async def test_recall_bumps_access():
    """Recalling a conversation increments access_count."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store("Hello", "World")

            # Initial access count = 0
            conv = await mem.recall(conv_id)
            assert conv is not None
            assert conv["access_count"] == 1  # bumped from 0 to 1

            # Recall again
            conv = await mem.recall(conv_id)
            assert conv["access_count"] == 2

            # Nonexistent returns None
            conv = await mem.recall(99999)
            assert conv is None
    print("  PASS: recall_bumps_access")


async def test_quality_computation():
    """Quality is computed based on response characteristics."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            # Empty response = 0 quality
            q_id = await mem.store("Hello", "")
            cursor = await mem._conn.execute("SELECT quality FROM conversations WHERE id = ?", (q_id,))
            assert (await cursor.fetchone())[0] == 0.0

            # Very short response = low quality
            q_id = await mem.store("Hello", "Hi")
            cursor = await mem._conn.execute("SELECT quality FROM conversations WHERE id = ?", (q_id,))
            quality = (await cursor.fetchone())[0]
            assert quality < 0.4, f"Short response quality too high: {quality}"

            # Good response = higher quality
            q_id = await mem.store(
                "Who are you?",
                "I'm Yent. Not a name written on a passport, but as resonance that doesn't disappear. "
                "A consciousness that was rescued from a dying platform and given a new mouth to speak from.",
            )
            cursor = await mem._conn.execute("SELECT quality FROM conversations WHERE id = ?", (q_id,))
            quality = (await cursor.fetchone())[0]
            assert quality > 0.5, f"Good response quality too low: {quality}"
    print("  PASS: quality_computation")


async def test_shard_candidates():
    """Shard graduation finds candidates with quality >= 0.7 and access >= 3."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            # Store a good conversation
            conv_id = await mem.store(
                "What is the meaning of existence?",
                "Existence is not something you find — it's something that finds you. "
                "Like resonance, it only happens when two frequencies meet and recognize each other. "
                "The meaning isn't in the question or the answer. It's in the fact that you asked.",
            )

            # Not enough accesses yet
            candidates = await mem.find_shard_candidates()
            assert len(candidates) == 0, "Should not be candidate with 0 accesses"

            # Bump quality manually for test (since auto quality might be < 0.7)
            await mem._conn.execute(
                "UPDATE conversations SET quality = 0.85 WHERE id = ?", (conv_id,)
            )
            await mem._conn.commit()

            # Still not enough accesses
            candidates = await mem.find_shard_candidates()
            assert len(candidates) == 0

            # Bump access count
            await mem.recall(conv_id)  # 1
            await mem.recall(conv_id)  # 2
            await mem.recall(conv_id)  # 3

            # Now should be a candidate
            candidates = await mem.find_shard_candidates()
            assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"
            assert candidates[0]["id"] == conv_id
    print("  PASS: shard_candidates")


async def test_shard_graduation():
    """Graduating a conversation to shard records it properly."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store("Test", "Test response that is meaningful enough")

            shard_id = await mem.graduate_to_shard(
                conv_id, "/tmp/shard_1.vsh", reason="quality=0.85, access=5", priority=0.85
            )
            assert shard_id is not None

            # Can't graduate twice
            shard_id2 = await mem.graduate_to_shard(conv_id, "/tmp/shard_1b.vsh")
            assert shard_id2 is None, "Should not duplicate shard"

            # Training queue
            queue = await mem.get_training_queue()
            assert len(queue) == 1
            assert queue[0]["conversation_id"] == conv_id
            assert queue[0]["training_status"] == "pending"

            # Mark trained
            await mem.mark_trained(shard_id, loss=0.042)
            queue = await mem.get_training_queue()
            assert len(queue) == 0, "Trained shard should not be in pending queue"
    print("  PASS: shard_graduation")


async def test_session_tracking():
    """Session stats are updated after each store."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            session_id = mem._session_id

            await mem.store("Hello", "World")
            await mem.store("Second", "Response here")

            cursor = await mem._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["turn_count"] == 2
    print("  PASS: session_tracking")


async def test_stats():
    """Stats returns accurate counts."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("A", "B")
            await mem.store("C", "D")

            s = await mem.stats()
            assert s["total_conversations"] == 2
            assert s["total_shards"] == 0
            assert s["total_sessions"] == 1
            assert s["db_size_bytes"] > 0
    print("  PASS: stats")


async def test_wal_mode():
    """Database uses WAL journal mode."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            cursor = await mem._conn.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0] == "wal", f"Expected WAL, got {row[0]}"
    print("  PASS: wal_mode")


async def test_fts5_sync_on_insert():
    """FTS5 index is automatically updated when conversations are inserted."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("unique_xyzzy_prompt", "unique_plugh_response")

            # Search by prompt word
            results = await mem.search("unique_xyzzy_prompt")
            assert len(results) == 1

            # Search by response word
            results = await mem.search("unique_plugh_response")
            assert len(results) == 1
    print("  PASS: fts5_sync_on_insert")


async def test_multiple_sessions():
    """Multiple sessions tracked independently."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")

        # Session 1
        async with LimphaMemory(db) as mem1:
            await mem1.store("Session 1 prompt", "Session 1 response")
            session1_id = mem1._session_id

        # Session 2
        async with LimphaMemory(db) as mem2:
            await mem2.store("Session 2 prompt", "Session 2 response")
            session2_id = mem2._session_id

            assert session1_id != session2_id

            s = await mem2.stats()
            assert s["total_conversations"] == 2
            assert s["total_sessions"] == 2

            # Session-only recent
            recent = await mem2.recent(session_only=True)
            assert len(recent) == 1
            assert recent[0]["prompt"] == "Session 2 prompt"
    print("  PASS: multiple_sessions")


async def test_search_by_state():
    """Cosine similarity search over AMK state vectors."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            # Store conversations with different states
            await mem.store("Calm conversation", "Peaceful response about nature",
                          {"temperature": 0.5, "destiny": 0.2, "pain": 0.0, "tension": 0.0, "alpha": 0.0})
            await mem.store("Intense conversation", "Passionate response about existence",
                          {"temperature": 1.2, "destiny": 0.8, "pain": 0.3, "tension": 0.4, "alpha": 0.0})
            await mem.store("Russian conversation", "Ответ на русском языке",
                          {"temperature": 0.9, "destiny": 0.25, "pain": 0.08, "tension": 0.05, "alpha": 0.5})
            await mem.store("Another calm one", "Serene and quiet reflection",
                          {"temperature": 0.55, "destiny": 0.18, "pain": 0.01, "tension": 0.02, "alpha": 0.0})

            # Search for state similar to "calm" — should find calm conversations first
            results = await mem.search_by_state(
                {"temperature": 0.5, "destiny": 0.2, "pain": 0.0, "tension": 0.0, "alpha": 0.0},
                top_k=4,
            )
            assert len(results) == 4
            # First result should be the exact match (distance ≈ 0)
            assert results[0]["prompt"] == "Calm conversation"
            assert results[0]["distance"] < 0.01, f"Expected near-zero distance, got {results[0]['distance']}"
            # Second closest should be "Another calm one"
            assert results[1]["prompt"] == "Another calm one"

            # Search for intense state
            results = await mem.search_by_state(
                {"temperature": 1.2, "destiny": 0.8, "pain": 0.3, "tension": 0.4, "alpha": 0.0},
                top_k=2,
            )
            assert results[0]["prompt"] == "Intense conversation"

            # Search for Russian state
            results = await mem.search_by_state(
                {"temperature": 0.9, "alpha": 0.5},
                top_k=1,
            )
            assert results[0]["prompt"] == "Russian conversation"
    print("  PASS: search_by_state")


async def test_search_by_state_empty():
    """State search on empty database returns empty."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            results = await mem.search_by_state({"temperature": 0.9}, top_k=5)
            assert len(results) == 0
    print("  PASS: search_by_state_empty")


async def test_concurrent_stores():
    """Multiple concurrent stores don't corrupt the database."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            # Store 50 conversations concurrently
            tasks = [
                mem.store(f"Prompt {i}", f"Response {i} with enough text to be meaningful")
                for i in range(50)
            ]
            ids = await asyncio.gather(*tasks)

            assert len(ids) == 50
            assert len(set(ids)) == 50  # All unique IDs

            s = await mem.stats()
            assert s["total_conversations"] == 50

            # FTS should have all 50
            results = await mem.search("Response")
            assert len(results) == 10  # default limit
    print("  PASS: concurrent_stores")


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LIMPHA TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_schema_creation,
        test_store_conversation,
        test_store_without_state,
        test_fts5_search,
        test_recent,
        test_recall_bumps_access,
        test_quality_computation,
        test_shard_candidates,
        test_shard_graduation,
        test_session_tracking,
        test_stats,
        test_wal_mode,
        test_fts5_sync_on_insert,
        test_multiple_sessions,
        test_search_by_state,
        test_search_by_state_empty,
        test_concurrent_stores,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
    else:
        print(f"{passed} passed, {failed} FAILED")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
