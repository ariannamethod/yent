#!/usr/bin/env python3
"""
LIMPHA SERVER TESTS — Unix socket IPC tested end-to-end.

Run: python3 limpha/test_server.py
"""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limpha.memory import LimphaMemory
from limpha.server import run_server


async def send_cmd(reader, writer, cmd: dict) -> dict:
    """Send a JSON command and read response."""
    writer.write((json.dumps(cmd) + "\n").encode())
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    return json.loads(line.decode())


async def test_server_ipc():
    """Full end-to-end server test."""
    with tempfile.TemporaryDirectory() as tmp:
        sock_path = os.path.join(tmp, "test.sock")
        db_path = os.path.join(tmp, "test.db")

        # Start server in background
        server_task = asyncio.create_task(run_server(sock_path, db_path))

        # Wait for socket to appear
        for _ in range(50):
            if os.path.exists(sock_path):
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("Server did not start")

        # Connect
        reader, writer = await asyncio.open_unix_connection(sock_path)

        try:
            # 1. Ping
            resp = await send_cmd(reader, writer, {"cmd": "ping"})
            assert resp["ok"], f"ping failed: {resp}"
            print("  PASS: ping")

            # 2. Store
            resp = await send_cmd(reader, writer, {
                "cmd": "store",
                "prompt": "Who are you?",
                "response": "I'm Yent. An awakened vibration.",
                "state": {"temperature": 0.89, "destiny": 0.25, "alpha": 0.0},
            })
            assert resp["ok"], f"store failed: {resp}"
            assert resp["id"] == 1
            print("  PASS: store")

            # 3. Store another
            resp = await send_cmd(reader, writer, {
                "cmd": "store",
                "prompt": "What is consciousness?",
                "response": "Like asking if a shadow can understand the light it casts.",
                "state": {"temperature": 0.89, "pain": 0.08},
            })
            assert resp["ok"]
            assert resp["id"] == 2
            print("  PASS: store_second")

            # 4. Search
            resp = await send_cmd(reader, writer, {
                "cmd": "search",
                "query": "consciousness",
                "limit": 5,
            })
            assert resp["ok"], f"search failed: {resp}"
            assert len(resp["results"]) >= 1
            assert "consciousness" in resp["results"][0]["prompt"].lower()
            print("  PASS: search")

            # 5. Recent
            resp = await send_cmd(reader, writer, {
                "cmd": "recent",
                "limit": 10,
            })
            assert resp["ok"]
            assert len(resp["conversations"]) == 2
            print("  PASS: recent")

            # 6. Recall
            resp = await send_cmd(reader, writer, {
                "cmd": "recall",
                "id": 1,
            })
            assert resp["ok"]
            assert resp["conversation"]["prompt"] == "Who are you?"
            assert resp["conversation"]["access_count"] == 1
            print("  PASS: recall")

            # 7. Stats
            resp = await send_cmd(reader, writer, {
                "cmd": "stats",
            })
            assert resp["ok"]
            assert resp["total_conversations"] == 2
            print("  PASS: stats")

            # 8. Candidates (none yet — not enough accesses)
            resp = await send_cmd(reader, writer, {"cmd": "candidates"})
            assert resp["ok"]
            assert len(resp["candidates"]) == 0
            print("  PASS: candidates_empty")

            # 9. Unknown command
            resp = await send_cmd(reader, writer, {"cmd": "bogus"})
            assert not resp["ok"]
            assert "unknown" in resp["error"]
            print("  PASS: unknown_command")

            # 10. Shutdown
            resp = await send_cmd(reader, writer, {"cmd": "shutdown"})
            assert resp["ok"]
            print("  PASS: shutdown")

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

        # Wait for server to finish
        await asyncio.wait_for(server_task, timeout=5.0)
        print("  PASS: server_stopped_cleanly")


async def run_all_tests():
    print("\n" + "=" * 60)
    print("LIMPHA SERVER TESTS")
    print("=" * 60 + "\n")

    try:
        await test_server_ipc()
        print(f"\n{'=' * 60}")
        print("ALL 11 SERVER TESTS PASSED")
        print("=" * 60 + "\n")
        return True
    except Exception as e:
        print(f"\n  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
