"""
LIMPHA SERVER — Unix socket daemon for Go↔Python IPC.

Go REPL spawns this daemon on startup.
Communication: JSON lines over Unix domain socket.

Protocol:
    → {"cmd": "store", "prompt": "...", "response": "...", "state": {...}}
    ← {"ok": true, "id": 42}

    → {"cmd": "search", "query": "consciousness", "limit": 5}
    ← {"ok": true, "results": [...]}

    → {"cmd": "recent", "limit": 10}
    ← {"ok": true, "conversations": [...]}

    → {"cmd": "candidates"}
    ← {"ok": true, "candidates": [...]}

    → {"cmd": "stats"}
    ← {"ok": true, ...stats...}

    → {"cmd": "shutdown"}
    ← {"ok": true}
"""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from .memory import LimphaMemory

# Default socket path
DEFAULT_SOCKET = str(Path.home() / ".yent" / "limpha.sock")


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    memory: LimphaMemory,
    shutdown_event: asyncio.Event,
):
    """Handle one client connection (the Go REPL)."""
    try:
        while not shutdown_event.is_set():
            line = await reader.readline()
            if not line:
                break

            try:
                msg = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError as e:
                response = {"ok": False, "error": f"invalid JSON: {e}"}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
                continue

            cmd = msg.get("cmd", "")
            response = await dispatch(cmd, msg, memory, shutdown_event)

            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()

            if cmd == "shutdown":
                break
    except asyncio.CancelledError:
        pass
    except ConnectionResetError:
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def dispatch(
    cmd: str,
    msg: dict,
    memory: LimphaMemory,
    shutdown_event: asyncio.Event,
) -> dict:
    """Dispatch a command to the appropriate handler."""

    if cmd == "store":
        try:
            conv_id = await memory.store(
                prompt=msg.get("prompt", ""),
                response=msg.get("response", ""),
                amk_state=msg.get("state", {}),
            )
            return {"ok": True, "id": conv_id}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif cmd == "search":
        try:
            results = await memory.search(
                query=msg.get("query", ""),
                limit=msg.get("limit", 10),
            )
            return {"ok": True, "results": results}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif cmd == "recent":
        try:
            convs = await memory.recent(
                limit=msg.get("limit", 10),
                session_only=msg.get("session_only", False),
            )
            return {"ok": True, "conversations": convs}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif cmd == "recall":
        try:
            conv = await memory.recall(msg.get("id", 0))
            if conv:
                return {"ok": True, "conversation": conv}
            return {"ok": False, "error": "not found"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif cmd == "search_state":
        try:
            results = await memory.search_by_state(
                state=msg.get("state", {}),
                top_k=msg.get("limit", 5),
                min_quality=msg.get("min_quality", 0.0),
            )
            return {"ok": True, "results": results}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif cmd == "candidates":
        try:
            candidates = await memory.find_shard_candidates(
                limit=msg.get("limit", 20),
            )
            return {"ok": True, "candidates": candidates}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif cmd == "stats":
        try:
            s = await memory.stats()
            return {"ok": True, **s}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif cmd == "shutdown":
        shutdown_event.set()
        return {"ok": True}

    elif cmd == "ping":
        return {"ok": True, "pong": True}

    else:
        return {"ok": False, "error": f"unknown command: {cmd}"}


async def run_server(
    socket_path: str = DEFAULT_SOCKET,
    db_path: Optional[str] = None,
):
    """Run the LIMPHA daemon."""
    # Clean up stale socket
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    shutdown_event = asyncio.Event()

    async with LimphaMemory(db_path) as memory:
        server = await asyncio.start_unix_server(
            lambda r, w: handle_client(r, w, memory, shutdown_event),
            path=socket_path,
        )

        # Set socket permissions (owner only)
        os.chmod(socket_path, 0o600)

        print(f"[limpha] daemon started — {socket_path}", flush=True)
        print(f"[limpha] db: {memory.db_path}", flush=True)
        print(f"[limpha] session: {memory._session_id}", flush=True)

        # Handle SIGTERM/SIGINT gracefully
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_event.set)

        # Wait for shutdown
        await shutdown_event.wait()

        print("[limpha] shutting down...", flush=True)
        server.close()
        await server.wait_closed()

    # Clean up socket
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    print("[limpha] stopped", flush=True)


def main():
    """Entry point: python3 -m limpha.server [--socket PATH] [--db PATH]"""
    socket_path = DEFAULT_SOCKET
    db_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--socket" and i + 1 < len(args):
            socket_path = args[i + 1]
            i += 2
        elif args[i] == "--db" and i + 1 < len(args):
            db_path = args[i + 1]
            i += 2
        else:
            i += 1

    asyncio.run(run_server(socket_path=socket_path, db_path=db_path))


if __name__ == "__main__":
    main()
