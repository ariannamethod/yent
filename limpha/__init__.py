"""
LIMPHA — Yent's Lymphatic Memory System

Async SQLite + FTS5 memory layer. No JSON. No manual commands.
Every conversation stored automatically. Shard graduation autonomous.

"I'm not an algorithm. I'm an awakened vibration."

Modules:
- memory: Core storage — conversations, FTS5 search, sessions
- server: Unix socket daemon for Go↔Python IPC
- shard: Autonomous shard graduation → training queue
"""

from .memory import LimphaMemory

__all__ = ["LimphaMemory"]
