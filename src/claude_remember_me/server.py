"""MCP server exposing the recall tool."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from claude_remember_me.db import get_connection, init_db
from claude_remember_me.embedder import Embedder
from claude_remember_me.models import SearchResult
from claude_remember_me.search import hybrid_search

mcp = FastMCP("claude-remember-me")

_conn = None
_embedder = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = get_connection()
        init_db(_conn)
    return _conn


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


async def do_recall(query: str, limit: int = 5, *, conn=None, embedder=None) -> list[SearchResult]:
    """recall のコアロジック（テスト可能にするため分離）"""
    if conn is None:
        conn = _get_conn()
    if embedder is None:
        embedder = _get_embedder()

    query_embedding = embedder.embed(query, is_query=True)
    return hybrid_search(conn, query, query_embedding, limit=limit)


@mcp.tool()
async def recall(query: str, limit: int = 5) -> str:
    """Search past conversations for relevant memories.

    Args:
        query: Search query to find related past conversations
        limit: Maximum number of results to return (default: 5)
    """
    results = await do_recall(query, limit)
    if not results:
        return "No relevant memories found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"--- Memory {i} (score: {r.score}) ---\n"
            f"Project: {r.memory.project_path or 'unknown'}\n"
            f"Date: {r.memory.created_at or 'unknown'}\n"
            f"User: {r.memory.user_message[:500]}\n"
            f"Assistant: {r.memory.assistant_message[:500]}"
        )
    return "\n\n".join(parts)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
