"""Hybrid search: FTS5 + vector → RRF fusion + time decay."""

from __future__ import annotations

import sqlite3

from sqlite_vec import serialize_float32

from claude_remember_me.models import Memory, SearchResult
from claude_remember_me.ranking import apply_time_decay, rrf_fusion


def _sanitize_fts_query(query: str) -> str:
    """FTS5クエリ用にサニタイズする。ダブルクォートを除去しフレーズ検索として扱う"""
    sanitized = query.replace('"', "")
    if not sanitized.strip():
        return '""'
    return f'"{sanitized}"'


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """FTS5 trigram キーワード検索"""
    sanitized = _sanitize_fts_query(query)
    if sanitized == '""':
        return []
    rows = conn.execute(
        """SELECT m.id, m.user_message, m.assistant_message, m.project_path,
                  m.created_at, m.session_id
           FROM memories_fts f
           JOIN memories m ON f.rowid = m.id
           WHERE memories_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (sanitized, limit),
    ).fetchall()
    results = []
    for rank_idx, row in enumerate(rows):
        results.append({
            "id": row[0],
            "user_message": row[1],
            "assistant_message": row[2],
            "project_path": row[3],
            "created_at": row[4],
            "session_id": row[5],
            "rank": rank_idx + 1,
        })
    return results


def search_vec(
    conn: sqlite3.Connection, query_embedding: list[float], limit: int = 50
) -> list[dict]:
    """sqlite-vec ベクトル検索（cosine距離）"""
    rows = conn.execute(
        """SELECT v.id, v.distance, m.user_message, m.assistant_message,
                  m.project_path, m.created_at, m.session_id
           FROM memories_vec v
           JOIN memories m ON v.id = m.id
           WHERE v.embedding MATCH ?
           AND k = ?
           ORDER BY v.distance""",
        (serialize_float32(query_embedding), limit),
    ).fetchall()
    results = []
    for rank_idx, row in enumerate(rows):
        results.append({
            "id": row[0],
            "distance": row[1],
            "user_message": row[2],
            "assistant_message": row[3],
            "project_path": row[4],
            "created_at": row[5],
            "session_id": row[6],
            "rank": rank_idx + 1,
        })
    return results


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    limit: int = 5,
) -> list[SearchResult]:
    """ハイブリッド検索: FTS5 + ベクトル → RRF → 時間減衰 → 上位N件"""
    fts_results = search_fts(conn, query, limit=50)
    vec_results = search_vec(conn, query_embedding, limit=50)
    fused = rrf_fusion(fts_results, vec_results)
    decayed = apply_time_decay(fused)
    results = []
    for r in decayed[:limit]:
        memory = Memory(
            id=r["id"],
            session_id=r.get("session_id", ""),
            chunk_index=r.get("chunk_index", 0),
            user_message=r["user_message"],
            assistant_message=r["assistant_message"],
            project_path=r.get("project_path"),
            created_at=r.get("created_at", ""),
        )
        results.append(SearchResult(memory=memory, score=round(r["final_score"], 4)))
    return results
