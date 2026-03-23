"""Hybrid search: FTS5 + vector → RRF fusion + time decay."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from math import pow

from sqlite_vec import serialize_float32

RRF_K = 60
TIME_DECAY_HALF_LIFE_DAYS = 30


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """FTS5 trigram キーワード検索"""
    rows = conn.execute(
        """SELECT m.id, m.user_message, m.assistant_message, m.project_path,
                  m.created_at, m.session_id
           FROM memories_fts f
           JOIN memories m ON f.rowid = m.id
           WHERE memories_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (f'"{query}"', limit),
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


def rrf_fusion(
    fts_results: list[dict], vec_results: list[dict], k: int = RRF_K
) -> list[dict]:
    """Reciprocal Rank Fusion で 2 つのランキングを統合する"""
    scores: dict[int, float] = {}
    metadata: dict[int, dict] = {}

    for r in fts_results:
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + r["rank"])
        metadata[rid] = r

    for r in vec_results:
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + r["rank"])
        if rid not in metadata:
            metadata[rid] = r

    results = []
    for rid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = {**metadata[rid], "score": score}
        results.append(entry)
    return results


def apply_time_decay(
    results: list[dict], half_life_days: int = TIME_DECAY_HALF_LIFE_DAYS
) -> list[dict]:
    """時間減衰を適用: decay = 0.5 ^ (days / half_life)"""
    now = datetime.now(timezone.utc)
    decayed = []
    for r in results:
        created_str = r.get("created_at", "")
        try:
            created = datetime.fromisoformat(created_str)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            created = now
        days_elapsed = max((now - created).total_seconds() / 86400, 0)
        decay = pow(0.5, days_elapsed / half_life_days)
        decayed.append({**r, "final_score": r.get("score", 0.0) * decay})
    decayed.sort(key=lambda x: x["final_score"], reverse=True)
    return decayed


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    limit: int = 5,
) -> list[dict]:
    """ハイブリッド検索: FTS5 + ベクトル → RRF → 時間減衰 → 上位N件"""
    fts_results = search_fts(conn, query, limit=50)
    vec_results = search_vec(conn, query_embedding, limit=50)
    fused = rrf_fusion(fts_results, vec_results)
    decayed = apply_time_decay(fused)
    return [
        {
            "user_message": r["user_message"],
            "assistant_message": r["assistant_message"],
            "project_path": r.get("project_path"),
            "created_at": r.get("created_at"),
            "score": round(r["final_score"], 4),
        }
        for r in decayed[:limit]
    ]
