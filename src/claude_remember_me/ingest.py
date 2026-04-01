"""Ingest CLI: reads Stop Hook stdin, parses transcript, saves to DB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from claude_remember_me.chunker import parse_transcript
from claude_remember_me.db import (
    DEFAULT_DB_PATH,
    get_connection,
    init_db,
    insert_memory,
    get_last_chunk_index,
    update_ingest_state,
)
from claude_remember_me.embedder import Embedder


def run_ingest(
    session_id: str,
    transcript_path: str,
    conn=None,
    embedder=None,
) -> int:
    """Transcript を読み込み、差分チャンクを DB に保存する。保存件数を返す。"""
    path = Path(transcript_path)
    if not path.exists():
        return 0

    jsonl_text = path.read_text(encoding="utf-8")
    pairs = parse_transcript(jsonl_text)
    if not pairs:
        return 0

    if conn is None:
        conn = get_connection()
        init_db(conn)
    if embedder is None:
        embedder = Embedder()

    last_index = get_last_chunk_index(conn, session_id)
    new_pairs = [p for p in pairs if p.chunk_index > last_index]
    if not new_pairs:
        return 0

    # バッチでベクトル化（user_message + assistant_message を結合してベクトル化）
    texts = [f"{p.user_message}\n{p.assistant_message}" for p in new_pairs]
    embeddings = embedder.embed_batch(texts)

    for pair, embedding in zip(new_pairs, embeddings):
        insert_memory(
            conn,
            session_id=session_id,
            chunk_index=pair.chunk_index,
            user_message=pair.user_message,
            assistant_message=pair.assistant_message,
            project_path=pair.project_path,
            embedding=embedding,
        )

    update_ingest_state(conn, session_id, new_pairs[-1].chunk_index)
    conn.commit()  # single commit for the whole batch
    return len(new_pairs)


def main():
    """Stop Hook のエントリポイント。stdin から JSON を読み取る。"""
    try:
        stdin_data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return

    session_id = stdin_data.get("session_id")
    transcript_path = stdin_data.get("transcript_path")

    if not session_id or not transcript_path:
        return

    saved = run_ingest(session_id, transcript_path)
    if saved > 0:
        print(f"claude-remember-me: saved {saved} new memories")


if __name__ == "__main__":
    main()
