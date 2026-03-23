import json
from pathlib import Path
from unittest.mock import patch

from claude_remember_me.ingest import run_ingest


def _make_transcript(tmp_path, pairs):
    """テスト用の transcript JSONL を作成"""
    transcript_path = tmp_path / "transcript.jsonl"
    lines = []
    for user_msg, asst_msg in pairs:
        lines.append(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": user_msg},
            "sessionId": "test-session",
            "cwd": "/tmp/project",
        }))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": asst_msg}]},
        }))
    transcript_path.write_text("\n".join(lines))
    return str(transcript_path)


def test_run_ingest_saves_new_chunks(tmp_path, db_conn):
    transcript_path = _make_transcript(tmp_path, [
        ("question 1", "answer 1"),
        ("question 2", "answer 2"),
    ])

    class FakeEmbedder:
        def embed_batch(self, texts, is_query=False):
            return [[0.1] * 768 for _ in texts]

    from claude_remember_me.db import init_db
    init_db(db_conn)

    run_ingest(
        session_id="test-session",
        transcript_path=transcript_path,
        conn=db_conn,
        embedder=FakeEmbedder(),
    )

    count = db_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 2


def test_run_ingest_skips_already_saved_chunks(tmp_path, db_conn):
    transcript_path = _make_transcript(tmp_path, [
        ("q1", "a1"),
        ("q2", "a2"),
        ("q3", "a3"),
    ])

    class FakeEmbedder:
        def embed_batch(self, texts, is_query=False):
            return [[0.1] * 768 for _ in texts]

    from claude_remember_me.db import init_db
    init_db(db_conn)

    # 1回目: 全3件保存
    run_ingest("test-session", transcript_path, db_conn, FakeEmbedder())
    assert db_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3

    # 2回目: 差分なし → 追加なし
    run_ingest("test-session", transcript_path, db_conn, FakeEmbedder())
    assert db_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3
