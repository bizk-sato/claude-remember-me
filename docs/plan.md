# claude-remember-me Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude Code 用の長期記憶システムを構築する。セッション横断で会話を記憶・検索できるようにする。

**Architecture:** Stop Hook で transcript を Q&A チャンクに分割・ベクトル化して SQLite に差分保存。MCP サーバーの `recall` ツールでハイブリッド検索（FTS5 + ベクトル → RRF + 時間減衰）を提供。

**Tech Stack:** Python 3.11+, uv, sentence-transformers (Ruri v3-310m), sqlite-vec, mcp[cli] (FastMCP)

**Spec:** [docs/design.md](design.md)

---

## File Structure

```
~/.claude/claude-remember-me/
├── pyproject.toml
├── src/
│   └── claude_remember_me/
│       ├── __init__.py
│       ├── db.py           # SQLite + sqlite-vec DB管理、スキーマ作成
│       ├── embedder.py     # Ruri v3-310m でテキストをベクトル化
│       ├── chunker.py      # JSONL transcript を Q&A ペアに分割
│       ├── ingest.py       # CLI エントリポイント: stdin → chunker → embedder → db
│       ├── search.py       # ハイブリッド検索: FTS5 + ベクトル → RRF + 時間減衰
│       └── server.py       # MCP サーバー: recall ツールを公開
└── tests/
    ├── conftest.py         # 共通 fixture（in-memory DB、テスト用 embedder）
    ├── test_db.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_ingest.py
    ├── test_search.py
    └── test_server.py
```

---

## Transcript JSONL Format（参考）

Claude Code の transcript は JSONL 形式。各行の JSON オブジェクトの主要フィールド：

```jsonl
{"type": "user", "message": {"role": "user", "content": "ユーザーのメッセージ"}, "sessionId": "...", "cwd": "..."}
{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Claudeの応答"}, {"type": "thinking", "thinking": "..."}]}}
```

- `type: "user"` の `message.content` は文字列またはコンテンツブロックのリスト
- `type: "assistant"` の `message.content` はリスト。`type: "text"` のブロックからテキストを抽出
- `type: "progress"`, `type: "file-history-snapshot"` 等は無視する

---

### Task 1: プロジェクト初期化（pyproject.toml + パッケージ構造）

**Files:**
- Create: `pyproject.toml`
- Create: `src/claude_remember_me/__init__.py`

- [ ] **Step 1: pyproject.toml を作成**

```toml
[project]
name = "claude-remember-me"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sentence-transformers>=3.0",
    "sqlite-vec>=0.1.6",
    "mcp[cli]>=1.25,<2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/claude_remember_me"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: __init__.py を作成**

```python
"""claude-remember-me: Long-term memory for Claude Code."""
```

- [ ] **Step 3: tests ディレクトリを作成**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: uv sync を実行**

```bash
cd ~/.claude/claude-remember-me && uv sync --all-extras
```

Expected: 依存パッケージがインストールされ .venv が作成される

- [ ] **Step 5: コミット**

```bash
git add -A
git commit -m "feat: initialize project with pyproject.toml and package structure"
```

---

### Task 2: db.py — SQLite + sqlite-vec データベース管理

**Files:**
- Create: `src/claude_remember_me/db.py`
- Create: `tests/conftest.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: tests/conftest.py に共通 fixture を作成**

```python
import pytest
import sqlite3
import sqlite_vec


@pytest.fixture
def db_conn(tmp_path):
    """テスト用の一時 SQLite DB を返す"""
    db_path = tmp_path / "test_memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    yield conn
    conn.close()
```

- [ ] **Step 2: tests/test_db.py にテストを書く**

```python
from claude_remember_me.db import init_db, insert_memory, get_last_chunk_index, update_ingest_state


def test_init_db_creates_tables(db_conn):
    init_db(db_conn)
    cursor = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert "memories" in tables
    assert "memories_fts" in tables
    assert "memories_vec" in tables
    assert "ingest_state" in tables


def test_insert_memory(db_conn):
    init_db(db_conn)
    embedding = [0.1] * 768
    insert_memory(
        db_conn,
        session_id="sess-1",
        chunk_index=0,
        user_message="hello",
        assistant_message="hi there",
        project_path="/tmp/project",
        embedding=embedding,
    )
    row = db_conn.execute("SELECT * FROM memories WHERE session_id='sess-1'").fetchone()
    assert row is not None


def test_insert_duplicate_is_ignored(db_conn):
    init_db(db_conn)
    embedding = [0.1] * 768
    for _ in range(2):
        insert_memory(
            db_conn,
            session_id="sess-1",
            chunk_index=0,
            user_message="hello",
            assistant_message="hi",
            project_path=None,
            embedding=embedding,
        )
    count = db_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 1


def test_get_last_chunk_index_returns_negative_one_for_new_session(db_conn):
    init_db(db_conn)
    assert get_last_chunk_index(db_conn, "new-session") == -1


def test_update_and_get_ingest_state(db_conn):
    init_db(db_conn)
    update_ingest_state(db_conn, "sess-1", 5)
    assert get_last_chunk_index(db_conn, "sess-1") == 5
    update_ingest_state(db_conn, "sess-1", 10)
    assert get_last_chunk_index(db_conn, "sess-1") == 10
```

- [ ] **Step 3: テストを実行して FAIL を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_db.py -v
```

Expected: ImportError (モジュールが存在しない)

- [ ] **Step 4: src/claude_remember_me/db.py を実装**

```python
"""SQLite + sqlite-vec database management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

DEFAULT_DB_PATH = Path.home() / ".claude" / "claude-remember-me" / "data" / "memory.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    project_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    user_message, assistant_message,
    content=memories, content_rowid=id,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, user_message, assistant_message)
    VALUES (new.id, new.user_message, new.assistant_message);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, user_message, assistant_message)
    VALUES ('delete', old.id, old.user_message, old.assistant_message);
END;

CREATE TABLE IF NOT EXISTS ingest_state (
    session_id TEXT PRIMARY KEY,
    last_chunk_index INTEGER NOT NULL
);
"""

VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine
);
"""


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute(VEC_TABLE_SQL)
    conn.commit()


def insert_memory(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    chunk_index: int,
    user_message: str,
    assistant_message: str,
    project_path: str | None,
    embedding: list[float],
) -> None:
    cursor = conn.execute(
        """INSERT OR IGNORE INTO memories
           (session_id, chunk_index, user_message, assistant_message, project_path)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, chunk_index, user_message, assistant_message, project_path),
    )
    if cursor.rowcount > 0:
        row_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            (row_id, serialize_float32(embedding)),
        )
    conn.commit()


def get_last_chunk_index(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT last_chunk_index FROM ingest_state WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row[0] if row else -1


def update_ingest_state(conn: sqlite3.Connection, session_id: str, last_chunk_index: int) -> None:
    conn.execute(
        """INSERT INTO ingest_state (session_id, last_chunk_index)
           VALUES (?, ?)
           ON CONFLICT(session_id) DO UPDATE SET last_chunk_index = excluded.last_chunk_index""",
        (session_id, last_chunk_index),
    )
    conn.commit()
```

- [ ] **Step 5: テストを実行して PASS を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_db.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "feat: add db module with SQLite + sqlite-vec schema and CRUD"
```

---

### Task 3: embedder.py — Ruri v3-310m ベクトル化

**Files:**
- Create: `src/claude_remember_me/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: tests/test_embedder.py にテストを書く**

```python
from claude_remember_me.embedder import Embedder


def test_embed_returns_768_dim_vector():
    embedder = Embedder()
    result = embedder.embed("テストの文章です")
    assert len(result) == 768
    assert isinstance(result[0], float)


def test_embed_batch():
    embedder = Embedder()
    results = embedder.embed_batch(["こんにちは", "さようなら"])
    assert len(results) == 2
    assert all(len(v) == 768 for v in results)


def test_similar_texts_have_higher_similarity():
    embedder = Embedder()
    v1 = embedder.embed("Pythonのプログラミング")
    v2 = embedder.embed("Pythonでコードを書く")
    v3 = embedder.embed("今日の天気は晴れです")
    # cosine similarity
    import numpy as np
    sim_12 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    sim_13 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    assert sim_12 > sim_13
```

- [ ] **Step 2: テストを実行して FAIL を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_embedder.py -v
```

Expected: ImportError

- [ ] **Step 3: src/claude_remember_me/embedder.py を実装**

```python
"""Text embedding using Ruri v3-310m."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "cl-nagoya/ruri-v3-310m"
QUERY_PREFIX = "検索クエリ: "
PASSAGE_PREFIX = "文章: "


class Embedder:
    def __init__(self) -> None:
        self._model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    def embed(self, text: str, is_query: bool = False) -> list[float]:
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        vector = self._model.encode(prefix + text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
```

注意: Ruri v3 はクエリと文章で異なるプレフィックスを使う。保存時は `"文章: "`, 検索時は `"検索クエリ: "`。

- [ ] **Step 4: テストを実行して PASS を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_embedder.py -v
```

Expected: All 3 tests PASS（初回はモデルダウンロードで数分かかる）

- [ ] **Step 5: コミット**

```bash
git add -A
git commit -m "feat: add embedder module with Ruri v3-310m"
```

---

### Task 4: chunker.py — Transcript を Q&A ペアに分割

**Files:**
- Create: `src/claude_remember_me/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: tests/test_chunker.py にテストを書く**

```python
import json

from claude_remember_me.chunker import parse_transcript, QAPair


def _make_jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(e) for e in entries)


def test_parse_simple_qa_pair():
    jsonl = _make_jsonl(
        {
            "type": "user",
            "message": {"role": "user", "content": "hello"},
            "sessionId": "sess-1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi there!"}],
            },
        },
    )
    pairs = parse_transcript(jsonl)
    assert len(pairs) == 1
    assert pairs[0].user_message == "hello"
    assert pairs[0].assistant_message == "hi there!"
    assert pairs[0].session_id == "sess-1"
    assert pairs[0].project_path == "/tmp"


def test_parse_user_content_as_list():
    jsonl = _make_jsonl(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "list content"}],
            },
            "sessionId": "sess-1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "response"}],
            },
        },
    )
    pairs = parse_transcript(jsonl)
    assert len(pairs) == 1
    assert pairs[0].user_message == "list content"


def test_skips_non_user_assistant_types():
    jsonl = _make_jsonl(
        {"type": "progress", "data": {"type": "hook_progress"}},
        {"type": "file-history-snapshot", "snapshot": {}},
        {
            "type": "user",
            "message": {"role": "user", "content": "real question"},
            "sessionId": "s1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "real answer"}],
            },
        },
    )
    pairs = parse_transcript(jsonl)
    assert len(pairs) == 1
    assert pairs[0].user_message == "real question"


def test_multiple_assistant_responses_concatenated():
    """複数の assistant メッセージが連続する場合、テキストを結合する"""
    jsonl = _make_jsonl(
        {
            "type": "user",
            "message": {"role": "user", "content": "question"},
            "sessionId": "s1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "part 1"}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "part 2"}],
            },
        },
    )
    pairs = parse_transcript(jsonl)
    assert len(pairs) == 1
    assert "part 1" in pairs[0].assistant_message
    assert "part 2" in pairs[0].assistant_message


def test_skips_empty_messages():
    jsonl = _make_jsonl(
        {
            "type": "user",
            "message": {"role": "user", "content": ""},
            "sessionId": "s1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": []},
        },
    )
    pairs = parse_transcript(jsonl)
    assert len(pairs) == 0


def test_multiple_qa_pairs():
    jsonl = _make_jsonl(
        {
            "type": "user",
            "message": {"role": "user", "content": "q1"},
            "sessionId": "s1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "a1"}],
            },
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "q2"},
            "sessionId": "s1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "a2"}],
            },
        },
    )
    pairs = parse_transcript(jsonl)
    assert len(pairs) == 2
    assert pairs[0].user_message == "q1"
    assert pairs[1].user_message == "q2"
```

- [ ] **Step 2: テストを実行して FAIL を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_chunker.py -v
```

- [ ] **Step 3: src/claude_remember_me/chunker.py を実装**

```python
"""Parse Claude Code JSONL transcripts into Q&A pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class QAPair:
    user_message: str
    assistant_message: str
    session_id: str
    project_path: str | None
    chunk_index: int


def _extract_text(content) -> str:
    """message.content からテキストを抽出する"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


def parse_transcript(jsonl_text: str) -> list[QAPair]:
    """JSONL テキストを Q&A ペアのリストに変換する"""
    pairs: list[QAPair] = []
    current_user: str | None = None
    current_session_id: str | None = None
    current_cwd: str | None = None
    assistant_parts: list[str] = []

    def _flush():
        nonlocal current_user, assistant_parts
        if current_user and assistant_parts:
            pairs.append(
                QAPair(
                    user_message=current_user,
                    assistant_message="\n\n".join(assistant_parts),
                    session_id=current_session_id or "",
                    project_path=current_cwd,
                    chunk_index=len(pairs),
                )
            )
        current_user = None
        assistant_parts = []

    for line in jsonl_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")
        message = obj.get("message", {})
        role = message.get("role")

        if msg_type == "user" and role == "user":
            text = _extract_text(message.get("content", ""))
            if not text:
                continue
            # 新しいユーザーメッセージが来たら、前の Q&A ペアを flush
            _flush()
            current_user = text
            current_session_id = obj.get("sessionId", current_session_id)
            current_cwd = obj.get("cwd", current_cwd)
            assistant_parts = []

        elif msg_type == "assistant" and role == "assistant":
            text = _extract_text(message.get("content", ""))
            if text and current_user:
                assistant_parts.append(text)

    # 最後の Q&A ペアを flush
    _flush()
    return pairs
```

- [ ] **Step 4: テストを実行して PASS を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_chunker.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: コミット**

```bash
git add -A
git commit -m "feat: add chunker to parse JSONL transcripts into Q&A pairs"
```

---

### Task 5: search.py — ハイブリッド検索（FTS5 + ベクトル + RRF + 時間減衰）

**Files:**
- Create: `src/claude_remember_me/search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: tests/conftest.py に embedder fixture を追加**

```python
# conftest.py に追加
@pytest.fixture
def populated_db(db_conn):
    """テストデータ入りの DB を返す"""
    from claude_remember_me.db import init_db, insert_memory

    init_db(db_conn)
    test_data = [
        ("s1", 0, "Pythonの型ヒントについて教えて", "Pythonの型ヒントはmypyで検証できます", "/project-a"),
        ("s1", 1, "Rustのライフタイムとは？", "Rustのライフタイムはメモリ安全性を保証する仕組みです", "/project-a"),
        ("s2", 0, "SQLiteのFTS5の使い方", "FTS5はSQLiteの全文検索拡張です", "/project-b"),
        ("s2", 1, "今日のランチは何にする？", "カレーがおすすめです", "/project-b"),
    ]
    for sid, idx, user_msg, asst_msg, path in test_data:
        # テスト用にダミーの embedding を使う（検索テストでは実際の embedder は使わない）
        embedding = [0.0] * 768
        # 簡易的にテキストのハッシュから embedding の一部を変える
        for i, ch in enumerate(user_msg[:10]):
            embedding[i % 768] = ord(ch) / 10000.0
        insert_memory(
            db_conn,
            session_id=sid,
            chunk_index=idx,
            user_message=user_msg,
            assistant_message=asst_msg,
            project_path=path,
            embedding=embedding,
        )
    return db_conn
```

- [ ] **Step 2: tests/test_search.py にテストを書く**

```python
from claude_remember_me.search import search_fts, search_vec, hybrid_search


def test_fts_search_finds_matching_keyword(populated_db):
    results = search_fts(populated_db, "Python", limit=10)
    assert len(results) > 0
    assert any("Python" in r["user_message"] for r in results)


def test_fts_search_returns_empty_for_no_match(populated_db):
    results = search_fts(populated_db, "xyznonexistent", limit=10)
    assert len(results) == 0


def test_vec_search_returns_results(populated_db):
    query_embedding = [0.0] * 768
    results = search_vec(populated_db, query_embedding, limit=10)
    assert len(results) > 0


def test_hybrid_search_returns_scored_results(populated_db):
    # hybrid_search は embedder を受け取る版は別途テストするので、
    # ここでは内部の rrf + time_decay ロジックをテスト
    from claude_remember_me.search import rrf_fusion, apply_time_decay
    from datetime import datetime, timedelta

    fts_results = [
        {"id": 1, "rank": 1},
        {"id": 2, "rank": 2},
    ]
    vec_results = [
        {"id": 2, "rank": 1},
        {"id": 3, "rank": 2},
    ]
    fused = rrf_fusion(fts_results, vec_results, k=60)
    assert len(fused) == 3  # ids: 1, 2, 3
    # id=2 は両方に出現するのでスコアが最も高い
    assert fused[0]["id"] == 2

    # time decay
    now = datetime.now()
    results = [
        {"id": 1, "score": 1.0, "created_at": now.isoformat()},
        {"id": 2, "score": 1.0, "created_at": (now - timedelta(days=30)).isoformat()},
        {"id": 3, "score": 1.0, "created_at": (now - timedelta(days=60)).isoformat()},
    ]
    decayed = apply_time_decay(results, half_life_days=30)
    assert decayed[0]["final_score"] > decayed[1]["final_score"]
    assert decayed[1]["final_score"] > decayed[2]["final_score"]
```

- [ ] **Step 3: テストを実行して FAIL を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_search.py -v
```

- [ ] **Step 4: src/claude_remember_me/search.py を実装**

```python
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
```

- [ ] **Step 5: テストを実行して PASS を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_search.py -v
```

Expected: All tests PASS

- [ ] **Step 6: コミット**

```bash
git add -A
git commit -m "feat: add hybrid search with FTS5 + vector + RRF + time decay"
```

---

### Task 6: ingest.py — Stop Hook CLI エントリポイント

**Files:**
- Create: `src/claude_remember_me/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: tests/test_ingest.py にテストを書く**

```python
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

    # ダミー embedder
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
```

- [ ] **Step 2: テストを実行して FAIL を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_ingest.py -v
```

- [ ] **Step 3: src/claude_remember_me/ingest.py を実装**

```python
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
        print(f"claude-remember-me: saved {saved} new memories", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストを実行して PASS を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_ingest.py -v
```

Expected: All tests PASS

- [ ] **Step 5: コミット**

```bash
git add -A
git commit -m "feat: add ingest CLI for Stop Hook transcript processing"
```

---

### Task 7: server.py — MCP サーバー（recall ツール）

**Files:**
- Create: `src/claude_remember_me/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: tests/test_server.py にテストを書く**

```python
import pytest
from unittest.mock import MagicMock, patch
from claude_remember_me.server import do_recall


@pytest.mark.asyncio
async def test_do_recall_returns_results():
    mock_conn = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 768

    fake_results = [
        {
            "user_message": "question",
            "assistant_message": "answer",
            "project_path": "/tmp",
            "created_at": "2026-03-23T00:00:00",
            "score": 0.85,
        }
    ]

    with patch("claude_remember_me.server.hybrid_search", return_value=fake_results):
        results = await do_recall("test query", 5, conn=mock_conn, embedder=mock_embedder)

    assert len(results) == 1
    assert results[0]["user_message"] == "question"
    mock_embedder.embed.assert_called_once_with("test query", is_query=True)
```

- [ ] **Step 2: テストを実行して FAIL を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_server.py -v
```

- [ ] **Step 3: src/claude_remember_me/server.py を実装**

```python
"""MCP server exposing the recall tool."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from claude_remember_me.db import get_connection, init_db
from claude_remember_me.embedder import Embedder
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


async def do_recall(query: str, limit: int = 5, *, conn=None, embedder=None) -> list[dict]:
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
            f"--- Memory {i} (score: {r['score']}) ---\n"
            f"Project: {r.get('project_path', 'unknown')}\n"
            f"Date: {r.get('created_at', 'unknown')}\n"
            f"User: {r['user_message'][:500]}\n"
            f"Assistant: {r['assistant_message'][:500]}"
        )
    return "\n\n".join(parts)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストを実行して PASS を確認**

```bash
cd ~/.claude/claude-remember-me && uv run pytest tests/test_server.py -v
```

Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add -A
git commit -m "feat: add MCP server with recall tool"
```

---

### Task 8: Claude Code 設定に Hook と MCP サーバーを登録

**Files:**
- Modify: `~/.claude/settings.json`

- [ ] **Step 1: ~/.claude/settings.json に Stop Hook を追加**

`hooks.Stop` 配列に以下を追加：

```json
{
  "matcher": "",
  "hooks": [
    {
      "type": "command",
      "command": "uv run --project ~/.claude/claude-remember-me python -m claude_remember_me.ingest",
      "timeout": 30000
    }
  ]
}
```

- [ ] **Step 2: ~/.claude/settings.json に MCP サーバーを登録**

`mcpServers` に以下を追加：

```json
{
  "claude-remember-me": {
    "command": "uv",
    "args": [
      "run",
      "--project",
      "/Users/kazukisato/.claude/claude-remember-me",
      "python",
      "-m",
      "claude_remember_me.server"
    ]
  }
}
```

- [ ] **Step 3: コミット（claude-remember-me リポジトリ）**

```bash
cd ~/.claude/claude-remember-me && git add -A && git commit -m "feat: complete initial implementation" && git push
```

---

### Task 9: 動作確認

- [ ] **Step 1: MCP サーバーの起動テスト**

```bash
cd ~/.claude/claude-remember-me && echo '{}' | uv run python -m claude_remember_me.server
```

MCP サーバーが stdio で起動することを確認（Ctrl+C で停止）

- [ ] **Step 2: ingest の手動テスト**

```bash
cd ~/.claude/claude-remember-me
echo '{"session_id": "manual-test", "transcript_path": "<実際の transcript パス>"}' | uv run python -m claude_remember_me.ingest
```

stderr に `saved N new memories` と表示されることを確認

- [ ] **Step 3: Claude Code を再起動して recall ツールが使えることを確認**

新しい Claude Code セッションを開始し、`recall` ツールが利用可能であることを確認
