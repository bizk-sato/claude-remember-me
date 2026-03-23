# claude-remember-me Design Spec

## Overview

Claude Code 用の長期記憶システム。セッションを横断して過去の会話を記憶・検索する。

## Design Principles

1. **外部サービス不依存** — SQLite 単一ファイルで全データ格納
2. **バックグラウンドのトークン消費なし** — LLM 不使用で記憶保存
3. **自動保存** — Stop Hook で毎ターン差分保存

## Architecture

```
~/.claude/claude-remember-me/
├── pyproject.toml
├── src/
│   └── claude_remember_me/
│       ├── __init__.py
│       ├── server.py       # MCP サーバー（recall ツール）
│       ├── ingest.py       # CLI: transcript → Q&A チャンク → SQLite 保存
│       ├── chunker.py      # transcript を Q&A ペアに分割
│       ├── embedder.py     # Ruri v3-310m でベクトル化
│       ├── search.py       # ハイブリッド検索（FTS5 + ベクトル + RRF）
│       └── db.py           # SQLite + sqlite-vec 管理
├── data/
│   └── memory.db           # 単一ファイルDB
└── docs/
    └── design.md           # この文書
```

## Data Flow

### Save（Stop Hook → ingest）

```
Claude stops → Stop Hook fires
  → stdin から JSON 受信（session_id, transcript_path）
  → transcript を読み込み
  → Q&A ペア（ユーザー発話 + Claude応答）に分割
  → ingest_state から前回保存位置を確認（差分管理）
  → 新しいチャンクのみ:
    → Ruri v3-310m でベクトル化
    → SQLite に保存（memories + memories_fts + memories_vec）
    → ingest_state を更新
```

### Recall（MCP ツール）

```
Claude calls recall(query, limit=5)
  → FTS5 キーワード検索（trigram）→ Top50
  → ベクトル検索（Ruri v3-310m cosine類似度）→ Top50
  → RRF（Reciprocal Rank Fusion, k=60）で統合
  → 時間減衰適用: decay = 0.5 ^ (days / 30)
  → final_score = rrf_score * decay
  → 上位N件を返す
```

## Data Model

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    project_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, chunk_index)
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    user_message, assistant_message,
    content=memories, content_rowid=id,
    tokenize='trigram'
);

CREATE VIRTUAL TABLE memories_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);

CREATE TABLE ingest_state (
    session_id TEXT PRIMARY KEY,
    last_chunk_index INTEGER NOT NULL
);
```

## MCP Tool Definition

```python
@server.tool()
async def recall(query: str, limit: int = 5) -> list[dict]:
    """過去の会話から関連する記憶を検索する"""
```

Returns:
```json
[
  {
    "user_message": "...",
    "assistant_message": "...",
    "project_path": "...",
    "created_at": "2026-03-23T...",
    "score": 0.85
  }
]
```

## Integration

### Stop Hook（~/.claude/settings.json）

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "uv run --project ~/.claude/claude-remember-me python -m claude_remember_me.ingest"
          }
        ]
      }
    ]
  }
}
```

### MCP Server（~/.claude/settings.json）

```json
{
  "mcpServers": {
    "claude-remember-me": {
      "command": "uv",
      "args": ["run", "--project", "~/.claude/claude-remember-me", "python", "-m", "claude_remember_me.server"]
    }
  }
}
```

## Dependencies

```toml
[project]
name = "claude-remember-me"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sentence-transformers",
    "sqlite-vec",
    "mcp[cli]",
]
```

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding model | cl-nagoya/ruri-v3-310m | JMTEB 77.2, 日本語最強クラス |
| Embedding dim | 768 | Ruri v3-310m の出力次元 |
| FTS tokenizer | trigram | 日本語に分かち書き不要 |
| RRF k | 60 | 一般的なデフォルト値 |
| Time decay half-life | 30 days | 最近の記憶を優先 |
| Default recall limit | 5 | コンテキスト圧迫を防ぐ |
| Max sequence length | 8,192 tokens | Ruri v3 の上限 |
