# claude-remember-me

Long-term memory system for Claude Code. Remembers and retrieves past conversations across sessions.

## Features

- **Single-file storage** — All data in one SQLite file, no external services
- **Zero token overhead** — No LLM calls for saving memories
- **Auto-save** — Stop Hook saves new Q&A pairs incrementally (diff-only)
- **On-demand recall** — MCP tool `recall` searches memories anytime during conversation
- **Hybrid search** — FTS5 (trigram) + vector search (Ruri v3-310m) → RRF fusion + time decay

## Setup

```bash
cd ~/.claude/claude-remember-me
uv sync
```

## Design

See [docs/design.md](docs/design.md)
