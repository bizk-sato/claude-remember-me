"""Domain models for claude-remember-me."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Memory:
    id: int
    session_id: str
    chunk_index: int
    user_message: str
    assistant_message: str
    project_path: str | None
    created_at: str


@dataclass
class SearchResult:
    memory: Memory
    score: float
