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


# task-notification等のシステム生成メッセージをスキップするパターン
_NOISE_PREFIXES = (
    "<task-notification>",
    "<system-reminder>",
)


def _is_noise(text: str) -> bool:
    """システム生成のノイズメッセージかどうか判定する"""
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in _NOISE_PREFIXES)


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
            # Skip system-like messages (skill prompts, hook outputs, etc.)
            user_type = obj.get("userType")
            if user_type is not None and user_type != "external":
                continue
            # Skip meta messages (skill expansions, system context injections)
            if obj.get("isMeta"):
                continue
            # ノイズメッセージはスキップ
            if _is_noise(text):
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
