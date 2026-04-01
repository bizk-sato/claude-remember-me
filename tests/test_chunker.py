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


def test_skips_meta_messages():
    """isMeta: True のメッセージ（スキル展開等）はスキップする"""
    jsonl = _make_jsonl(
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
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "Base directory for this skill: /path/to/skill\n# Brainstorming",
            },
            "isMeta": True,
            "userType": "external",
            "sessionId": "s1",
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "skill response"}],
            },
        },
    )
    pairs = parse_transcript(jsonl)
    assert len(pairs) == 1
    assert pairs[0].user_message == "real question"


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
