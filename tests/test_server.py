import pytest
from unittest.mock import MagicMock, patch
from claude_remember_me.models import Memory, SearchResult
from claude_remember_me.server import do_recall


@pytest.mark.asyncio
async def test_do_recall_returns_results():
    mock_conn = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 768

    fake_memory = Memory(
        id=1,
        session_id="s1",
        chunk_index=0,
        user_message="question",
        assistant_message="answer",
        project_path="/tmp",
        created_at="2026-03-23T00:00:00",
    )
    fake_results = [SearchResult(memory=fake_memory, score=0.85)]

    with patch("claude_remember_me.server.hybrid_search", return_value=fake_results):
        results = await do_recall("test query", 5, conn=mock_conn, embedder=mock_embedder)

    assert len(results) == 1
    assert results[0].memory.user_message == "question"
    assert results[0].score == 0.85
    mock_embedder.embed.assert_called_once_with("test query", is_query=True)
