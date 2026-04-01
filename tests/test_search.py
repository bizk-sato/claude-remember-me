from claude_remember_me.search import search_fts, search_vec, hybrid_search, _sanitize_fts_query


def test_sanitize_fts_query_strips_double_quotes():
    assert _sanitize_fts_query('foo"bar') == '"foobar"'
    assert _sanitize_fts_query('""') == '""'
    assert _sanitize_fts_query("") == '""'
    assert _sanitize_fts_query("   ") == '""'
    assert _sanitize_fts_query("normal query") == '"normal query"'


def test_fts_search_returns_empty_for_empty_query(populated_db):
    assert search_fts(populated_db, "", limit=10) == []
    assert search_fts(populated_db, '""', limit=10) == []
    assert search_fts(populated_db, "   ", limit=10) == []


def test_fts_search_with_double_quotes_in_query(populated_db):
    """ダブルクォートを含むクエリでクラッシュしないことを確認"""
    results = search_fts(populated_db, 'Python"injection', limit=10)
    assert isinstance(results, list)


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
    from claude_remember_me.ranking import apply_time_decay, rrf_fusion
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
    assert len(fused) == 3
    assert fused[0]["id"] == 2

    now = datetime.now()
    results = [
        {"id": 1, "score": 1.0, "created_at": now.isoformat()},
        {"id": 2, "score": 1.0, "created_at": (now - timedelta(days=30)).isoformat()},
        {"id": 3, "score": 1.0, "created_at": (now - timedelta(days=60)).isoformat()},
    ]
    decayed = apply_time_decay(results, half_life_days=30)
    assert decayed[0]["final_score"] > decayed[1]["final_score"]
    assert decayed[1]["final_score"] > decayed[2]["final_score"]


def test_time_decay_with_invalid_created_at():
    """パース不能な created_at は低スコア (decay=0.1) になる"""
    from claude_remember_me.ranking import apply_time_decay
    from datetime import datetime

    now = datetime.now()
    results = [
        {"id": 1, "score": 1.0, "created_at": now.isoformat()},
        {"id": 2, "score": 1.0, "created_at": "not-a-date"},
        {"id": 3, "score": 1.0, "created_at": ""},
    ]
    decayed = apply_time_decay(results, half_life_days=30)
    fresh = next(r for r in decayed if r["id"] == 1)
    invalid1 = next(r for r in decayed if r["id"] == 2)
    invalid2 = next(r for r in decayed if r["id"] == 3)
    # 正常なレコードが不正レコードより高スコア
    assert fresh["final_score"] > invalid1["final_score"]
    # 不正レコードは decay=0.1 が適用される
    assert abs(invalid1["final_score"] - 0.1) < 0.001
    assert abs(invalid2["final_score"] - 0.1) < 0.001
