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
