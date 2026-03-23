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
