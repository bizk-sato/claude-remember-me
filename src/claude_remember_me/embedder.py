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
