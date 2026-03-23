"""Text embedding using Ruri v3-310m."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "cl-nagoya/ruri-v3-310m"
QUERY_PREFIX = "検索クエリ: "
PASSAGE_PREFIX = "文章: "
MAX_CHARS = 2000
BATCH_SIZE = 8


def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    return text[:max_chars] if len(text) > max_chars else text


class Embedder:
    def __init__(self) -> None:
        self._model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    def embed(self, text: str, is_query: bool = False) -> list[float]:
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        vector = self._model.encode(
            prefix + _truncate(text), normalize_embeddings=True
        )
        return vector.tolist()

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        prefixed = [prefix + _truncate(t) for t in texts]
        all_vectors = []
        for i in range(0, len(prefixed), BATCH_SIZE):
            batch = prefixed[i : i + BATCH_SIZE]
            vectors = self._model.encode(batch, normalize_embeddings=True)
            all_vectors.extend(v.tolist() for v in vectors)
        return all_vectors
