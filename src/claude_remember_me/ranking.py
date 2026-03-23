"""Pure ranking algorithms: RRF fusion and time decay."""

from __future__ import annotations

from datetime import datetime, timezone
from math import pow

RRF_K = 60
TIME_DECAY_HALF_LIFE_DAYS = 30


def rrf_fusion(
    fts_results: list[dict], vec_results: list[dict], k: int = RRF_K
) -> list[dict]:
    """Reciprocal Rank Fusion で 2 つのランキングを統合する"""
    scores: dict[int, float] = {}
    metadata: dict[int, dict] = {}

    for r in fts_results:
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + r["rank"])
        metadata[rid] = r

    for r in vec_results:
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + r["rank"])
        if rid not in metadata:
            metadata[rid] = r

    results = []
    for rid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = {**metadata[rid], "score": score}
        results.append(entry)
    return results


def apply_time_decay(
    results: list[dict], half_life_days: int = TIME_DECAY_HALF_LIFE_DAYS
) -> list[dict]:
    """時間減衰を適用: decay = 0.5 ^ (days / half_life)"""
    now = datetime.now(timezone.utc)
    decayed = []
    for r in results:
        created_str = r.get("created_at", "")
        try:
            created = datetime.fromisoformat(created_str)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            created = now
        days_elapsed = max((now - created).total_seconds() / 86400, 0)
        decay = pow(0.5, days_elapsed / half_life_days)
        decayed.append({**r, "final_score": r.get("score", 0.0) * decay})
    decayed.sort(key=lambda x: x["final_score"], reverse=True)
    return decayed
