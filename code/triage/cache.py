"""In-memory semantic cache for duplicate or near-duplicate tickets."""

from __future__ import annotations

from typing import Any

import numpy as np


class SemanticCache:
    """Tiny process-local cache keyed by dense ticket embeddings."""

    def __init__(self) -> None:
        self.cache: list[dict[str, Any]] = []

    def check_cache(self, query_embedding: Any, threshold: float = 0.95) -> Any | None:
        """Return the cached result when cosine similarity is above threshold."""

        if query_embedding is None or not self.cache:
            return None

        query = _as_vector(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return None

        best_score = -1.0
        best_result: Any | None = None
        for item in self.cache:
            cached = _as_vector(item["embedding"])
            cached_norm = np.linalg.norm(cached)
            if cached_norm == 0.0:
                continue
            score = float(np.dot(query, cached) / (query_norm * cached_norm))
            if score > best_score:
                best_score = score
                best_result = item["result"]

        if best_score > threshold:
            return _copy_result(best_result)
        return None

    def add_to_cache(self, query_embedding: Any, result: Any) -> None:
        """Store a processed ticket result for future semantic reuse."""

        if query_embedding is None:
            return
        self.cache.append(
            {
                "embedding": _as_vector(query_embedding).copy(),
                "result": _copy_result(result),
            }
        )


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _copy_result(result: Any) -> Any:
    if hasattr(result, "model_copy"):
        return result.model_copy(deep=True)
    return result
