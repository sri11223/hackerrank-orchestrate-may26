"""Stage 4 hybrid retrieval: BM25 + dense BGE + reciprocal-rank fusion."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from .config import DEFAULT_CHUNKS_JSONL, PROCESSED_DATA_DIR, RRF_K, get_settings
from .schema import ChunkRecord, RetrievedChunk


logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./:%+\-']*", re.IGNORECASE)
_DOMAIN_ALIASES = {
    "hackerrank": "hackerrank",
    "hacker rank": "hackerrank",
    "claude": "claude",
    "anthropic": "claude",
    "visa": "visa",
}
_SINGLETON_LOCK = Lock()


class RetrievalDependencyError(RuntimeError):
    """Raised when a required retrieval dependency is missing."""


@dataclass(frozen=True)
class _ScopeIndex:
    """BM25 index plus global chunk ids for one search scope."""

    name: str
    indices: tuple[int, ...]
    bm25: Any


def load_chunks(chunks_path: Path = DEFAULT_CHUNKS_JSONL) -> list[ChunkRecord]:
    """Load and validate the ingestion JSONL file."""

    resolved = chunks_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Chunk file does not exist: {resolved}. Run `python -m triage.cli ingest` first."
        )

    chunks: list[ChunkRecord] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunks.append(ChunkRecord.model_validate(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Invalid chunk JSONL at {resolved}:{line_number}") from exc

    if not chunks:
        raise ValueError(f"Chunk file is empty: {resolved}")
    return chunks


def retrieve(
    query: str,
    domain: str | None = None,
    k: int = 5,
    *,
    chunks_path: Path = DEFAULT_CHUNKS_JSONL,
) -> list[RetrievedChunk]:
    """Retrieve top chunks using the process-wide singleton retriever.

    Empty queries return an empty list without initializing the dense model.
    """

    if not query or not query.strip() or k <= 0:
        return []
    return get_retriever(chunks_path).retrieve(query=query, domain=domain, k=k)


def embed_query(query: str, *, chunks_path: Path = DEFAULT_CHUNKS_JSONL) -> Any | None:
    """Embed a ticket query using the process-wide dense retriever singleton."""

    if not query or not query.strip():
        return None
    return get_retriever(chunks_path).embed_query(query)


def get_retriever(chunks_path: Path = DEFAULT_CHUNKS_JSONL) -> "HybridRetriever":
    """Return the singleton retriever for a chunk file.

    The lock prevents two concurrent ticket workers from loading the BGE model
    twice. The cached object holds BM25 indexes, BGE model, and dense embeddings
    in memory for the whole run.
    """

    resolved = str(chunks_path.resolve())
    settings = get_settings()
    with _SINGLETON_LOCK:
        return _get_retriever_cached(
            resolved,
            settings.dense_retrieval_model,
            settings.dense_max_seq_length,
        )


@lru_cache(maxsize=2)
def _get_retriever_cached(
    chunks_path: str,
    model_name: str,
    dense_max_seq_length: int,
) -> "HybridRetriever":
    return HybridRetriever(
        chunks_path=Path(chunks_path),
        dense_model_name=model_name,
        dense_max_seq_length=dense_max_seq_length,
    )


class HybridRetriever:
    """In-memory BM25 + dense retrieval with RRF fusion."""

    def __init__(self, chunks_path: Path, dense_model_name: str, dense_max_seq_length: int) -> None:
        self.chunks_path = chunks_path.resolve()
        self.dense_model_name = dense_model_name
        self.dense_max_seq_length = dense_max_seq_length
        self.chunks = load_chunks(self.chunks_path)
        self._tokenized_corpus = [tokenize(sparse_document_text(chunk)) for chunk in self.chunks]
        self._scope_indexes = self._build_bm25_scopes()
        self._embedding_model = None
        self._embeddings = None

        logger.info("Loaded %s chunks from %s", len(self.chunks), self.chunks_path)
        self._load_dense_index()

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({chunk.domain for chunk in self.chunks}))

    def retrieve(self, query: str, domain: str | None = None, k: int = 5) -> list[RetrievedChunk]:
        """Run BM25 and BGE retrieval, then combine normalized rankings using RRF."""

        normalized_query = " ".join((query or "").split())
        if not normalized_query or k <= 0:
            return []

        scope = self._scope_for_domain(domain)
        pool_size = max(k * 20, 100)
        bm25_rankings = self._rank_bm25(normalized_query, scope, pool_size)
        dense_rankings = self._rank_dense(normalized_query, scope, pool_size)

        fused: dict[int, float] = {}
        for global_index, (rank, _score) in bm25_rankings.items():
            fused[global_index] = fused.get(global_index, 0.0) + (1.0 / (RRF_K + rank))
        for global_index, (rank, _score) in dense_rankings.items():
            fused[global_index] = fused.get(global_index, 0.0) + (1.0 / (RRF_K + rank))

        if not fused:
            return []

        max_possible = 2.0 / (RRF_K + 1)
        ranked_indices = sorted(fused, key=lambda index: (-fused[index], index))[:k]
        results: list[RetrievedChunk] = []
        for global_index in ranked_indices:
            chunk = self.chunks[global_index]
            bm25 = bm25_rankings.get(global_index)
            dense = dense_rankings.get(global_index)
            rrf_score = fused[global_index]
            results.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    rrf_score=rrf_score,
                    normalized_score=min(1.0, rrf_score / max_possible),
                    bm25_rank=bm25[0] if bm25 else None,
                    bm25_score=bm25[1] if bm25 else None,
                    dense_rank=dense[0] if dense else None,
                    dense_score=dense[1] if dense else None,
                )
            )
        return results

    def _build_bm25_scopes(self) -> dict[str, _ScopeIndex]:
        BM25Okapi = _load_bm25_class()
        domains: dict[str, list[int]] = {"all": list(range(len(self.chunks)))}
        for index, chunk in enumerate(self.chunks):
            domains.setdefault(chunk.domain.casefold(), []).append(index)

        scopes: dict[str, _ScopeIndex] = {}
        for name, indices in domains.items():
            tokenized = [self._tokenized_corpus[index] for index in indices]
            scopes[name] = _ScopeIndex(name=name, indices=tuple(indices), bm25=BM25Okapi(tokenized))
            logger.info("Built BM25 scope '%s' over %s chunks", name, len(indices))
        return scopes

    def _load_dense_index(self) -> None:
        model_cls = _load_sentence_transformer_class()
        logger.info("Loading dense retrieval model once: %s", self.dense_model_name)
        try:
            self._embedding_model = model_cls(self.dense_model_name)
        except Exception as exc:
            raise RetrievalDependencyError(
                f"Could not load dense retrieval model {self.dense_model_name!r}. "
                "Set DENSE_RETRIEVAL_MODEL to a locally available SentenceTransformer model "
                "or ensure the model can be downloaded."
            ) from exc
        if self.dense_max_seq_length > 0:
            self._embedding_model.max_seq_length = self.dense_max_seq_length

        cached = self._load_embedding_cache()
        if cached is not None:
            self._embeddings = cached
            logger.info("Loaded dense embeddings from cache for %s chunks", len(self.chunks))
            return

        logger.info(
            "Encoding %s corpus chunks with %s (max_seq_length=%s)",
            len(self.chunks),
            self.dense_model_name,
            getattr(self._embedding_model, "max_seq_length", "unknown"),
        )
        try:
            self._embeddings = self._embedding_model.encode(
                [dense_document_text(chunk) for chunk in self.chunks],
                batch_size=128,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        except Exception as exc:
            raise RetrievalDependencyError("Could not encode corpus chunks for dense retrieval.") from exc
        self._save_embedding_cache(self._embeddings)

    def _scope_for_domain(self, domain: str | None) -> _ScopeIndex:
        normalized = normalize_domain(domain)
        if normalized and normalized in self._scope_indexes:
            return self._scope_indexes[normalized]
        return self._scope_indexes["all"]

    def _rank_bm25(
        self,
        query: str,
        scope: _ScopeIndex,
        pool_size: int,
    ) -> dict[int, tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return {}

        import numpy as np

        raw_scores = np.asarray(scope.bm25.get_scores(tokens), dtype=float)
        if raw_scores.size == 0:
            return {}
        scores = _normalize_nonnegative_scores(raw_scores)

        ordered_positions = np.argsort(scores)[::-1]
        rankings: dict[int, tuple[int, float]] = {}
        rank = 1
        for position in ordered_positions:
            if float(raw_scores[position]) <= 0.0:
                break
            score = float(scores[position])
            rankings[scope.indices[int(position)]] = (rank, score)
            rank += 1
            if len(rankings) >= pool_size:
                break
        return rankings

    def _rank_dense(
        self,
        query: str,
        scope: _ScopeIndex,
        pool_size: int,
    ) -> dict[int, tuple[int, float]]:
        if self._embedding_model is None or self._embeddings is None:
            raise RuntimeError("Dense index was not initialized")

        import numpy as np

        query_embedding = self.embed_query(query)
        if query_embedding is None:
            return {}
        scope_indices = np.asarray(scope.indices, dtype=int)
        if scope_indices.size == 0:
            return {}

        raw_scores = self._embeddings[scope_indices] @ query_embedding
        scores = _normalize_cosine_scores(raw_scores)
        ordered_positions = np.argsort(scores)[::-1][:pool_size]
        return {
            scope.indices[int(position)]: (rank, float(scores[position]))
            for rank, position in enumerate(ordered_positions, start=1)
        }

    def embed_query(self, query: str) -> Any | None:
        """Return the normalized BGE embedding for a support query."""

        normalized_query = " ".join((query or "").split())
        if not normalized_query:
            return None
        if self._embedding_model is None:
            raise RuntimeError("Dense model was not initialized")
        return self._embedding_model.encode(
            ["Represent this sentence for searching relevant passages: " + normalized_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def _cache_paths(self) -> tuple[Path, Path]:
        safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.dense_model_name).strip("_")
        suffix = f"{safe_model}_seq{self.dense_max_seq_length}"
        return (
            PROCESSED_DATA_DIR / f"embeddings_{suffix}.npy",
            PROCESSED_DATA_DIR / f"embeddings_{suffix}.json",
        )

    def _load_embedding_cache(self) -> Any | None:
        import numpy as np

        embeddings_path, metadata_path = self._cache_paths()
        if not embeddings_path.exists() or not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        expected_ids = [chunk.chunk_id for chunk in self.chunks]
        if (
            metadata.get("model_name") != self.dense_model_name
            or metadata.get("dense_max_seq_length") != self.dense_max_seq_length
            or metadata.get("chunk_ids") != expected_ids
        ):
            return None

        return np.load(embeddings_path)

    def _save_embedding_cache(self, embeddings: Any) -> None:
        import numpy as np

        embeddings_path, metadata_path = self._cache_paths()
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        metadata = {
            "model_name": self.dense_model_name,
            "dense_max_seq_length": self.dense_max_seq_length,
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25 while preserving useful support artifacts."""

    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).casefold()
        tokens.append(token)
        if len(token) > 4 and token.endswith("'s"):
            tokens.append(token[:-2])
        elif len(token) > 4 and token.endswith("s"):
            tokens.append(token[:-1])
    return tokens


def dense_document_text(chunk: ChunkRecord) -> str:
    """Text passed to the embedding model."""

    return f"{chunk.heading_path}\n{chunk.text}".strip()


def sparse_document_text(chunk: ChunkRecord) -> str:
    """Text passed to BM25.

    Headings carry product names, issuer names, and exact support article titles
    that are often absent from the body text. Including them makes exact-match
    retrieval much stronger for error codes, named issuers, and support actions.
    """

    return f"{chunk.heading_path}\n{chunk.text}".strip()


def normalize_domain(domain: str | None) -> str | None:
    """Map stated company/domain text to a corpus domain, if known."""

    if domain is None:
        return None
    normalized = " ".join(str(domain).strip().casefold().split())
    if normalized in {"", "none", "null", "unknown", "n/a"}:
        return None
    return _DOMAIN_ALIASES.get(normalized)


def _normalize_nonnegative_scores(scores: Any) -> Any:
    """Normalize BM25-style nonnegative scores to 0..1 before fusion metadata."""

    import numpy as np

    clipped = np.maximum(np.asarray(scores, dtype=float), 0.0)
    max_score = float(np.max(clipped)) if clipped.size else 0.0
    if max_score <= 0.0:
        return clipped
    return clipped / max_score


def _normalize_cosine_scores(scores: Any) -> Any:
    """Normalize cosine similarities from -1..1 into 0..1 before ranking fusion."""

    import numpy as np

    values = np.asarray(scores, dtype=float)
    return np.clip((values + 1.0) / 2.0, 0.0, 1.0)


def _load_bm25_class() -> Any:
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError as exc:
        raise RetrievalDependencyError(
            "rank_bm25 is required for sparse retrieval. Install it with `pip install rank-bm25`."
        ) from exc
    return BM25Okapi


def _load_sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RetrievalDependencyError(
            "sentence-transformers is required for dense retrieval. "
            "Install it with `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer
