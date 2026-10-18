"""Retrievers for baselines B1 (BM25), B2 (dense) and B3 (static hybrid + RRF)."""
from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from corpus import Chunk

RRF_C = 60  # smoothing constant, Cormack et al. 2009


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[int]], c: int = RRF_C) -> list[int]:
    """Reciprocal Rank Fusion over lists of chunk indices (best first)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class BM25Retriever:
    name = "B1 BM25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25Okapi([_tokens(c.text) for c in chunks])

    def rank(self, query: str, depth: int = 50) -> list[int]:
        scores = self.index.get_scores(_tokens(query))
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:depth]


class DenseRetriever:
    """Cosine search over embeddings. `embed` maps list[str] -> list[vector]."""

    name = "B2 Dense"

    def __init__(self, chunks: list[Chunk], embed, cache: Path | None = None):
        import numpy as np

        self.chunks, self.embed, self.np = chunks, embed, np
        if cache is not None and cache.exists():
            m = np.load(cache)
        else:
            m = np.array(embed([c.text for c in chunks]), dtype="float32")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, m)
        assert len(m) == len(chunks), "stale embedding cache"
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def rank(self, query: str, depth: int = 50) -> list[int]:
        q = self.np.array(self.embed([query])[0], dtype="float32")
        q /= self.np.linalg.norm(q)
        return list((self.matrix @ q).argsort()[::-1][:depth])


class HybridRetriever:
    name = "B3 Hybrid (RRF)"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.parts = (bm25, dense)

    def rank(self, query: str, depth: int = 50) -> list[int]:
        return rrf_fuse([p.rank(query, depth) for p in self.parts])[:depth]
