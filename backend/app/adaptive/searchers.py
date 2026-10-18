"""Searchers: BM25 + dense retrieval over one manual, with metadata pre-filtering."""
from __future__ import annotations

import re
from typing import Callable, Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .context import metadata_matches
from .types import Hit


class Searcher(Protocol):
    def bm25(self, query: str, k: int, flt: dict) -> list[Hit]: ...
    def dense(self, query: str, k: int, flt: dict) -> list[Hit]: ...


def stem(word: str) -> str:
    """Light suffix stripping so "start" matches "starting" and "tires" matches "tire"."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


class InMemorySearcher:
    """Sparse + dense search over pre-chunked sections.

    `embed_query` maps str -> vector. Pass `matrix` (n_chunks x dim) to reuse stored embeddings,
    otherwise `embed_documents` is called once on all chunk texts.
    """

    def __init__(
        self,
        texts: Sequence[str],
        pages: Sequence[int],
        metas: Sequence[dict] | None = None,
        *,
        embed_query: Callable[[str], Sequence[float]],
        embed_documents: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
        matrix: np.ndarray | None = None,
    ):
        self.texts, self.pages = list(texts), list(pages)
        self.metas = list(metas) if metas is not None else [{} for _ in self.texts]
        self.embed_query = embed_query
        self._bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        if matrix is None:
            if embed_documents is None:
                raise ValueError("provide `matrix` or `embed_documents`")
            matrix = np.array(embed_documents(self.texts), dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.where(norms == 0, 1, norms)

    def _allowed(self, flt: dict) -> np.ndarray:
        if not flt:
            return np.ones(len(self.texts), dtype=bool)
        return np.array([metadata_matches(m, flt) for m in self.metas])

    def _top(self, scores: np.ndarray, k: int, flt: dict) -> list[Hit]:
        scores = np.where(self._allowed(flt), scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [
            Hit(int(i), self.pages[i], self.texts[i], float(scores[i]), self.metas[i])
            for i in order
            if np.isfinite(scores[i])
        ]

    def bm25(self, query: str, k: int, flt: dict) -> list[Hit]:
        return self._top(np.asarray(self._bm25.get_scores(tokenize(query))), k, flt)

    def dense(self, query: str, k: int, flt: dict) -> list[Hit]:
        q = np.asarray(self.embed_query(query), dtype="float32")
        n = np.linalg.norm(q)
        return self._top(self._matrix @ (q / (n or 1.0)), k, flt)


def chroma_searcher(store, embed_query) -> InMemorySearcher:
    """Build a searcher from a persisted Chroma collection (one manual)."""
    data = store.get(include=["documents", "metadatas", "embeddings"])
    metas = [dict(m or {}) for m in data["metadatas"]]
    return InMemorySearcher(
        data["documents"],
        [int(m.get("page", 0)) for m in metas],
        metas,
        embed_query=embed_query,
        matrix=np.array(data["embeddings"], dtype="float32"),
    )
