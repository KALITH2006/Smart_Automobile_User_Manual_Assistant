"""Retrieval metrics with bootstrap confidence intervals."""
from __future__ import annotations

import random


def first_hit_rank(ranked_pages: list[int], gold: set[int]) -> int | None:
    """1-based rank of the first retrieved chunk whose page is gold, else None."""
    for r, page in enumerate(ranked_pages, start=1):
        if page in gold:
            return r
    return None


def recall_at_k(ranks: list[int | None], k: int) -> list[float]:
    return [1.0 if r is not None and r <= k else 0.0 for r in ranks]


def reciprocal_rank(ranks: list[int | None]) -> list[float]:
    return [1.0 / r if r else 0.0 for r in ranks]


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high) with a 95% percentile bootstrap interval."""
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n)
    )
    return sum(values) / len(values), means[int(0.025 * n)], means[int(0.975 * n) - 1]
