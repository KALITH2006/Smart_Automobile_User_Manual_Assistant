"""Normalised evidence-sufficiency score S = alpha*rel + beta*cov + gamma*mar (Eq. 3).

Deviation from the paper's wording: with a single retriever (mode M1) the rank-based term
RRF(d1) / (R/(c+1)) equals 1 for every query, and the margin of rank-based scores is a constant,
so neither carries information. For R = 1 we therefore use (a) the fraction of distinct query
terms present in the top section for `rel`, and (b) BM25 scores for `mar`.
"""
from __future__ import annotations

import re

from .types import AdaptiveConfig, Hit

_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or should the to "
    "what when where which why with you your me we car vehicle".split()
)


def _terms(text: str) -> set[str]:
    from .searchers import stem

    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def rel_fused(top_fused_score: float, n_retrievers: int, c: int) -> float:
    """Top fused score divided by its maximum (rank 1 in every retriever)."""
    best = n_retrievers / (c + 1)
    return min(1.0, top_fused_score / best) if best else 0.0


def rel_sparse(query: str, top: Hit | None) -> float:
    q = _terms(query)
    if not q or top is None:
        return 0.0
    return len(q & _terms(top.text)) / len(q)


def coverage(dtcs: list[str], evidence: list[Hit]) -> float:
    if not dtcs:
        return 1.0
    blob = " ".join(h.text for h in evidence).upper()
    return sum(code.upper() in blob for code in dtcs) / len(dtcs)


def margin(scores: list[float]) -> float:
    """(s1 - sn) / s1 over the evidence set; 0 if there is nothing to separate."""
    if len(scores) < 2 or scores[0] <= 0:
        return 0.0
    return max(0.0, min(1.0, (scores[0] - scores[-1]) / scores[0]))


def sufficiency(rel: float, cov: float, mar: float, cfg: AdaptiveConfig) -> float:
    total = cfg.alpha + cfg.beta + cfg.gamma
    return (cfg.alpha * rel + cfg.beta * cov + cfg.gamma * mar) / total
