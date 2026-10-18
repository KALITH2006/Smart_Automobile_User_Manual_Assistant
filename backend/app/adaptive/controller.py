"""Algorithm 1: route, retrieve, score sufficiency, escalate within a budget, or abstain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .context import EncodedContext, encode
from .router import Router, escalate
from .searchers import Searcher
from .sufficiency import coverage, margin, rel_fused, rel_sparse, sufficiency
from .types import (
    AdaptiveConfig,
    DiagnosticQuery,
    Hit,
    Mode,
    Outcome,
    RouteDecision,
    Step,
)

# (current query, evidence so far, original complaint) -> rewritten query
Rewriter = Callable[[str, "list[Hit]", str], str]


@dataclass
class Retrieval:
    evidence: list[Hit]
    scores: list[float]
    rel: float
    exact: bool = False  # M0 hit: S is set to 1 directly


def rrf_fuse(rankings: list[list[Hit]], c: int) -> list[Hit]:
    """Reciprocal Rank Fusion: sum over retrievers of 1 / (c + rank). Hit.score = fused score."""
    fused: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (c + rank)
            by_id.setdefault(hit.id, hit)
    ordered = sorted(fused, key=fused.__getitem__, reverse=True)
    return [Hit(i, by_id[i].page, by_id[i].text, fused[i], by_id[i].meta) for i in ordered]


def retrieve(
    mode: Mode,
    text: str,
    ctx: EncodedContext,
    searcher: Searcher,
    cfg: AdaptiveConfig,
    dtc_table: dict[str, dict],
) -> Retrieval:
    flt, n = ctx.filter, cfg.n_evidence

    if mode is Mode.M0:
        hits = [
            Hit(-1, 0, f"{code}: {dtc_table[code]['description']}", 1.0,
                {"system": dtc_table[code].get("system"), "source": "dtc_table"})
            for code in ctx.dtcs
            if code in dtc_table
        ]
        return Retrieval(hits, [1.0] * len(hits), 1.0, exact=bool(hits))

    if mode is Mode.M1:
        hits = searcher.bm25(text, n, flt)
        return Retrieval(hits, [h.score for h in hits], rel_sparse(text, hits[0] if hits else None))

    fused = rrf_fuse(
        [searcher.bm25(text, cfg.depth, flt), searcher.dense(text, cfg.depth, flt)], cfg.rrf_c
    )[:n]
    top = fused[0].score if fused else 0.0
    return Retrieval(fused, [h.score for h in fused], rel_fused(top, 2, cfg.rrf_c))


def score(retrieval: Retrieval, ctx: EncodedContext, cfg: AdaptiveConfig) -> float:
    if retrieval.exact:
        return 1.0
    if not retrieval.evidence:
        return 0.0
    cov = coverage(ctx.dtcs, retrieval.evidence)
    return sufficiency(retrieval.rel, cov, margin(retrieval.scores), cfg)


def run_adaptive(
    query: DiagnosticQuery,
    searcher: Searcher,
    *,
    cfg: AdaptiveConfig = AdaptiveConfig(),
    router: Router | None = None,
    dtc_table: dict[str, dict] | None = None,
    rewriter: Optional[Rewriter] = None,
) -> Outcome:
    dtc_table = dtc_table or {}
    router = router or Router(dtc_table, theta_r=cfg.theta_r)

    if cfg.use_encoder:
        ctx = encode(query, cfg.sensor_margin)
    else:  # ablation: raw text only, no filter and no sensor tokens
        ctx = EncodedContext(text=query.text.strip())
        ctx.dtcs = [c.upper() for c in query.dtcs]
    text = ctx.text

    route = router.route(query, ctx) if cfg.use_router else RouteDecision(Mode.M2, 1.0, "router disabled")
    mode, llm_calls = route.mode, route.llm_calls
    steps: list[Step] = []
    retrieval, s = Retrieval([], [], 0.0), 0.0

    for r in range(1, cfg.t_max + 1):
        retrieval = retrieve(mode, text, ctx, searcher, cfg, dtc_table)
        if mode is Mode.M0 and not retrieval.exact:  # no table hit: fall through to hybrid at once
            mode = Mode.M2
            retrieval = retrieve(mode, text, ctx, searcher, cfg, dtc_table)
        s = score(retrieval, ctx, cfg)
        steps.append(Step(r, mode, s, text))
        if s >= cfg.tau_for(mode) or r == cfg.t_max:
            break
        mode = escalate(mode)
        if rewriter is not None:
            text = rewriter(text, retrieval.evidence, query.text)
            llm_calls += 1

    abstained = cfg.allow_abstain and s < cfg.tau_for(mode)
    return Outcome(
        abstained=abstained,
        sufficiency=s,
        mode=steps[-1].mode,
        route=route,
        evidence=retrieval.evidence,  # kept even when abstaining; callers decide whether to use it
        steps=steps,
        sensor_tokens=ctx.sensor_tokens,
        final_query=text,
        llm_calls=llm_calls,
    )
