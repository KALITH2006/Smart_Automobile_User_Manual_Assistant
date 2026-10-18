"""Evaluate ADAPT-Diag (B4) against B1-B3 and ablations.

Protocol (paper Section VII):
  * alpha/beta/gamma and tau are tuned on the `dev` split only;
  * all reported numbers are on the `test` split;
  * 95% bootstrap CIs, plus a paired bootstrap of B4 vs B3.

Usage (from eval/):  python adaptive_eval.py [--no-llm] [--seed 0]
Needs `ollama serve` with nomic-embed-text (and llama3.1 unless --no-llm).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "backend"))

from app.adaptive import AdaptiveConfig, DiagnosticQuery, InMemorySearcher, Router, load_dtc_table, run_adaptive  # noqa: E402
from app.adaptive.context import encode  # noqa: E402
from app.adaptive.controller import retrieve, rrf_fuse  # noqa: E402
from app.adaptive.llm import make_classifier, make_rewriter  # noqa: E402
from app.adaptive.sufficiency import coverage, margin, sufficiency  # noqa: E402
from app.adaptive.router import escalate  # noqa: E402
from app.adaptive.types import Mode  # noqa: E402
from corpus import MANUALS, load_chunks  # noqa: E402
from metrics import bootstrap_ci, first_hit_rank, recall_at_k, reciprocal_rank  # noqa: E402

CACHE = HERE / "data" / "cache"
N_EVIDENCE = 5


# ------------------------------------------------------------------ cached model access

class DiskCache:
    def __init__(self, name: str):
        self.path = CACHE / name
        self.data: dict = json.loads(self.path.read_text(encoding="utf8")) if self.path.exists() else {}

    def get_or(self, key: str, fn):
        if key not in self.data:
            self.data[key] = fn()
            CACHE.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data), encoding="utf8")
        return self.data[key]


def make_embed_query(cache: DiskCache):
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return lambda q: cache.get_or(q, lambda: emb.embed_query(q))


class CachedLLM:
    """temperature-0 llama3.1 with an on-disk prompt cache so runs are reproducible."""

    def __init__(self, cache: DiskCache):
        from langchain_ollama import ChatOllama

        self.llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0, seed=0)
        self.cache = cache

    def invoke(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        return self.cache.get_or(key, lambda: self.llm.invoke(prompt).content)


def build_searchers() -> dict[str, InMemorySearcher]:
    embed_query = make_embed_query(DiskCache("query_emb.json"))
    out = {}
    for m in MANUALS:
        chunks = load_chunks(m)
        matrix = np.load(CACHE / f"{m}_nomic.npy")
        assert len(matrix) == len(chunks), f"stale embedding cache for {m}"
        out[m] = InMemorySearcher([c.text for c in chunks], [c.page for c in chunks],
                                  embed_query=embed_query, matrix=matrix)
    return out


# ------------------------------------------------------------------ tuning on dev

def components(q: dict, searchers, dtc, cfg):
    """Per-query inputs for tuning: router's initial mode, and (rel, cov, mar) under M1 and M2.

    Without an LLM rewriter the retrieval text never changes, so a query's score depends only on
    the mode (M3 retrieves exactly like M2). That lets the accept/escalate loop be simulated exactly.
    """
    query = DiagnosticQuery(q["question"])
    ctx = encode(query)
    init = Router(dtc, theta_r=cfg.theta_r).route(query, ctx).mode if cfg.use_router else Mode.M2
    per_mode = {}
    for mode in (Mode.M0, Mode.M1, Mode.M2):
        if mode is Mode.M0 and init is not Mode.M0:
            continue
        r = retrieve(mode, ctx.text, ctx, searchers[q["manual"]], cfg, dtc)
        if r.exact:
            per_mode[mode] = (1.0, 1.0, 1.0, True)
        elif mode is not Mode.M0:
            per_mode[mode] = (r.rel, coverage(ctx.dtcs, r.evidence), margin(r.scores), False)
    return init, per_mode


def simulate(init: Mode, s_by_mode: dict, cfg: AdaptiveConfig) -> tuple[float, Mode]:
    """Replay the controller's accept/escalate loop on precomputed scores -> (final S, final mode)."""
    mode = init
    for r in range(1, cfg.t_max + 1):
        if mode is Mode.M0 and mode not in s_by_mode:
            mode = Mode.M2
        score = s_by_mode[Mode.M2 if mode is Mode.M3 else mode]
        if score >= cfg.tau_for(mode) or r == cfg.t_max:
            return score, mode
        mode = escalate(mode)
    raise AssertionError("unreachable")


def simplex(step=0.1):
    n = round(1 / step)
    for a, b in itertools.product(range(n + 1), repeat=2):
        if a + b <= n:
            yield a * step, b * step, (n - a - b) * step


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random answerable > score of a random unanswerable)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def tune(dev: list[dict], searchers, dtc, cfg: AdaptiveConfig):
    """Grid-search (alpha, beta, gamma, tau, tau_m1) on dev by replaying the real decision loop.

    Objective: balanced accuracy of answer-vs-abstain (mean of correct-abstention rate on
    unanswerable and acceptance rate on answerable); ties go to fewer false abstentions, then to
    weights closest to the paper's defaults, then to the shared threshold (tau_m1=None).
    """
    comps = [(components(q, searchers, dtc, cfg), bool(q["gold_pages"])) for q in dev]
    n_ans = max(1, sum(ans for _, ans in comps))
    n_un = max(1, sum(not ans for _, ans in comps))

    def scores_for(probe):
        return [({m: (1.0 if ex else sufficiency(rel, cov, mar, probe))
                  for m, (rel, cov, mar, ex) in pm.items()}, init, ans) for (init, pm), ans in comps]

    def evaluate(scored, probe):
        tp = tn = 0
        for sb, init, ans in scored:
            score, mode = simulate(init, sb, probe)
            abstain = score < probe.tau_for(mode)
            tp += (not ans) and abstain
            tn += ans and not abstain
        return (tp / n_un + tn / n_ans) / 2, tn / n_ans

    best = None
    for a, b, g in simplex():
        if a + b + g == 0:
            continue
        probe = replace(cfg, alpha=a, beta=b, gamma=g)
        scored = scores_for(probe)
        all_s = sorted({v for sb, _, _ in scored for v in sb.values()})
        taus = [(x + y) / 2 for x, y in zip(all_s, all_s[1:])]
        m1_s = sorted({sb[Mode.M1] for sb, _, _ in scored if Mode.M1 in sb})
        tau_m1s = [None] + [(x + y) / 2 for x, y in zip(m1_s, m1_s[1:])]
        dist = abs(a - cfg.alpha) + abs(b - cfg.beta) + abs(g - cfg.gamma)
        for tau in taus:
            for tm1 in tau_m1s:
                p2 = replace(probe, tau=tau, tau_m1=tm1)
                bal, tn = evaluate(scored, p2)
                key = (bal, tn, -dist, tm1 is None)
                if best is None or key > best[0]:
                    best = (key, p2)
    (bal, _, _, _), tuned = best
    tuned = replace(tuned, alpha=round(tuned.alpha, 2), beta=round(tuned.beta, 2),
                    gamma=round(tuned.gamma, 2), tau=round(tuned.tau, 3),
                    tau_m1=None if tuned.tau_m1 is None else round(tuned.tau_m1, 3))
    pos, neg = [], []
    for sb, init, ans in scores_for(tuned):
        score, mode = simulate(init, sb, tuned)
        (pos if ans else neg).append(score - tuned.tau_for(mode))
    return tuned, bal, auroc(pos, neg)


# ------------------------------------------------------------------ evaluation

def ranks_for(pages: list[int], gold: list[int]):
    return first_hit_rank(pages, set(gold))


def paired_bootstrap(a: list[float], b: list[float], n=5000, seed=0):
    """Mean difference a-b with a 95% percentile CI over resampled queries."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def fmt(vals):
    m, lo, hi = bootstrap_ci(vals)
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM classifier/rewriter")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / "golden.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    searchers = build_searchers()
    dtc = load_dtc_table()

    llm = None if args.no_llm else CachedLLM(DiskCache("llm.json"))
    classifier = make_classifier(llm) if llm else None
    rewriter = make_rewriter(llm) if llm else None

    base = AdaptiveConfig(n_evidence=N_EVIDENCE)
    tuned, bal, dev_auc = tune(dev, searchers, dtc, base)
    from collections import Counter as _C
    print(f"tuned on dev (n={len(dev)}): alpha={tuned.alpha} beta={tuned.beta} gamma={tuned.gamma} "
          f"tau={tuned.tau} tau_m1={tuned.tau_m1}  balanced-acc={bal:.2f}  AUROC={dev_auc:.2f}")

    # every ablation is re-tuned on dev, so differences are not artefacts of B4's thresholds
    variant = {
        "router": tune(dev, searchers, dtc, replace(base, use_router=False))[0],
        "iteration": tune(dev, searchers, dtc, replace(base, t_max=1))[0],
    }

    def adaptive(cfg):
        def run(q):
            return run_adaptive(DiagnosticQuery(q["question"]), searchers[q["manual"]], cfg=cfg,
                                router=Router(dtc, classifier, cfg.theta_r), dtc_table=dtc, rewriter=rewriter)
        return run

    def static(kind):
        def run(q):
            s = searchers[q["manual"]]
            if kind == "bm25":
                hits = s.bm25(q["question"], N_EVIDENCE, {})
            elif kind == "dense":
                hits = s.dense(q["question"], N_EVIDENCE, {})
            else:
                hits = rrf_fuse([s.bm25(q["question"], 50, {}), s.dense(q["question"], 50, {})], 60)[:N_EVIDENCE]
            return type("O", (), {"evidence": hits, "abstained": False, "llm_calls": 0, "mode": None})
        return run

    systems = {
        "B1 BM25": static("bm25"),
        "B2 Dense": static("dense"),
        "B3 Static hybrid (RRF)": static("hybrid"),
        "B4 ADAPT-Diag": adaptive(tuned),
        "  - router (always M2)": adaptive(variant["router"]),
        "  - context encoder": adaptive(replace(tuned, use_encoder=False)),
        "  - iteration (Tmax=1)": adaptive(variant["iteration"]),
        "  - abstention": adaptive(replace(tuned, allow_abstain=False)),
    }

    answerable = [q for q in test if q["gold_pages"]]
    unanswerable = [q for q in test if not q["gold_pages"]]
    per_system, lines = {}, []
    lines.append(f"Test split: {len(answerable)} answerable, {len(unanswerable)} unanswerable queries. "
                 f"Tuned on dev: alpha={tuned.alpha}, beta={tuned.beta}, gamma={tuned.gamma}, tau={tuned.tau} "
                 f"(dev balanced accuracy {bal:.2f}, AUROC {dev_auc:.2f}).\n")
    lines += ["| System | R@1 | R@3 | R@5 | MRR | correct abstain | false abstain | LLM calls/q |",
              "|---|---|---|---|---|---|---|---|"]
    for name, run in systems.items():
        ans = [(q, run(q)) for q in answerable]
        ranks = [ranks_for([h.page for h in o.evidence], q["gold_pages"]) for q, o in ans]
        per_system[name] = {"hit5": recall_at_k(ranks, 5), "rr": reciprocal_rank(ranks)}
        unans = [run(q) for q in unanswerable]
        can = fmt([float(o.abstained) for o in unans])
        fa = fmt([float(o.abstained) for _, o in ans])
        calls = sum(o.llm_calls for _, o in ans) / len(ans)
        lines.append(f"| {name} | {fmt(recall_at_k(ranks, 1))} | {fmt(recall_at_k(ranks, 3))} | "
                     f"{fmt(recall_at_k(ranks, 5))} | {fmt(reciprocal_rank(ranks))} | {can} | {fa} | {calls:.2f} |")

    modes = Counter(adaptive(tuned)(q).route.mode.value for q in answerable + unanswerable)
    lines.append(f"\nRouter initial-mode distribution (test): {dict(sorted(modes.items()))}")
    for metric, label in (("hit5", "Recall@5"), ("rr", "MRR")):
        d, lo, hi = paired_bootstrap(per_system["B4 ADAPT-Diag"][metric], per_system["B3 Static hybrid (RRF)"][metric])
        lines.append(f"Paired bootstrap B4 - B3 {label}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    out = "\n".join(lines)
    print("\n" + out)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "adaptive.md").write_text(out + "\n", encoding="utf8")
    (HERE / "results" / "tuned.json").write_text(
        json.dumps({k: getattr(tuned, k) for k in ("alpha", "beta", "gamma", "tau", "tau_m1", "theta_r", "t_max", "n_evidence")}, indent=2),
        encoding="utf8")


if __name__ == "__main__":
    main()
