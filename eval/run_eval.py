"""Run retrieval baselines over eval/golden.jsonl and write a results table.

Usage (from eval/):
    python run_eval.py                 # BM25 only (no Ollama needed)
    python run_eval.py --dense         # + dense and hybrid, needs `ollama serve`
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from corpus import MANUALS, load_chunks
from metrics import bootstrap_ci, first_hit_rank, recall_at_k, reciprocal_rank
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever

HERE = Path(__file__).parent
KS = (1, 3, 5)
TOP_K = 5  # chunks passed to the generator, as RETRIEVER_TOP_K in the backend


def ollama_embedder(model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
    """Batched, retrying embedder: one giant request can crash the Ollama runner."""
    import time

    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=model, base_url=base_url)

    def embed(texts: list[str], batch: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            for attempt in range(4):
                try:
                    out += emb.embed_documents(texts[i : i + batch])
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))
        return out

    return embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--golden", default=str(HERE / "golden.jsonl"))
    args = ap.parse_args()

    queries = [json.loads(l) for l in Path(args.golden).read_text(encoding="utf8").splitlines() if l.strip()]
    answerable = [q for q in queries if q["gold_pages"]]

    chunks = {m: load_chunks(m) for m in MANUALS}
    systems: dict[str, dict] = {}
    for m, cs in chunks.items():
        bm25 = BM25Retriever(cs)
        systems.setdefault(bm25.name, {})[m] = bm25
        if args.dense:
            dense = DenseRetriever(cs, ollama_embedder(), cache=HERE / "data" / "cache" / f"{m}_nomic.npy")
            hybrid = HybridRetriever(bm25, dense)
            systems.setdefault(dense.name, {})[m] = dense
            systems.setdefault(hybrid.name, {})[m] = hybrid

    rows = []
    for name, per_manual in systems.items():
        ranks, by_group = [], defaultdict(list)
        for q in answerable:
            r = per_manual[q["manual"]]
            idx = r.rank(q["question"], depth=50)
            pages = [chunks[q["manual"]][i].page for i in idx]
            rank = first_hit_rank(pages, set(q["gold_pages"]))
            ranks.append(rank)
            by_group[q["group"]].append(rank)
        rows.append((name, "all", ranks))
        rows += [(name, g, rs) for g, rs in sorted(by_group.items())]

    lines = ["| System | Group | n | R@1 | R@3 | R@5 | MRR |", "|---|---|---|---|---|---|---|"]
    for name, group, ranks in rows:
        cells = []
        for k in KS:
            m, lo, hi = bootstrap_ci(recall_at_k(ranks, k))
            cells.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
        m, lo, hi = bootstrap_ci(reciprocal_rank(ranks))
        cells.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
        lines.append(f"| {name} | {group} | {len(ranks)} | " + " | ".join(cells) + " |")

    out = "\n".join(lines)
    print(out)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "retrieval.md").write_text(out + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
