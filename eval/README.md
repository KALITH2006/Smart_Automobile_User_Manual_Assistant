# Evaluation: ADAPT-Diag vs. static retrieval baselines

Reproducible retrieval and abstention benchmark for the adaptive pipeline in `backend/app/adaptive/`.

```bash
cd eval
python build_golden.py            # derive gold pages from evidence regexes -> golden.jsonl
python run_eval.py --dense        # B1-B3 over all answerable queries (needs `ollama serve`)
python adaptive_eval.py --no-llm  # B4 + ablations, tuned on dev, reported on test
python adaptive_eval.py           # same, with the LLM router fallback + query rewriter
```

Data: two public owner's manuals (BMW M4, 282 pp; Suzuki Ciaz, 300 pp), not redistributed
(`eval/data/` is gitignored; put the PDFs in `eval/data/manuals/` and regenerate the page dumps).

## Benchmark

| | dev | test |
|---|---|---|
| answerable (procedure / lookup / symptom) | 18 | 20 |
| unanswerable (not in an owner's manual; expected outcome is abstention) | 14 | 19 |

* **Labels are auditable.** Each answerable question carries an evidence regex; gold pages are
  *derived* by matching it against the page text, and the build fails if it matches 0 or >4 pages.
  Unanswerable questions are checked to have no mention of their key terms anywhere in the manual.
* **Split.** Deterministic hash of (manual, question). Weights and thresholds are tuned on `dev` only.

## Results (test split, 95% bootstrap CIs)

Shipped configuration (no LLM in the loop; `--no-llm`):

| System | R@1 | R@3 | R@5 | MRR | correct abstain | false abstain |
|---|---|---|---|---|---|---|
| B1 BM25 (stemmed) | 0.65 | 0.80 | **0.95** [0.85, 1.00] | 0.76 | – | – |
| B2 Dense (nomic-embed-text) | 0.50 | 0.80 | 0.90 [0.75, 1.00] | 0.66 | – | – |
| B3 Static hybrid (RRF) | 0.60 | 0.80 | 0.80 [0.60, 0.95] | 0.70 | – | – |
| **B4 ADAPT-Diag** | 0.60 | 0.80 | 0.90 [0.75, 1.00] | 0.72 | 0.58 [0.37, 0.79] | 0.10 [0.00, 0.25] |
| &nbsp;&nbsp;− router (always M2) | 0.60 | 0.80 | 0.80 | 0.70 | **0.84** [0.68, 1.00] | 0.20 [0.05, 0.40] |

Full tables: `results/adaptive_no_llm.md` and `results/adaptive.md` (with the LLM).

## What the results say (read this before quoting numbers)

1. **Adaptive retrieval did not beat the best static baseline.** B4 vs. the static hybrid B3:
   Recall@5 +0.10, paired 95% CI [0.00, +0.25]; MRR +0.02, CI [-0.08, +0.15]. Neither is
   significant at n=20. Stemmed BM25 alone (B1) is as good as anything here. Owner's-manual
   questions are mostly keyword-friendly, which is a poor setting for showing off routing.
2. **The paper's sufficiency score is only partly useful.** On dev, hybrid `rel` separates
   answerable from unanswerable queries well (AUROC 0.97), but the *margin* term is
   anti-informative (AUROC 0.20) and the tuner zeroes it (`gamma = 0`). The paper's `rel` is
   degenerate for a single retriever (always 1), so M1 scores query-term coverage instead.
3. **The router costs abstention quality here.** With each variant re-tuned on dev, "always M2" abstains
   *better* (0.84 vs. 0.58 correct abstention). The sparse-only path has weaker evidence of sufficiency.
4. **LLM query rewriting breaks abstention** (correct abstention 0.58 → 0.11, R@5 0.90 → 0.95): rewriting
   makes unanswerable queries sound like the manual, inflating S past a threshold tuned without it.
   Thresholds must be re-tuned jointly with the rewriter. It is therefore **off by default**.
5. **The context encoder and DTC lookup (M0) are not exercised** by this benchmark: the queries contain
   no trouble codes, sensor data or vehicle profile. They are unit-tested, not benchmarked.

## Deviations from the paper

* Router rule order is M0, **M3**, M1, M2 (the paper tests M1 before M3, which sends a
  multi-symptom query containing one part word to a single-retriever lookup).
* `rel` and `mar` for single-retriever mode use query-term coverage and BM25 scores (see `sufficiency.py`).
* Per-scale threshold: M1 may carry its own `tau_m1` (None → shared `tau`).
* Tuning replays the real accept/escalate loop on cached scores; it does not model the LLM rewriter.

## Limitations

* **Small and single-annotator.** 39 test queries; CIs are wide. No inter-annotator agreement (Cohen's κ).
* **Not a clean blind test.** I wrote the questions with the evidence pages in view, and I looked at
  test-split numbers while developing (the stemming fix, per-scale thresholds and the fair ablation
  re-tuning were all made after seeing them). Treat the test split as development-adjacent and
  collect a fresh, independently written set before claiming a result.
* **Threshold is brittle.** `tau = 0.986` sits between two dev scores; in a live check "How do I
  jump start the car?" retrieved the right pages (274-275) but scored 0.977 and abstained.
* No answer-quality metrics yet (faithfulness, top-1 root-cause accuracy, RAGAS). Retrieval and
  abstention only. No latency numbers.
* Ground truth is "a page that contains the evidence phrase"; other pages may also answer the question.
