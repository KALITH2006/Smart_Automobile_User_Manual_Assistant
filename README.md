# Smart Automobile User Manual Assistant

Upload a vehicle owner's manual (PDF) and ask questions in plain English, such as
*"How do I jump start the car?"*. You get a step-by-step answer with the **exact manual pages**
it came from. The assistant answers only from the manual, and when the retrieved evidence is too
weak it **says so instead of guessing**.

It is a retrieval-augmented generation (RAG) system with an **adaptive retrieval layer**
(ADAPT-Diag) that routes each query to a suitable search strategy, scores whether the evidence is
sufficient, and abstains when it is not. The adaptive layer is evaluated on a reproducible
benchmark, and the results, including the unflattering ones, are published in [`eval/`](eval/README.md).

## Highlights

- **Cited answers.** Every answer shows source pages and snippets; page numbers survive chunking.
- **Adaptive retrieval.** Exact trouble-code lookup, keyword search, hybrid BM25 + dense search with
  reciprocal-rank fusion, or iterative retrieval, chosen per query by a rule-based router.
- **Evidence-sufficiency check and abstention.** A normalised score decides whether to answer,
  escalate to a stronger search mode within a retry budget, or decline.
- **Deterministic safety notices** for brakes, airbags, fuel leaks, high-voltage systems and overheating.
- **Multiple manuals** with per-manual vector collections, persistent conversation history, and
  follow-up questions.
- **Pluggable LLM backend:** OpenAI, or fully local Llama via Ollama (no API key).
- **Tested and measured:** 31 unit tests and an evaluation harness with bootstrap confidence intervals.
- **One-command deployment** with Docker Compose.

## Architecture

```
 Next.js chat UI  ──── question ────▶  FastAPI
 (React / Tailwind)                     │
        ▲                               ▼
        │                  ┌──────────────────────────┐
        │                  │ Context encoder           │  DTCs, vehicle filter, sensor tokens
        │                  ├──────────────────────────┤
        │                  │ Router                    │  M0 exact code · M1 BM25 · M2 hybrid · M3 iterative
        │                  ├──────────────────────────┤
        │                  │ Retrieval (per manual)    │  BM25 (stemmed) + dense (Chroma) → RRF
        │                  ├──────────────────────────┤
        │                  │ Sufficiency score S       │  S ≥ τ ?  ── no ──▶ escalate M1→M2→M3 (≤ T_max)
        │                  └──────────┬───────────────┘                    │ still no
        │                             │ yes                                ▼
        │                             ▼                               abstain, show
        │                   LLM answers ONLY from                     closest pages
        │                   the retrieved sections
        │                             │
        └──── answer + page citations + mode + evidence score ◀──┘
                     SQLite: manuals, conversations, messages
```

| Mode | When the router picks it | Retrieval |
|---|---|---|
| **M0** | one trouble code, no other text, code is in the table | exact table lookup |
| **M1** | short query naming a part or identifier, one system | BM25 |
| **M2** | everything else (default) | BM25 + dense, reciprocal-rank fusion |
| **M3** | several symptoms or systems, or low classifier confidence | hybrid, iterative with query rewriting |

Design choices:

- **One Chroma collection per manual**, keyed by a hash of the file, so re-uploading is idempotent and
  vehicles never cross-contaminate.
- **The model only sees retrieved sections.** The system prompt forbids invented page numbers, parts
  or specifications.
- **Sufficiency is scored, not assumed.** See [Evaluation](#evaluation) for how well that works.

## Tech stack

| Layer | Choice |
|---|---|
| Backend API | FastAPI (Python 3.11+) |
| RAG | LangChain, ChromaDB, `rank-bm25` |
| LLM / embeddings | OpenAI (`gpt-4o-mini`) or Llama 3.1 + `nomic-embed-text` via Ollama |
| Database | SQLite via SQLAlchemy |
| Frontend | Next.js 14 (App Router), React, Tailwind CSS |
| Tests | pytest |
| Deployment | Docker, Docker Compose |

## Project structure

```
├── backend/
│   ├── app/
│   │   ├── main.py, config.py, database.py, schemas.py
│   │   ├── rag_pipeline.py          # ingestion, baseline answer(), answer_adaptive()
│   │   ├── routers/                 # /api/manuals, /api/chat, /api/conversations
│   │   └── adaptive/                # ADAPT-Diag
│   │       ├── context.py           # DTC extraction, vehicle filter, sensor tokens
│   │       ├── router.py            # mode selection and escalation
│   │       ├── searchers.py         # stemmed BM25 + dense search with metadata filters
│   │       ├── sufficiency.py       # relevance / coverage / margin → S
│   │       ├── controller.py        # route → retrieve → score → escalate → abstain
│   │       ├── llm.py               # classifier, rewriter, structured generation helpers
│   │       └── data/dtc_generic.json
│   ├── tests/test_adaptive.py
│   └── requirements.txt, requirements-dev.txt, Dockerfile, .env.example
├── frontend/app/                    # page, components (Sidebar, ChatWindow, MessageBubble, FileUpload), lib/api.ts
├── eval/                            # benchmark, baselines, ablations, results (see eval/README.md)
└── docker-compose.yml
```

## Quick start

**Prerequisites:** Python 3.11+, Node.js 18+, and either an OpenAI API key or [Ollama](https://ollama.com).
Docker Desktop is needed only for the Compose route. The frontend runs on port `3000`, the backend on `8000`.

### Option A: Docker Compose

```bash
cp backend/.env.example backend/.env      # Windows: Copy-Item backend/.env.example backend/.env
# edit backend/.env: LLM_PROVIDER=openai + OPENAI_API_KEY, or keep ollama
docker compose --profile ollama up --build     # with local Ollama
# docker compose up --build                    # with OpenAI
```

With the Ollama profile, pull the models once:

```bash
docker exec -it manual-assistant-ollama ollama pull llama3.1
docker exec -it manual-assistant-ollama ollama pull nomic-embed-text
```

Open http://localhost:3000. Health check: http://localhost:8000/api/health. API docs: http://localhost:8000/docs.

### Option B: run locally

```bash
# backend
cd backend
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env                # Windows: Copy-Item .env.example .env
python -m uvicorn app.main:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
cp .env.local.example .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

For Ollama, run `ollama serve` and `ollama pull llama3.1 && ollama pull nomic-embed-text`.
Never commit `.env`, `.env.local`, API keys or model files.

### Run the tests

```bash
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

## Configuration

Set in `backend/.env` (all optional except the provider choice):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `ADAPTIVE_RAG` | `true` | use the adaptive pipeline; `false` falls back to plain top-k retrieval |
| `ADAPTIVE_USE_LLM` | `false` | LLM router fallback and query rewriting (see the warning below) |
| `ADAPTIVE_TAU`, `ADAPTIVE_TAU_M1` | `0.986`, `0.562` | sufficiency thresholds (answer if S ≥ τ) |
| `ADAPTIVE_ALPHA/BETA/GAMMA` | `0.5/0.5/0.0` | weights of relevance / DTC coverage / margin |
| `ADAPTIVE_T_MAX` | `3` | retrieval rounds before abstaining |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVER_TOP_K` | `1000`, `150`, `5` | chunking and evidence size |

The thresholds were tuned on a small development split. Re-tune them (see [`eval/`](eval/README.md))
if you change the model, chunking or corpus.

> **Warning:** leave `ADAPTIVE_USE_LLM=false` unless you re-tune the thresholds with it on. In the
> benchmark, LLM query rewriting made unanswerable questions look answerable (correct abstention fell from 0.58 to 0.11).

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/manuals/upload` | upload a PDF; returns manual metadata |
| GET | `/api/manuals` | list manuals |
| DELETE | `/api/manuals/{id}` | remove a manual |
| POST | `/api/chat` | ask a question; returns answer, citations and retrieval diagnostics |
| GET | `/api/conversations?manual_id=` | list conversations for a manual |
| GET | `/api/conversations/{id}` | full message history |
| DELETE | `/api/conversations/{id}` | delete a conversation |
| GET | `/api/health` | health check |

```jsonc
// POST /api/chat  request
{ "manual_id": "…", "question": "How often should the tires be rotated?" }

// response
{
  "conversation_id": "…",
  "answer": "According to page 251: tires should be rotated every 10000 km (6000 miles).",
  "sources": [{ "page": 251, "snippet": "Tire Rotation 4-tire rotation … Tires should be rotated every 10000 km …" }],
  "retrieval_mode": "M2",     // M0 exact code · M1 keyword · M2 hybrid · M3 iterative
  "sufficiency": 1.0,         // how well the evidence supports an answer, 0..1
  "abstained": false          // true → the assistant declined; sources are the closest pages
}
```

## Evaluation

The adaptive layer is evaluated against static baselines on two public owner's manuals (BMW M4,
Suzuki Ciaz) with a 71-question golden set (38 answerable, 33 unanswerable). Gold pages are derived from
evidence phrases rather than hand-typed. Weights and thresholds are tuned on a dev split and reported on test.

Test split, 20 answerable and 19 unanswerable queries, 95% bootstrap CIs:

| System | Recall@5 | MRR | correct abstention | false abstention |
|---|---|---|---|---|
| BM25 (stemmed) | **0.95** [0.85, 1.00] | 0.76 | n/a | n/a |
| Dense | 0.90 [0.75, 1.00] | 0.66 | n/a | n/a |
| Static hybrid (RRF) | 0.80 [0.60, 0.95] | 0.70 | n/a | n/a |
| **ADAPT-Diag** | 0.90 [0.75, 1.00] | 0.72 | 0.58 [0.37, 0.79] | 0.10 [0.00, 0.25] |

**What this does and does not show:**

- The adaptive pipeline is competitive with the baselines but **not significantly better**. The
  paired Recall@5 difference against the static hybrid is +0.10, with a 95% CI of [0.00, +0.25].
- Its main contribution is **abstention**, which the baselines cannot do at all. It is still
  imperfect: it declines 58% of unanswerable questions and wrongly declines 10% of answerable ones.
- The benchmark is small and single-annotator, and the test split is not a clean blind test.
  Trouble-code lookup and the context encoder are unit-tested but not benchmarked.

Method, ablations, deviations from the paper and limitations: [`eval/README.md`](eval/README.md).

## Roadmap

- Fresh, independently written benchmark questions and a second annotator
- A real OBD-II trouble-code corpus to benchmark exact lookup and the context encoder
- Answer-quality metrics (faithfulness, root-cause accuracy) and latency
- Joint tuning of thresholds with the query rewriter
- Streaming responses (SSE), OCR for scanned PDFs, cross-encoder re-ranking
- Authentication and multi-tenant manual access

## Notes

Owner's manuals are copyrighted. This repository does not include them; add your own PDFs through
the UI or `eval/data/manuals/`. The assistant supports a qualified technician and does not replace
one: for safety-critical systems always have the vehicle inspected.
