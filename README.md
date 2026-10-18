# 🚗 AI Vehicle Manual Assistant

Upload a vehicle owner's manual (PDF) and ask it questions in plain English —
*"How do I reset the tire pressure?"* — and get a direct, step-by-step answer
with the **exact page number(s)** it came from. This is the pattern used by
in-house support tools at Toyota, BMW, Mercedes, HARMAN, and Bosch: a
retrieval-augmented generation (RAG) assistant scoped tightly to authoritative
source documents, so it never invents specs or procedures.

## How it works (architecture)

```
┌─────────────┐      upload PDF       ┌──────────────────────────────┐
│   Next.js    │ ─────────────────────▶│           FastAPI             │
│  Chat UI     │                        │  ┌─────────────────────────┐ │
│ (React/TSX)  │◀───────────────────────│  │  1. PyPDFLoader          │ │
└─────────────┘   answer + citations   │  │     -> pages              │ │
                                        │  │  2. RecursiveCharacter    │ │
                                        │  │     TextSplitter (chunks, │ │
                                        │  │     page metadata kept)   │ │
                                        │  │  3. Embeddings            │ │
                                        │  │     -> ChromaDB           │ │
                                        │  │     (1 collection/manual) │ │
                                        │  │  4. Similarity search     │ │
                                        │  │     (top-k chunks)        │ │
                                        │  │  5. LLM answers ONLY from │ │
                                        │  │     retrieved context     │ │
                                        │  │  6. Citations = source    │ │
                                        │  │     page + snippet        │ │
                                        │  └─────────────────────────┘ │
                                        │       SQLite (conversations,  │
                                        │       messages, manual index) │
                                        └──────────────────────────────┘
```

**Why this design:**
- **One Chroma collection per manual** (keyed by a hash of the file bytes) —
  re-uploading the same PDF is idempotent and different vehicles never
  cross-contaminate answers.
- **Page metadata survives chunking** — every chunk knows which page it came
  from, so citations are exact, not approximate.
- **The system prompt forces "answer only from context"** — this is what
  keeps the assistant from hallucinating torque specs or safety procedures
  that aren't in the manual.
- **Conversation history is persisted in SQLite** and the last few turns are
  fed back to the LLM, so follow-up questions ("what about the rear tires?")
  work.

## Features

- 📄 PDF upload & automatic ingestion (chunking + embeddings)
- 🔍 Semantic search over the manual (ChromaDB vector store)
- 💬 Chat interface with streaming-style UX and Markdown answers
- 📌 Source citations (page number + snippet) shown inline, expandable
- 🗂️ Multiple manuals supported, switch between vehicles
- 🕘 Persistent conversation history per manual
- 🐳 One-command Docker Compose deployment
- 🔁 Pluggable LLM backend: **OpenAI** (cloud) or **Ollama/Llama** (fully local, no API key)

## Tech stack

| Layer          | Choice                                              |
|----------------|------------------------------------------------------|
| Backend API    | FastAPI (Python)                                      |
| RAG framework  | LangChain (loaders, splitters, prompt chains)         |
| Vector store   | ChromaDB (persistent, on-disk)                        |
| LLM/Embeddings | OpenAI (`gpt-4o-mini`) or Llama via Ollama (local)     |
| Database       | SQLite via SQLAlchemy (manuals, conversations, messages) |
| Frontend       | Next.js 14 (App Router) + React + Tailwind CSS         |
| Deployment     | Docker + Docker Compose                                |

## Project structure

```
vehicle-manual-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + CORS + startup
│   │   ├── config.py          # env-driven settings (OpenAI/Ollama toggle)
│   │   ├── database.py        # SQLAlchemy models (Manual, Conversation, Message)
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── rag_pipeline.py    # PDF ingest, chunking, Chroma, retrieval + answer
│   │   └── routers/
│   │       ├── manuals.py     # /api/manuals endpoints
│   │       └── chat.py        # /api/chat, /api/conversations endpoints
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # main page (sidebar + chat)
│   │   ├── layout.tsx
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   └── FileUpload.tsx
│   │   └── lib/api.ts         # typed fetch client for the backend
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Quick start

### Prerequisites

- Python 3.11 or newer
- Node.js 18 or newer and npm
- Docker Desktop, if using Docker Compose
- An LLM provider: OpenAI, or Ollama for local models

The frontend runs on port `3000` and the backend runs on port `8000`.

### Option A - Docker Compose

From the repository root, create the backend environment file and edit it:

**Windows PowerShell:**
```powershell
Copy-Item backend/.env.example backend/.env
notepad backend/.env
```

**macOS/Linux:**
```bash
cp backend/.env.example backend/.env
nano backend/.env
```

For OpenAI, set `LLM_PROVIDER=openai` and provide a valid
`OPENAI_API_KEY`. To use Ollama in Docker, leave `LLM_PROVIDER=ollama` and
start the Ollama profile:

```powershell
docker compose --profile ollama up --build
```

In a second terminal, download the models required by the default Ollama
configuration:

```powershell
docker exec -it manual-assistant-ollama ollama pull llama3.1
docker exec -it manual-assistant-ollama ollama pull nomic-embed-text
```

For OpenAI, start the regular services instead:

```powershell
docker compose up --build
```

Open **http://localhost:3000** after the containers finish starting. The
backend health check is **http://localhost:8000/api/health** and the API docs
are **http://localhost:8000/docs**.

Stop the services with `Ctrl+C`, or from another terminal run:

```powershell
docker compose down
```

### Option B - Run locally without Docker

#### 1. Configure the backend

**Windows PowerShell:**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

If PowerShell blocks activation, run this once in PowerShell as your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
nano .env
```

Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` for OpenAI, or keep
`LLM_PROVIDER=ollama` and make sure Ollama is running locally with the models
listed in `.env.example`.

Start the backend from the `backend` directory:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Configure and start the frontend

Open a second terminal at the repository root:

**Windows PowerShell:**
```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

**macOS/Linux:**
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

The frontend expects `NEXT_PUBLIC_API_URL=http://localhost:8000` in
`.env.local`. Open **http://localhost:3000**. Verify the backend first at
**http://localhost:8000/api/health** if the frontend shows `Failed to fetch`.

Do not commit `.env`, `.env.local`, API keys, or downloaded model files.

## API reference (backend)

| Method | Endpoint                          | Description                                  |
|--------|------------------------------------|-----------------------------------------------|
| POST   | `/api/manuals/upload`              | Upload a PDF, returns manual metadata          |
| GET    | `/api/manuals`                     | List all uploaded manuals                      |
| DELETE | `/api/manuals/{id}`                | Remove a manual                                |
| POST   | `/api/chat`                        | Ask a question, returns answer + page citations|
| GET    | `/api/conversations?manual_id=`    | List conversations for a manual                |
| GET    | `/api/conversations/{id}`          | Full message history for a conversation        |
| DELETE | `/api/conversations/{id}`          | Delete a conversation                          |
| GET    | `/api/health`                      | Health check                                   |

Example `POST /api/chat` request/response:

```json
// request
{ "manual_id": "…", "question": "How do I reset the tire pressure?" }

// response
{
  "conversation_id": "…",
  "answer": "1. Turn the ignition to ON without starting the engine.\n2. Press and hold the TPMS reset button until the light blinks twice.\n3. Start the engine and drive for 10 minutes to recalibrate.",
  "sources": [
    { "page": 4, "snippet": "To reset the tire pressure monitoring system (TPMS): 1. Turn ignition to ON position..." }
  ]
}
```

## Extending this for production

This starter is deliberately scoped to be a clean, correct baseline. Natural
next steps for a production deployment (the kind Toyota/BMW/Bosch-style teams
would add):

- **Multi-tenant auth** — JWT/OAuth so each dealership or fleet only sees its own manuals
- **OCR fallback** for scanned (image-only) manual PDFs, e.g. via Tesseract
- **Streaming responses** (SSE) instead of a single blocking chat call
- **Re-ranking** retrieved chunks (e.g. Cohere rerank or a cross-encoder) for higher precision
- **Multi-language support** for international manuals
- **Analytics** on unanswered/low-confidence questions to find manual content gaps
- **VIN-based manual auto-selection** instead of manual upload
- **Guardrails/eval suite** to continuously test citation accuracy against ground truth
#   S m a r t _ A u t o m o b i l e _ U s e r _ M a n u a l _ A s s i s t a n t  
 