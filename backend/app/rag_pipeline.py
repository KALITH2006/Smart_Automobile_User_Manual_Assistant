"""
RAG pipeline for the AI Vehicle Manual Assistant.

Flow:
  PDF -> PyPDFLoader (per-page docs) -> RecursiveCharacterTextSplitter (chunks,
  page metadata preserved) -> embeddings -> Chroma persistent collection
  (one collection per uploaded manual) -> similarity search at query time ->
  LLM answers ONLY from retrieved chunks -> structured citations (page + snippet)
  are returned alongside the answer.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.adaptive import AdaptiveConfig, DiagnosticQuery, Router, chroma_searcher, load_dtc_table, run_adaptive
from app.adaptive.llm import make_classifier, make_rewriter, safety_notice
from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert automotive technician and the official AI \
assistant for reading vehicle owner's manuals. Answer the user's question using \
ONLY the manual excerpts provided in the context below.

Rules:
- If the context does not contain the answer, say clearly that the manual does \
not appear to cover that, and suggest the closest related topic if one exists \
in the context.
- Be precise and step-by-step for procedures (e.g. resetting TPMS, checking oil).
- Always write safety warnings from the manual verbatim in your own words if present.
- Do not invent page numbers, part names, or specifications that are not in the context.
- Keep answers concise but complete: use numbered steps for procedures.

Context from the manual (each excerpt is tagged with its page number):
{context}

Conversation history so far:
{chat_history}
"""

USER_PROMPT = "Question: {question}"


def _get_embeddings():
    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBEDDING_MODEL
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_EMBEDDING_MODEL
    )


def _get_llm():
    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_CHAT_MODEL, temperature=0.1)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_CHAT_MODEL, temperature=0.1
    )


@dataclass
class AdaptiveMeta:
    mode: str
    sufficiency: float
    abstained: bool


ABSTAIN_MESSAGE = (
    "I could not find enough support for that in this manual, so I will not guess. "
    "Try rephrasing with the part or warning light name, or check that this is the right manual. "
    "For anything safety-related, please have a qualified technician inspect the vehicle."
)


def collection_name_for(file_bytes: bytes, filename: str) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    safe_name = re.sub(r"[^a-zA-Z0-9]", "-", filename.lower())[:30]
    return f"manual-{safe_name}-{digest}"


class ManualRAGPipeline:
    """Handles ingestion and Q&A for a single vehicle manual collection."""

    def __init__(self):
        self.embeddings = _get_embeddings()
        self.llm = _get_llm()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ---------- Ingestion ----------

    def ingest_pdf(self, pdf_path: str, collection_name: str) -> tuple[int, int]:
        """Load, split, and embed a PDF into its own Chroma collection.

        Returns (page_count, chunk_count).
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # one Document per page, metadata["page"] is 0-indexed

        chunks: list[Document] = []
        for page_doc in pages:
            page_num = page_doc.metadata.get("page", 0) + 1  # human-friendly 1-indexed
            for chunk in self.splitter.split_documents([page_doc]):
                chunk.metadata["page"] = page_num
                chunks.append(chunk)

        store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DIR,
        )
        if chunks:
            store.add_documents(chunks)
        self.forget_collection(collection_name)  # drop any stale BM25/embedding cache

        return len(pages), len(chunks)

    # ---------- Retrieval + Answering ----------

    def _get_store(self, collection_name: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DIR,
        )

    def answer(
        self,
        collection_name: str,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> tuple[str, list[dict]]:
        """Retrieve relevant chunks and answer, returning (answer_text, sources).

        sources is a list of {"page": int, "snippet": str} deduplicated by page.
        """
        store = self._get_store(collection_name)
        retriever = store.as_retriever(search_kwargs={"k": settings.RETRIEVER_TOP_K})
        docs: list[Document] = retriever.invoke(question)

        if not docs:
            return (
                "I couldn't find anything relevant to that in the uploaded manual. "
                "Try rephrasing, or check that this is the right manual.",
                [],
            )

        context_blocks = []
        seen_pages = {}
        for d in docs:
            page = d.metadata.get("page", "?")
            context_blocks.append(f"[Page {page}]\n{d.page_content}")
            if page not in seen_pages:
                snippet = d.page_content.strip().replace("\n", " ")
                seen_pages[page] = snippet[:220] + ("..." if len(snippet) > 220 else "")

        context = "\n\n---\n\n".join(context_blocks)

        history_text = "None"
        if chat_history:
            recent = chat_history[-settings.MAX_HISTORY_TURNS :]
            history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )
        chain = prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke(
            {"context": context, "chat_history": history_text, "question": question}
        )

        sources = [
            {"page": page, "snippet": snippet}
            for page, snippet in sorted(seen_pages.items(), key=lambda x: x[0] if isinstance(x[0], int) else 0)
        ]
        return answer_text, sources


    # ---------- Adaptive retrieval (ADAPT-Diag) ----------

    def _adaptive_config(self) -> AdaptiveConfig:
        return AdaptiveConfig(
            n_evidence=settings.RETRIEVER_TOP_K,
            t_max=settings.ADAPTIVE_T_MAX,
            alpha=settings.ADAPTIVE_ALPHA,
            beta=settings.ADAPTIVE_BETA,
            gamma=settings.ADAPTIVE_GAMMA,
            tau=settings.ADAPTIVE_TAU,
            tau_m1=settings.ADAPTIVE_TAU_M1,
        )

    def _get_searcher(self, collection_name: str):
        # BM25 index + stored embeddings are built once per manual and reused across requests
        cache = self.__dict__.setdefault("_searchers", {})
        if collection_name not in cache:
            cache[collection_name] = chroma_searcher(self._get_store(collection_name), self.embeddings.embed_query)
        return cache[collection_name]

    def forget_collection(self, collection_name: str) -> None:
        self.__dict__.setdefault("_searchers", {}).pop(collection_name, None)

    def answer_adaptive(
        self,
        collection_name: str,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> tuple[str, list[dict], AdaptiveMeta]:
        """Route, retrieve, check sufficiency, then answer or abstain when the evidence is too weak."""
        cfg = self._adaptive_config()
        use_llm = settings.ADAPTIVE_USE_LLM
        dtc = load_dtc_table()
        outcome = run_adaptive(
            DiagnosticQuery(question),
            self._get_searcher(collection_name),
            cfg=cfg,
            router=Router(dtc, make_classifier(self.llm) if use_llm else None, cfg.theta_r),
            dtc_table=dtc,
            rewriter=make_rewriter(self.llm) if use_llm else None,
        )
        logger.info(
            "adaptive route=%s final_mode=%s S=%.3f abstained=%s rounds=%d llm_calls=%d",
            outcome.route.mode.value, outcome.mode.value, outcome.sufficiency,
            outcome.abstained, len(outcome.steps), outcome.llm_calls,
        )
        meta = AdaptiveMeta(outcome.mode.value, round(outcome.sufficiency, 3), outcome.abstained)
        sources = self._sources_from(outcome.evidence)
        if outcome.abstained:
            return ABSTAIN_MESSAGE, sources, meta

        context = "\n\n---\n\n".join(
            f"[Page {h.page}]\n{h.text}" if h.page else f"[Code table]\n{h.text}" for h in outcome.evidence
        )
        history_text = "None"
        if chat_history:
            recent = chat_history[-settings.MAX_HISTORY_TURNS :]
            history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
        text = (prompt | self.llm | StrOutputParser()).invoke(
            {"context": context, "chat_history": history_text, "question": question}
        )
        notice = safety_notice(question, outcome.evidence)
        return (f"{text}\n\n> Warning: {notice}" if notice else text), sources, meta

    @staticmethod
    def _sources_from(evidence) -> list[dict]:
        seen: dict[int, str] = {}
        for h in evidence:
            if h.page and h.page not in seen:
                snippet = h.text.strip().replace("\n", " ")
                seen[h.page] = snippet[:220] + ("..." if len(snippet) > 220 else "")
        return [{"page": p, "snippet": sn} for p, sn in sorted(seen.items())]


_pipeline_singleton: ManualRAGPipeline | None = None


def get_pipeline() -> ManualRAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = ManualRAGPipeline()
    return _pipeline_singleton
