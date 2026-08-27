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
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import get_settings

settings = get_settings()

print("LLM_PROVIDER =", settings.LLM_PROVIDER)
print("OLLAMA MODEL =", settings.OLLAMA_CHAT_MODEL)

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


_pipeline_singleton: ManualRAGPipeline | None = None


def get_pipeline() -> ManualRAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = ManualRAGPipeline()
    return _pipeline_singleton
