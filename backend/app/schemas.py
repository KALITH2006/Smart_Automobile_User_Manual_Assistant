import datetime as dt
from pydantic import BaseModel


class ManualOut(BaseModel):
    id: str
    filename: str
    vehicle_name: str
    page_count: int
    chunk_count: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


class Source(BaseModel):
    page: int
    snippet: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: list[Source] = []
    created_at: dt.datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: str
    manual_id: str
    title: str
    created_at: dt.datetime

    class Config:
        from_attributes = True


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut] = []


class ChatRequest(BaseModel):
    manual_id: str
    question: str
    conversation_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[Source]
    # Adaptive retrieval diagnostics (None when ADAPTIVE_RAG is off)
    retrieval_mode: str | None = None  # M0 exact DTC lookup | M1 sparse | M2 hybrid | M3 iterative
    sufficiency: float | None = None
    abstained: bool = False
