"""
Lightweight SQLite persistence layer for:
- Manual metadata (uploaded PDFs -> vector collection mapping)
- Conversations
- Messages (with source-citation JSON attached to assistant replies)
"""
import datetime as dt
import os
import uuid

from sqlalchemy import create_engine, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.config import get_settings

settings = get_settings()
os.makedirs(os.path.dirname(settings.SQLITE_PATH), exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.SQLITE_PATH}", connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Manual(Base):
    __tablename__ = "manuals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String)
    vehicle_name: Mapped[str] = mapped_column(String, default="Unknown Vehicle")
    collection_name: Mapped[str] = mapped_column(String)
    page_count: Mapped[int] = mapped_column(default=0)
    chunk_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="manual")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    manual_id: Mapped[str] = mapped_column(ForeignKey("manuals.id"))
    title: Mapped[str] = mapped_column(String, default="New conversation")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    manual: Mapped["Manual"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    # JSON-encoded list of {page, snippet} citation objects (assistant messages only)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
