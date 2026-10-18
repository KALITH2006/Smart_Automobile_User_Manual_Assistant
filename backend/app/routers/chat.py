import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, Manual, Conversation, Message
from app.schemas import ChatRequest, ChatResponse, ConversationOut, ConversationDetailOut, Source
from app.config import get_settings
from app.rag_pipeline import get_pipeline

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    manual = db.get(Manual, req.manual_id)
    if not manual:
        raise HTTPException(404, "Manual not found. Upload it first.")

    if req.conversation_id:
        conversation = db.get(Conversation, req.conversation_id)
        if not conversation:
            raise HTTPException(404, "Conversation not found.")
    else:
        title = req.question.strip()[:60] or "New conversation"
        conversation = Conversation(manual_id=manual.id, title=title)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Build chat history as (question, answer) tuples for context
    history = []
    prior_msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
        .all()
    )
    pending_q = None
    for m in prior_msgs:
        if m.role == "user":
            pending_q = m.content
        elif m.role == "assistant" and pending_q is not None:
            history.append((pending_q, m.content))
            pending_q = None

    pipeline = get_pipeline()
    meta = None
    try:
        if get_settings().ADAPTIVE_RAG:
            answer_text, sources, meta = pipeline.answer_adaptive(manual.collection_name, req.question, history)
        else:
            answer_text, sources = pipeline.answer(manual.collection_name, req.question, history)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate answer: {e}")

    user_msg = Message(conversation_id=conversation.id, role="user", content=req.question)
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer_text,
        sources=json.dumps(sources),
    )
    db.add_all([user_msg, assistant_msg])
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        answer=answer_text,
        sources=[Source(**s) for s in sources],
        retrieval_mode=meta.mode if meta else None,
        sufficiency=meta.sufficiency if meta else None,
        abstained=meta.abstained if meta else False,
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(manual_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(Conversation)
    if manual_id:
        q = q.filter(Conversation.manual_id == manual_id)
    return q.order_by(Conversation.created_at.desc()).all()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    convo = db.get(Conversation, conversation_id)
    if not convo:
        raise HTTPException(404, "Conversation not found.")

    out_messages = []
    for m in convo.messages:
        sources = json.loads(m.sources) if m.sources else []
        out_messages.append(
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": sources,
                "created_at": m.created_at,
            }
        )
    return {
        "id": convo.id,
        "manual_id": convo.manual_id,
        "title": convo.title,
        "created_at": convo.created_at,
        "messages": out_messages,
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    convo = db.get(Conversation, conversation_id)
    if not convo:
        raise HTTPException(404, "Conversation not found.")
    db.delete(convo)
    db.commit()
    return {"deleted": conversation_id}
