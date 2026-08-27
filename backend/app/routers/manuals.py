import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, Manual
from app.schemas import ManualOut
from app.rag_pipeline import get_pipeline, collection_name_for

router = APIRouter(prefix="/api/manuals", tags=["manuals"])
settings = get_settings()


@router.post("/upload", response_model=ManualOut)
async def upload_manual(
    file: UploadFile = File(...),
    vehicle_name: str = Form("Unknown Vehicle"),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty.")

    os.makedirs(settings.MANUALS_DIR, exist_ok=True)
    collection = collection_name_for(file_bytes, file.filename)
    save_path = os.path.join(settings.MANUALS_DIR, f"{collection}.pdf")

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    pipeline = get_pipeline()
    try:
        page_count, chunk_count = pipeline.ingest_pdf(save_path, collection)
    except Exception as e:
        raise HTTPException(500, f"Failed to process PDF: {e}")

    manual = Manual(
        filename=file.filename,
        vehicle_name=vehicle_name,
        collection_name=collection,
        page_count=page_count,
        chunk_count=chunk_count,
    )
    db.add(manual)
    db.commit()
    db.refresh(manual)
    return manual


@router.get("", response_model=list[ManualOut])
def list_manuals(db: Session = Depends(get_db)):
    return db.query(Manual).order_by(Manual.created_at.desc()).all()


@router.delete("/{manual_id}")
def delete_manual(manual_id: str, db: Session = Depends(get_db)):
    manual = db.get(Manual, manual_id)
    if not manual:
        raise HTTPException(404, "Manual not found.")
    db.delete(manual)
    db.commit()
    return {"deleted": manual_id}
