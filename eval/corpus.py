"""Load the evaluation manuals and chunk them with the same settings as the backend."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA = Path(__file__).parent / "data"
MANUALS = {"bmw": "BMW", "suzuki_ciaz": "Suzuki Ciaz"}

CHUNK_SIZE = 1000  # keep in sync with backend/app/config.py
CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    manual: str
    page: int  # 1-indexed PDF page, same convention as the backend citations
    text: str


def load_chunks(manual: str) -> list[Chunk]:
    pages = json.loads((DATA / f"{manual}_pages.json").read_text(encoding="utf8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [
        Chunk(manual, i + 1, piece)
        for i, text in enumerate(pages)
        for piece in splitter.split_text(text)
        if piece.strip()
    ]
