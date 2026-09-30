"""Step 3 - Meaningful chunking (sentence-aware, configurable size + overlap)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .document_loader import LoadedDocument
from .text_cleaner import clean_text


@dataclass
class Chunk:
    chunk_id: str        # e.g. "syllabus.pdf::p3::c007"
    text: str
    source: str          # filename
    page: int | None
    index: int           # running number inside the document


def _split_units(text: str, max_len: int) -> list[str]:
    """Split into sentences / paragraphs; hard-split anything longer than max_len."""
    units: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        part = part.strip()
        while len(part) > max_len:
            units.append(part[:max_len].strip())
            part = part[max_len:].strip()
        if part:
            units.append(part)
    return units


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[str] = []
    current: list[str] = []
    for unit in _split_units(text, chunk_size):
        if current and len(" ".join(current)) + 1 + len(unit) > chunk_size:
            chunks.append(" ".join(current))
            carry: list[str] = []          # keep the last sentences as overlap
            carry_len = 0
            for sentence in reversed(current):
                if carry_len + len(sentence) + 1 > overlap:
                    break
                carry.insert(0, sentence)
                carry_len += len(sentence) + 1
            current = carry
            if current and len(" ".join(current)) + 1 + len(unit) > chunk_size:
                current = []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: LoadedDocument, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        for piece in split_text(cleaned, chunk_size, overlap):
            index = len(chunks)
            page_tag = f"p{page.page}" if page.page is not None else "doc"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.filename}::{page_tag}::c{index:03d}",
                    text=piece,
                    source=doc.filename,
                    page=page.page,
                    index=index,
                )
            )
    return chunks
