from __future__ import annotations

import re

from src.dataset.schema import Chunk, Document


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_chars: int = 80,
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")

    text = normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]
        split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("다. "))
        if split_at > min_chunk_chars and end < len(text):
            end = start + split_at + 1
            window = text[start:end]
        if len(window.strip()) >= min_chunk_chars:
            chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for idx, text in enumerate(
            split_text(document.text, chunk_size, chunk_overlap, min_chunk_chars)
        ):
            chunk_id = f"{document.doc_id}::chunk-{idx:04d}"
            metadata = {
                **document.metadata,
                "source_path": document.path,
                "chunk_index": str(idx),
            }
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=document.doc_id, text=text, metadata=metadata))
    return chunks
