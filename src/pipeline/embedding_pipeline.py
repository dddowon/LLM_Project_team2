"""Embedding pipeline for Shinwoo chunk JSONL format.

Required input row fields:
- chunk_id, chunk_type, chunk_text, metadata

Output row fields:
- keep input fields and append embedding
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from src.utils.jsonl import read_jsonl, write_jsonl


def embed_prechunked_jsonl(
    input_path: Path,
    output_path: Path,
    model: str,
    batch_size: int = 64,
    force_real: bool = False,
) -> int:
    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"입력 JSONL이 비어있습니다: {input_path}")

    _validate_chunk_rows(rows)
    texts = [_extract_text(row) for row in rows]
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if force_real and not has_key:
        raise RuntimeError("force_real=True 인데 OPENAI_API_KEY가 없습니다.")
    use_mock = not has_key and not force_real

    if use_mock:
        embeddings = [_mock_embedding(text) for text in texts]
        embedding_source = "mock"
    else:
        from src.models.openai_client import OpenAIModelClient

        client = OpenAIModelClient()
        embeddings = client.embed_texts(texts=texts, model=model, batch_size=batch_size)
        embedding_source = "openai"

    output_rows = []
    for row, embedding in zip(rows, embeddings, strict=False):
        out = dict(row)
        metadata = dict(out.get("metadata", {})) if isinstance(out.get("metadata"), dict) else {}
        metadata["embedding_source"] = embedding_source
        out["metadata"] = metadata
        out["embedding"] = embedding
        output_rows.append(out)

    write_jsonl(output_path, output_rows)
    return len(output_rows)


def _validate_chunk_rows(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        for key in ("chunk_id", "chunk_type", "chunk_text"):
            if key not in row:
                raise ValueError(f"{idx}번째 row에 필수 필드가 없습니다: {key}")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError(f"{idx}번째 row의 metadata는 dict 여야 합니다.")


def _extract_text(row: dict[str, Any]) -> str:
    return str(row["chunk_text"])


def _mock_embedding(text: str, dim: int = 1536) -> list[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for i in range(dim):
        byte = seed[i % len(seed)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    return values
