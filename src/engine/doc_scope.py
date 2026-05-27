from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dataset.schema import Chunk


def normalize_doc_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value).strip())
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def build_doc_id_lookup(chunks: list[Chunk]) -> dict[str, str]:
    """Map normalized doc keys to canonical metadata file_name for Chroma filters."""
    lookup: dict[str, str] = {}
    for chunk in chunks:
        meta = chunk.metadata or {}
        canonical = str(meta.get("file_name") or chunk.doc_id or "").strip()
        if not canonical:
            continue
        for candidate in (chunk.doc_id, meta.get("file_name"), meta.get("source_file")):
            if not candidate:
                continue
            lookup[normalize_doc_key(str(candidate))] = canonical
    return lookup


def resolve_doc_filter_value(doc_id: str, lookup: dict[str, str]) -> str | None:
    raw = str(doc_id).strip()
    if not raw:
        return None

    key = normalize_doc_key(raw)
    if key in lookup:
        return lookup[key]

    candidates: list[str] = []
    for norm_key, canonical in lookup.items():
        if key in norm_key or norm_key in key:
            candidates.append(canonical)
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]

    return raw


def doc_keys_match(expected_doc_id: str, retrieved_name: str) -> bool:
    if not expected_doc_id or not retrieved_name:
        return False
    left = normalize_doc_key(expected_doc_id)
    right = normalize_doc_key(retrieved_name)
    return left in right or right in left
