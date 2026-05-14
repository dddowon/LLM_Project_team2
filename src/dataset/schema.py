from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Document:
    doc_id: str
    path: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
