from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.dataset.schema import Chunk


class FaissVectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk]) -> None:
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: list[list[float]]) -> "FaissVectorStore":
        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunks=chunks)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]
        (index_dir / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "FaissVectorStore":
        index = faiss.read_index(str(index_dir / "index.faiss"))
        payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload]
        return cls(index=index, chunks=chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        vector = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results
