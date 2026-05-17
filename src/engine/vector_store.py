from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from src.dataset.schema import Chunk


COLLECTION_NAME = "rfp_chunks"


class ChromaVectorStore:
    def __init__(
        self,
        collection: Any,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.collection = collection
        self.chunks = chunks
        self.embeddings = embeddings

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> "ChromaVectorStore":
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "row_idx": idx,
            }
            for idx, chunk in enumerate(chunks)
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return cls(collection=collection, chunks=chunks, embeddings=embeddings)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)

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

        if self.embeddings is None:
            raise ValueError("Cannot save ChromaVectorStore because embeddings are missing.")

        client = chromadb.PersistentClient(path=str(index_dir))

        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        collection.add(
            ids=[chunk.chunk_id for chunk in self.chunks],
            documents=[chunk.text for chunk in self.chunks],
            metadatas=[
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "row_idx": idx,
                }
                for idx, chunk in enumerate(self.chunks)
            ],
            embeddings=self.embeddings,
        )

    @classmethod
    def load(cls, index_dir: Path) -> "ChromaVectorStore":
        payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload]

        client = chromadb.PersistentClient(path=str(index_dir))
        collection = client.get_collection(name=COLLECTION_NAME)

        return cls(collection=collection, chunks=chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        search_results: list[tuple[Chunk, float]] = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            row_idx = int(metadata["row_idx"])
            score = 1.0 - float(distance)
            search_results.append((self.chunks[row_idx], score))

        return search_results
