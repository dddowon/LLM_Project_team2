from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from src.dataset.schema import Chunk
from src.engine.doc_scope import build_doc_id_lookup, resolve_doc_filter_value


COLLECTION_NAME = "rfp_chunks"
# Chroma Rust backend enforces a max records-per-add batch (often ~5k–6k).
DEFAULT_CHROMA_ADD_BATCH_SIZE = 4096


def _add_to_collection_in_batches(
    collection: Any,
    *,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch_size: int = DEFAULT_CHROMA_ADD_BATCH_SIZE,
) -> None:
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        raise ValueError("ids, documents, metadatas, embeddings length mismatch")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


class ChromaVectorStore:
    def __init__(
        self,
        collection: Any,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
        *,
        doc_id_lookup: dict[str, str] | None = None,
    ) -> None:
        self.collection = collection
        self.chunks = chunks
        self.embeddings = embeddings
        self._doc_id_lookup = doc_id_lookup or build_doc_id_lookup(chunks)

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
        metadatas = []
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "row_idx": idx,
                }
            )
            metadatas.append(metadata)

        _add_to_collection_in_batches(
            collection,
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

        metadatas = []
        for idx, chunk in enumerate(self.chunks):
            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "row_idx": idx,
                }
            )
            metadatas.append(metadata)

        _add_to_collection_in_batches(
            collection,
            ids=[chunk.chunk_id for chunk in self.chunks],
            documents=[chunk.text for chunk in self.chunks],
            metadatas=metadatas,
            embeddings=self.embeddings,
        )

    @classmethod
    def load(cls, index_dir: Path) -> "ChromaVectorStore":
        payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload]

        client = chromadb.PersistentClient(path=str(index_dir))
        collection = client.get_collection(name=COLLECTION_NAME)

        return cls(collection=collection, chunks=chunks)

    def resolve_doc_id(self, doc_id: str | None) -> str | None:
        if not doc_id:
            return None
        return resolve_doc_filter_value(doc_id, self._doc_id_lookup)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }
        resolved_doc = self.resolve_doc_id(doc_id)
        if resolved_doc:
            query_kwargs["where"] = {
                "$or": [
                    {"file_name": {"$eq": resolved_doc}},
                    {"doc_id": {"$eq": resolved_doc}},
                ]
            }

        results = self.collection.query(**query_kwargs)

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        search_results: list[tuple[Chunk, float]] = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            row_idx = int(metadata["row_idx"])
            score = 1.0 - float(distance)
            search_results.append((self.chunks[row_idx], score))

        return search_results
