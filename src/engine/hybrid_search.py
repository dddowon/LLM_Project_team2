from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from src.config import RetrievalConfig
from src.dataset.schema import Chunk


TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z]{2,}[-_/]?\d{1,5}              # SFR-005, DAR_008
    |\d{1,3}(?:,\d{3})+(?:원|천원|만원|억원)? # 157,300,000원
    |\d+(?:\.\d+)+(?:년|월|일|%)?          # 2024. 03., 1.2.3
    |[가-힣A-Za-z0-9]+(?:[-_/][가-힣A-Za-z0-9]+)*
    |[①-⑳]
    |[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]
    """,
    re.VERBOSE,
)


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    tokens = TOKEN_PATTERN.findall(normalized)
    expanded: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        expanded.append(token)
        compact = re.sub(r"[-_/,.\s]+", "", token)
        if compact and compact != token:
            expanded.append(compact)
    return expanded


def _metadata_text(metadata: dict[str, Any]) -> str:
    fields = (
        "file_name",
        "source_file",
        "section_path_text",
        "section_type",
        "heading",
        "content_type",
        "chunk_type",
        "table_id",
        "table_type",
        "row_range",
    )
    values: list[str] = []
    for field in fields:
        value = metadata.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, dict):
            values.extend(str(item) for item in value.values() if str(item).strip())
        else:
            values.append(str(value))
    return "\n".join(values)


def _chunk_search_text(chunk: Chunk) -> str:
    return "\n".join(
        part
        for part in (
            chunk.doc_id,
            _metadata_text(chunk.metadata or {}),
            chunk.text,
        )
        if str(part).strip()
    )


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class DocResolver(Protocol):
    def resolve_doc_id(self, doc_id: str | None) -> str | None: ...


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], *, doc_resolver: DocResolver) -> None:
        self.chunks = chunks
        self.doc_resolver = doc_resolver
        self.doc_tokens: list[list[str]] = []
        self.doc_token_counts: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.doc_frequency: Counter[str] = Counter()
        self.index_by_doc_key: dict[str, set[int]] = defaultdict(set)

        for index, chunk in enumerate(chunks):
            tokens = tokenize_for_bm25(_chunk_search_text(chunk))
            token_counts = Counter(tokens)
            self.doc_tokens.append(tokens)
            self.doc_token_counts.append(token_counts)
            self.doc_lengths.append(len(tokens))
            self.doc_frequency.update(token_counts.keys())
            self._index_doc_keys(index, chunk)

        self.document_count = len(chunks)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 0.0
        )

    def _index_doc_keys(self, index: int, chunk: Chunk) -> None:
        metadata = chunk.metadata or {}
        for value in (chunk.doc_id, metadata.get("file_name"), metadata.get("source_file")):
            if not value:
                continue
            self.index_by_doc_key[str(value)].add(index)

    def _candidate_indices(self, doc_id: str | None) -> Iterable[int]:
        resolved_doc = self.doc_resolver.resolve_doc_id(doc_id)
        if not resolved_doc:
            return range(len(self.chunks))
        return self.index_by_doc_key.get(resolved_doc, set())

    def search(self, query: str, top_k: int, *, doc_id: str | None = None) -> list[RankedChunk]:
        if top_k <= 0 or not self.chunks:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        query_terms = list(dict.fromkeys(query_tokens))
        scores: list[tuple[int, float]] = []
        for index in self._candidate_indices(doc_id):
            score = self._score_document(index, query_terms)
            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        selected = scores[:top_k]
        normalized = _normalize_scores([score for _, score in selected])
        return [
            RankedChunk(self.chunks[index], score)
            for (index, _raw_score), score in zip(selected, normalized, strict=False)
        ]

    def _score_document(self, index: int, query_terms: list[str]) -> float:
        k1 = 1.5
        b = 0.75
        token_counts = self.doc_token_counts[index]
        doc_length = self.doc_lengths[index]
        if not token_counts or not self.avg_doc_length:
            return 0.0

        score = 0.0
        for term in query_terms:
            term_frequency = token_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            document_frequency = self.doc_frequency.get(term, 0)
            idf = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1.0 - b + b * (doc_length / self.avg_doc_length)
            )
            score += idf * (term_frequency * (k1 + 1.0)) / denominator
        return score


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._model: Any | None = None

    @staticmethod
    def _resolve_device(device: str) -> str | None:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "reranker.enabled=true requires sentence-transformers. "
                'Install it with: pip install "sentence-transformers>=3.0.0"'
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(
        self,
        question: str,
        candidates: list[RankedChunk],
        *,
        top_k: int,
    ) -> list[RankedChunk]:
        if top_k <= 0 or not candidates:
            return []

        model = self._load_model()
        pairs = [(question, candidate.chunk.text) for candidate in candidates]
        raw_scores = model.predict(pairs)
        scores = _normalize_scores([float(score) for score in raw_scores])
        reranked = sorted(
            (
                RankedChunk(candidate.chunk, score)
                for candidate, score in zip(candidates, scores, strict=False)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return reranked[:top_k]


class HybridSearcher:
    def __init__(self, vector_store: Any, config: RetrievalConfig) -> None:
        self.vector_store = vector_store
        self.config = config
        self.bm25_index = BM25Index(vector_store.chunks, doc_resolver=vector_store)
        self.reranker: CrossEncoderReranker | None = None
        if config.reranker.enabled:
            self.reranker = CrossEncoderReranker(
                config.reranker.model,
                device=config.reranker.device,
            )

    def search(
        self,
        question: str,
        query_embedding: list[float],
        *,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self.config.hybrid.enabled:
            candidates = self._hybrid_candidates(question, query_embedding, doc_id=doc_id)
        else:
            dense_results = self.vector_store.search(
                query_embedding,
                max(self.config.reranker.top_k, self.config.top_k)
                if self.config.reranker.enabled
                else self.config.top_k,
                doc_id=doc_id,
            )
            candidates = [RankedChunk(chunk, score) for chunk, score in dense_results]

        if self.reranker is not None:
            candidates = candidates[: max(self.config.top_k, self.config.reranker.top_k)]
            candidates = self.reranker.rerank(
                question,
                candidates,
                top_k=self.config.top_k,
            )

        return [(item.chunk, item.score) for item in candidates[: self.config.top_k]]

    def _hybrid_candidates(
        self,
        question: str,
        query_embedding: list[float],
        *,
        doc_id: str | None,
    ) -> list[RankedChunk]:
        dense_results = self.vector_store.search(
            query_embedding,
            self.config.hybrid.dense_top_k,
            doc_id=doc_id,
        )
        bm25_results = self.bm25_index.search(
            question,
            self.config.hybrid.bm25_top_k,
            doc_id=doc_id,
        )

        chunk_by_id: dict[str, Chunk] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        self._add_rrf_scores(
            ((chunk, score) for chunk, score in dense_results),
            rrf_scores=rrf_scores,
            chunk_by_id=chunk_by_id,
        )
        self._add_rrf_scores(
            ((item.chunk, item.score) for item in bm25_results),
            rrf_scores=rrf_scores,
            chunk_by_id=chunk_by_id,
        )

        ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        normalized = _normalize_scores([score for _, score in ranked])
        return [
            RankedChunk(chunk_by_id[chunk_id], score)
            for (chunk_id, _raw_score), score in zip(ranked, normalized, strict=False)
        ]

    def _add_rrf_scores(
        self,
        results: Iterable[tuple[Chunk, float]],
        *,
        rrf_scores: dict[str, float],
        chunk_by_id: dict[str, Chunk],
    ) -> None:
        for rank, (chunk, _score) in enumerate(results, start=1):
            chunk_by_id.setdefault(chunk.chunk_id, chunk)
            rrf_scores[chunk.chunk_id] += 1.0 / (self.config.hybrid.rrf_k + rank)
