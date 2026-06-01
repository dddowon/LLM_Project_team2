from __future__ import annotations

from src.config import RetrievalConfig
from src.dataset.schema import Chunk
from src.engine.hybrid_search import BM25Index, HybridSearcher


class FakeVectorStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def resolve_doc_id(self, doc_id: str | None) -> str | None:
        return doc_id

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        selected = [
            chunk
            for chunk in self.chunks
            if doc_id is None or chunk.metadata.get("file_name") == doc_id
        ]
        return [(chunk, 1.0 - (index * 0.1)) for index, chunk in enumerate(selected[:top_k])]


def test_bm25_matches_exact_requirement_id_and_preserves_chunk_id() -> None:
    chunks = [
        Chunk(
            chunk_id="slim_chunk_00000001",
            doc_id="doc-a.hwp",
            text="일반 사업 개요입니다.",
            metadata={"file_name": "doc-a.hwp"},
        ),
        Chunk(
            chunk_id="slim_chunk_00000002",
            doc_id="doc-a.hwp",
            text="SFR-005 요구사항은 사용자 로그인 기능 개선을 포함한다.",
            metadata={"file_name": "doc-a.hwp", "section_path_text": "요구사항"},
        ),
    ]

    index = BM25Index(chunks, doc_resolver=FakeVectorStore(chunks))
    results = index.search("SFR-005 요구사항", 5, doc_id="doc-a.hwp")

    assert results
    assert results[0].chunk.chunk_id == "slim_chunk_00000002"


def test_bm25_respects_doc_id_filter() -> None:
    chunks = [
        Chunk(
            chunk_id="slim_chunk_00000001",
            doc_id="doc-a.hwp",
            text="사업비는 157,300,000원입니다.",
            metadata={"file_name": "doc-a.hwp"},
        ),
        Chunk(
            chunk_id="slim_chunk_00000002",
            doc_id="doc-b.hwp",
            text="사업비는 157,300,000원입니다.",
            metadata={"file_name": "doc-b.hwp"},
        ),
    ]

    index = BM25Index(chunks, doc_resolver=FakeVectorStore(chunks))
    results = index.search("157,300,000원", 5, doc_id="doc-b.hwp")

    assert [result.chunk.chunk_id for result in results] == ["slim_chunk_00000002"]


def test_hybrid_search_uses_existing_chunk_ids() -> None:
    chunks = [
        Chunk(
            chunk_id="slim_chunk_00000001",
            doc_id="doc-a.hwp",
            text="일반 사업 개요입니다.",
            metadata={"file_name": "doc-a.hwp"},
        ),
        Chunk(
            chunk_id="slim_chunk_00000002",
            doc_id="doc-a.hwp",
            text="DAR-008 데이터 이관 요구사항입니다.",
            metadata={"file_name": "doc-a.hwp"},
        ),
    ]
    config = RetrievalConfig(top_k=2, score_threshold=0.0)
    config.hybrid.enabled = True
    config.hybrid.dense_top_k = 2
    config.hybrid.bm25_top_k = 2
    config.reranker.enabled = False

    searcher = HybridSearcher(FakeVectorStore(chunks), config)
    results = searcher.search("DAR-008", [0.1, 0.2], doc_id="doc-a.hwp")

    assert {chunk.chunk_id for chunk, _score in results} == {
        "slim_chunk_00000001",
        "slim_chunk_00000002",
    }
