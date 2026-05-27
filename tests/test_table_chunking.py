from argparse import Namespace

from src.chunking.chunking import build_rag_chunks, compact_output_metadata
from src.pipeline.hwp_slim_pipeline import build_table_raw_rows


def _chunk_args() -> Namespace:
    return Namespace(
        text_chunk_size=900,
        text_overlap=120,
        table_chunk_size=1000,
        max_table_rows=6,
        min_text_chars=40,
        short_context_chars=40,
        include_cover=True,
        include_toc=False,
    )


def test_table_chunks_are_emitted_with_text_chunks_and_next_table_marker() -> None:
    records = [
        {
            "file_name": "sample.hwp",
            "content_type": "section_text",
            "section_path": ["III. 제안요청내용", "2. 요구사항 세부내용"],
            "section_type": "requirements",
            "heading": "2. 요구사항 세부내용",
            "text": (
                "다음 표는 홈페이지 디자인 요구사항을 설명한다. "
                "요구사항 고유번호, 명칭, 정의, 세부내용, 산출정보를 함께 확인해야 한다."
            ),
        },
        {
            "file_name": "sample.hwp",
            "content_type": "table",
            "table_id": "table_0001",
            "table_type": "requirement_table",
            "section_path": ["III. 제안요청내용", "2. 요구사항 세부내용"],
            "section_type": "requirements",
            "table": {
                "rows": 7,
                "cols": 3,
                "cell_count": 12,
                "shape": "vertical_key_value",
                "rows_data": [
                    {
                        "row_index": 1,
                        "text": (
                            "요구사항 고유번호: SFR-003 / 요구사항 명칭: 홈페이지 디자인 / "
                            "정의: 홈페이지 디자인 개선 / 세부 내용: UX/UI 디자인 구성"
                        ),
                        "cells": {
                            "요구사항 고유번호": "SFR-003",
                            "요구사항 명칭": "홈페이지 디자인",
                            "정의": "홈페이지 디자인 개선",
                            "세부 내용": "UX/UI 디자인 구성",
                        },
                    }
                ],
                "grid": [
                    ["요구사항 고유번호", "", "SFR-003"],
                    ["요구사항 명칭", "", "홈페이지 디자인"],
                ],
            },
        },
    ]

    chunks = build_rag_chunks(records, _chunk_args())
    compact_output_metadata(chunks)

    chunk_types = [chunk["chunk_type"] for chunk in chunks]
    assert "section_text" in chunk_types
    assert "table_rows/requirement_table" in chunk_types

    text_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "section_text")
    assert "[다음 표: table_0001" in text_chunk["chunk_text"]
    assert "유형=requirement_table" in text_chunk["chunk_text"]
    assert text_chunk["metadata"]["next_table_id"] == "table_0001"
    assert text_chunk["metadata"]["next_table_type"] == "requirement_table"
    assert text_chunk["metadata"]["next_table_shape"] == "vertical_key_value"

    table_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "table_rows/requirement_table")
    assert "자료유형: 표" in table_chunk["chunk_text"]
    assert "요구사항 고유번호: SFR-003" in table_chunk["chunk_text"]
    assert table_chunk["metadata"]["table_id"] == "table_0001"
    assert table_chunk["metadata"]["table_type"] == "requirement_table"
    assert table_chunk["metadata"]["table_shape"] == "vertical_key_value"


def test_table_raw_rows_are_separated_from_rag_chunks() -> None:
    records = [
        {
            "file_name": "sample.hwp",
            "content_type": "table",
            "table_id": "table_0002",
            "table_type": "schedule_table",
            "section_path": ["II. 사업추진 방안", "5. 추진 일정"],
            "section_type": "body",
            "table": {
                "rows": 3,
                "cols": 4,
                "cell_count": 12,
                "shape": "matrix",
                "grid": [
                    ["일정 구분", "M", "M+1", "M+2"],
                    ["시스템 구축", "", "", ""],
                ],
            },
        }
    ]

    table_rows = build_table_raw_rows(records)

    assert len(table_rows) == 1
    row = table_rows[0]
    assert row["table_doc_id"] == "table_doc_00000001"
    assert row["table_id"] == "table_0002"
    assert row["table_type"] == "schedule_table"
    assert row["table_shape"] == "matrix"
    assert row["section_path_text"] == "II. 사업추진 방안 > 5. 추진 일정"
    assert row["table_grid"][0] == ["일정 구분", "M", "M+1", "M+2"]
