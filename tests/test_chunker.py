import argparse

from src.chunking.chunking import build_rag_chunks
from src.preprocessing.chunker import split_text


def test_split_text_respects_overlap_and_min_size() -> None:
    text = "가나다라마바사아자차카타파하. " * 200
    chunks = split_text(text, chunk_size=120, chunk_overlap=20, min_chunk_chars=30)

    assert len(chunks) > 1
    assert all(len(chunk) >= 30 for chunk in chunks)


def test_split_text_rejects_invalid_overlap() -> None:
    try:
        split_text("hello", chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_build_rag_chunks_marks_split_table_continuations() -> None:
    args = argparse.Namespace(
        text_chunk_size=900,
        text_overlap=150,
        table_chunk_size=1000,
        max_table_rows=6,
        min_text_chars=40,
        short_context_chars=140,
        include_cover=True,
        include_toc=False,
    )
    records = [
        {
            "file_name": "sample.hwp",
            "content_type": "table",
            "table_id": "table_0001",
            "table_type": "data_table",
            "section_path": ["section"],
            "section_type": "body",
            "table": {
                "rows": 2,
                "cols": 2,
                "cell_count": 4,
                "shape": "header_rows",
                "rows_data": [
                    {"row_index": 1, "cells": {"Item": "Project", "Value": "A"}},
                    {"row_index": 2, "cells": {"Item": "Period", "Value": "One month"}},
                ],
                "markdown": "| Item | Value |\n| --- | --- |\n| Project | A |",
            },
        },
        {
            "file_name": "sample.hwp",
            "content_type": "section_text",
            "section_path": ["section"],
            "section_type": "body",
            "text": "2",
        },
        {
            "file_name": "sample.hwp",
            "content_type": "table",
            "table_id": "table_0002",
            "table_type": "data_table",
            "section_path": ["section"],
            "section_type": "body",
            "table": {
                "rows": 1,
                "cols": 2,
                "cell_count": 2,
                "shape": "header_rows",
                "rows_data": [
                    {"row_index": 1, "cells": {"Item": "Budget", "Value": "100"}},
                ],
                "markdown": "| Item | Value |\n| --- | --- |\n| Budget | 100 |",
            },
        },
    ]

    chunks = build_rag_chunks(records, args)
    table_chunks = [chunk for chunk in chunks if str(chunk.get("chunk_type", "")).startswith("table_")]

    assert table_chunks[0]["metadata"]["continuation_group_id"] == "table_continuation_0001"
    assert table_chunks[0]["metadata"]["table_continuation_role"] == "start"
    assert table_chunks[0]["metadata"]["continued_to_table_id"] == "table_0002"
    assert table_chunks[1]["metadata"]["table_continuation_role"] == "continued"
    assert table_chunks[1]["metadata"]["continued_from_table_id"] == "table_0001"
    assert "TABLE CONTINUATION table_continuation_0001" in table_chunks[1]["chunk_text"]
