from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.merge_embedded import discover_embedded_jsonl, merge_embedded_jsonl


def test_discover_and_merge_embedded(tmp_path: Path) -> None:
    doc_a = tmp_path / "doc_a"
    doc_b = tmp_path / "doc_b"
    doc_a.mkdir()
    doc_b.mkdir()

    row_a = {
        "chunk_id": "chunk_00000001_abc",
        "chunk_type": "text",
        "chunk_text": "alpha",
        "metadata": {"file_name": "a.hwp"},
        "embedding": [0.1, 0.2],
    }
    row_b = {
        "chunk_id": "chunk_00000001_abc",
        "chunk_type": "text",
        "chunk_text": "beta",
        "metadata": {"file_name": "b.hwp"},
        "embedding": [0.3, 0.4],
    }

    (doc_a / "a_embedded.jsonl").write_text(json.dumps(row_a, ensure_ascii=False) + "\n", encoding="utf-8")
    (doc_b / "b_embedded.jsonl").write_text(json.dumps(row_b, ensure_ascii=False) + "\n", encoding="utf-8")

    sources = discover_embedded_jsonl(tmp_path, pattern="*_embedded.jsonl", recursive=True)
    assert len(sources) == 2

    merged_path = tmp_path / "all_embedded.jsonl"
    merged, collisions = merge_embedded_jsonl(sources, merged_path)
    assert len(merged) == 2
    assert collisions == 1
    assert merged[0]["chunk_id"] == "chunk_00000001_abc"
    assert merged[1]["chunk_id"] != merged[0]["chunk_id"]
