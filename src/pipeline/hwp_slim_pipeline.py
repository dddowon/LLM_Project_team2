from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.Parsing.parsing import build_prechunk_records
from src.chunking.chunking import build_rag_chunks, compact_output_metadata


def discover_hwp_files(input_dir: Path, glob_pattern: str = "*.hwp", recursive: bool = False) -> list[Path]:
    pattern = f"**/{glob_pattern}" if recursive else glob_pattern
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def section_path_text(path_items: list[Any]) -> str:
    return " > ".join(str(item).strip() for item in path_items if str(item).strip())


def default_tables_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_tables.jsonl")


def table_markdown_row(record: dict[str, Any], index: int) -> dict[str, Any]:
    table = record.get("table") or {}
    section_path = record.get("section_path") or []
    return {
        "table_doc_id": f"table_doc_{index:08d}",
        "file_name": record.get("file_name", ""),
        "table_id": record.get("table_id", ""),
        "table_type": record.get("table_type", ""),
        "section_path": section_path,
        "section_path_text": section_path_text(section_path),
        "section_type": record.get("section_type", ""),
        "rows": table.get("rows"),
        "cols": table.get("cols"),
        "cell_count": table.get("cell_count"),
        "table_shape": table.get("shape", ""),
        "table_markdown": table.get("markdown", ""),
    }


def build_table_markdown_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("content_type") != "table":
            continue
        row = table_markdown_row(record, len(rows) + 1)
        if row["table_markdown"]:
            rows.append(row)
    return rows


def slim_chunk_row(chunk: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    allowed_metadata = {
        "file_name",
        "section_path_text",
        "section_type",
        "heading",
        "chunk_type",
        "table_type",
        "table_shape",
        "table_id",
        "row_range",
        "next_table_id",
        "next_table_type",
        "next_table_shape",
        "row_indices",
        "summary_group_index",
        "part_index",
        "source_content_type",
    }
    slim_metadata = {
        key: value
        for key, value in metadata.items()
        if key in allowed_metadata and value not in ("", None, [])
    }
    slim_metadata["global_index"] = index

    return {
        "chunk_id": f"slim_chunk_{index:08d}",
        "chunk_type": chunk.get("chunk_type", ""),
        "chunk_text": chunk.get("chunk_text", ""),
        "metadata": slim_metadata,
    }


def build_slim_chunks(
    records: list[dict[str, Any]],
    *,
    text_chunk_size: int,
    text_overlap: int,
    table_chunk_size: int,
    max_table_rows: int,
    min_text_chars: int,
    short_context_chars: int,
    include_cover: bool,
    include_toc: bool,
) -> list[dict[str, Any]]:
    args = argparse.Namespace(
        text_chunk_size=text_chunk_size,
        text_overlap=text_overlap,
        table_chunk_size=table_chunk_size,
        max_table_rows=max_table_rows,
        min_text_chars=min_text_chars,
        short_context_chars=short_context_chars,
        include_cover=include_cover,
        include_toc=include_toc,
    )
    chunks = build_rag_chunks(records, args)
    compact_output_metadata(chunks)
    return [slim_chunk_row(chunk, index) for index, chunk in enumerate(chunks, start=1)]


def chunk_hwp_dir_to_slim_jsonl(
    *,
    input_dir: Path,
    output_path: Path,
    errors_path: Path | None = None,
    tables_output_path: Path | None = None,
    input_files: list[Path] | None = None,
    glob_pattern: str = "*.hwp",
    recursive: bool = False,
    limit_files: int = 0,
    group_size: int = 8,
    text_chunk_size: int = 900,
    text_overlap: int = 150,
    table_chunk_size: int = 1000,
    max_table_rows: int = 6,
    min_text_chars: int = 40,
    short_context_chars: int = 140,
    include_cover: bool = True,
    include_toc: bool = False,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    selected_files = input_files or discover_hwp_files(input_dir, glob_pattern=glob_pattern, recursive=recursive)
    selected_files = [path for path in selected_files if path.is_file()]
    if limit_files:
        selected_files = selected_files[:limit_files]
    if not selected_files:
        raise RuntimeError(f"No HWP files found under {input_dir}")

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, input_file in enumerate(selected_files, start=1):
        print(f"[{index}/{len(selected_files)}] parsing: {input_file.name}")
        try:
            records.extend(build_prechunk_records(input_file, group_size=group_size))
        except Exception as exc:
            error = {
                "file_name": input_file.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)
            print(f"  error: {error['error_type']}: {error['error']}")
            if stop_on_error:
                raise

    chunks = build_slim_chunks(
        records,
        text_chunk_size=text_chunk_size,
        text_overlap=text_overlap,
        table_chunk_size=table_chunk_size,
        max_table_rows=max_table_rows,
        min_text_chars=min_text_chars,
        short_context_chars=short_context_chars,
        include_cover=include_cover,
        include_toc=include_toc,
    )
    write_jsonl(output_path, chunks)

    table_rows = build_table_markdown_rows(records)
    if tables_output_path is None:
        tables_output_path = default_tables_output_path(output_path)
    write_jsonl(tables_output_path, table_rows)

    if errors_path is None:
        errors_path = output_path.with_name(f"{output_path.stem}_errors.json")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    table_chunks = sum(1 for chunk in chunks if str(chunk.get("chunk_type", "")).startswith("table_"))
    return {
        "target_files": len(selected_files),
        "error_files": len(errors),
        "prechunk_records": len(records),
        "chunks": len(chunks),
        "table_chunks": table_chunks,
        "table_markdown_rows": len(table_rows),
        "text_chunks": len(chunks) - table_chunks,
        "output": str(output_path),
        "tables_output": str(tables_output_path),
        "errors": str(errors_path),
    }


def safe_output_stem(value: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|&\s]+", "_", value).strip("._")
    return re.sub(r"_+", "_", stem) or "hwp_chunks"
