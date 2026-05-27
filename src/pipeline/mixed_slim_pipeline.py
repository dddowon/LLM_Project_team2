from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.Parsing.parsing import build_prechunk_records as build_hwp_prechunk_records
from src.Parsing.pdf_parsing import build_prechunk_records as build_pdf_prechunk_records
from src.pipeline.hwp_slim_pipeline import (
    build_slim_chunks,
    build_table_raw_rows,
    default_tables_output_path,
    write_jsonl,
)


SUPPORTED_SUFFIXES = {".hwp", ".pdf"}


def safe_output_stem(value: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|&\s]+", "_", value).strip("._")
    return re.sub(r"_+", "_", stem) or "mixed_chunks"


def discover_source_files(
    input_dir: Path,
    *,
    glob_pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    iterator = input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern)
    return sorted(
        [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES],
        key=lambda path: str(path).casefold(),
    )


def parse_source_file(
    input_file: Path,
    *,
    group_size: int,
    pdf_backend: str,
    pdf_extract_tables: bool,
) -> list[dict[str, Any]]:
    suffix = input_file.suffix.lower()
    if suffix == ".hwp":
        return build_hwp_prechunk_records(input_file, group_size=group_size)
    if suffix == ".pdf":
        return build_pdf_prechunk_records(
            input_file,
            group_size=group_size,
            backend=pdf_backend,
            extract_tables=pdf_extract_tables,
        )
    raise ValueError(f"Unsupported file extension: {input_file}")


def chunk_mixed_dir_to_slim_jsonl(
    *,
    input_dir: Path,
    output_path: Path,
    errors_path: Path | None = None,
    tables_output_path: Path | None = None,
    input_files: list[Path] | None = None,
    glob_pattern: str = "*",
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
    pdf_backend: str = "auto",
    pdf_extract_tables: bool = True,
) -> dict[str, Any]:
    selected_files = input_files or discover_source_files(
        input_dir,
        glob_pattern=glob_pattern,
        recursive=recursive,
    )
    selected_files = [path for path in selected_files if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    if limit_files:
        selected_files = selected_files[:limit_files]
    if not selected_files:
        raise RuntimeError(f"No HWP/PDF files found under {input_dir}")

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    parsed_by_suffix = {".hwp": 0, ".pdf": 0}
    for index, input_file in enumerate(selected_files, start=1):
        print(f"[{index}/{len(selected_files)}] parsing: {input_file.name}")
        try:
            file_records = parse_source_file(
                input_file,
                group_size=group_size,
                pdf_backend=pdf_backend,
                pdf_extract_tables=pdf_extract_tables,
            )
            parsed_by_suffix[input_file.suffix.lower()] = parsed_by_suffix.get(input_file.suffix.lower(), 0) + 1
            records.extend(file_records)
            table_records = sum(1 for record in file_records if record.get("content_type") == "table")
            print(f"  records={len(file_records)} tables={table_records}")
        except Exception as exc:
            error = {
                "file_name": input_file.name,
                "suffix": input_file.suffix.lower(),
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

    table_rows = build_table_raw_rows(records)
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
        "parsed_hwp_files": parsed_by_suffix.get(".hwp", 0),
        "parsed_pdf_files": parsed_by_suffix.get(".pdf", 0),
        "error_files": len(errors),
        "prechunk_records": len(records),
        "chunks": len(chunks),
        "text_chunks": len(chunks) - table_chunks,
        "table_chunks": table_chunks,
        "table_raw_rows": len(table_rows),
        "pdf_backend": pdf_backend,
        "pdf_extract_tables": pdf_extract_tables,
        "output": str(output_path),
        "tables_output": str(tables_output_path),
        "errors": str(errors_path),
    }
