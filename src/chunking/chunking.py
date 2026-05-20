#!/usr/bin/env python3
"""Convert pre-chunk HWP JSONL into retrieval-ready RAG chunks.

Input is produced by hwp_parse_prechunk_jsonl.py or hwp_parse_prechunk_jsonl_all.py.

Chunking policy:
- section_text: merge adjacent text within the same section path, then split by
  paragraph/list boundaries with overlap.
- table rows_data: rebuild row text from cells as key-value lines, deduplicate
  repeated merged-cell rows, then group rows by character budget.
- table row_groups / summary_lines: preserve their compact table structure and
  chunk by group/line budget.
- toc: skipped by default because it is usually navigation noise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


TEXT_BOUNDARY_RE = re.compile(
    r"\n\s*(?=(?:\d{1,2}(?:\.\d{1,2})*[.)]\s|[가-힣][.)]\s|[①-⑳]|[○●□■※ㆍ\-]))"
)
SPACE_RE = re.compile(r"[ \t\u00a0]+")


def normalize_inline(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_text_block(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]

    cleaned: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank and cleaned:
                cleaned.append("")
            blank = True
            continue
        cleaned.append(line)
        blank = False
    return "\n".join(cleaned).strip()


def compact_key(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def stable_hash(value: str, *, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def path_text(path_items: Iterable[Any]) -> str:
    items = [normalize_inline(item) for item in path_items if normalize_inline(item)]
    return " > ".join(items)


def is_low_value_text(body: str, *, content_type: str) -> bool:
    compact = compact_key(body)
    if not compact:
        return True
    if compact in {"목차", "차례", "tableofcontents"}:
        return True
    if content_type == "cover_text" and len(compact) <= 30 and ("목차" in compact or "차례" in compact):
        return True
    return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_source_record_index"] = line_number
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_summary_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.csv")


def default_sample_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_sample.jsonl")


def text_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", clean_text_block(text)):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = [part.strip() for part in TEXT_BOUNDARY_RE.split(paragraph) if part.strip()]
        units.extend(parts or [paragraph])
    return units


def split_oversized_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    text = clean_text_block(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind("다. ", start, end))
            if boundary > start + int(chunk_size * 0.55):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_section_text(text: str, *, chunk_size: int, overlap: int, min_chars: int) -> list[str]:
    units = text_units(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        joined = clean_text_block("\n\n".join(current))
        current = []
        current_len = 0
        if not joined:
            return
        if len(joined) > chunk_size:
            chunks.extend(split_oversized_text(joined, chunk_size=chunk_size, overlap=overlap))
        else:
            chunks.append(joined)

    for unit in units:
        unit_len = len(unit)
        separator_len = 2 if current else 0
        if current and current_len + separator_len + unit_len > chunk_size:
            flush()
        if unit_len > chunk_size:
            flush()
            chunks.extend(split_oversized_text(unit, chunk_size=chunk_size, overlap=overlap))
        else:
            current.append(unit)
            current_len += separator_len + unit_len
    flush()

    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = clean_text_block(chunks[-2] + "\n\n" + chunks[-1])
        chunks.pop()

    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0 or overlap <= 0:
            overlapped.append(chunk)
            continue
        tail = chunks[index - 1][-overlap:].strip()
        if tail:
            overlapped.append(clean_text_block(tail + "\n\n" + chunk))
        else:
            overlapped.append(chunk)
    return [chunk for chunk in overlapped if chunk]


def base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    section_path = record.get("section_path") or []
    metadata = {
        "file_name": record.get("file_name", ""),
        "source_record_index": record.get("_source_record_index"),
        "source_content_type": record.get("content_type", ""),
        "section_path": section_path,
        "section_path_text": path_text(section_path),
        "section_type": record.get("section_type", ""),
        "heading": record.get("heading", ""),
    }
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}


def chunk_type_label(chunk_type: str) -> str:
    if chunk_type == "section_text":
        return "본문"
    if chunk_type == "cover_text":
        return "표지"
    if chunk_type.startswith("table_"):
        return "표"
    return chunk_type


def context_prefix(metadata: dict[str, Any], *, chunk_type: str, extra_lines: list[str] | None = None) -> str:
    lines = [f"문서명: {metadata.get('file_name', '')}"]
    section = metadata.get("section_path_text")
    if section:
        lines.append(f"섹션경로: {section}")
    heading = metadata.get("heading")
    if heading and heading != section:
        lines.append(f"제목: {heading}")
    lines.append(f"자료유형: {chunk_type_label(chunk_type)}")
    if extra_lines:
        lines.extend(line for line in extra_lines if normalize_inline(line))
    return "\n".join(lines)


def make_chunk(
    chunk_no: int,
    *,
    chunk_type: str,
    body: str,
    metadata: dict[str, Any],
    extra_prefix_lines: list[str] | None = None,
) -> dict[str, Any]:
    clean_body = clean_text_block(body)
    prefix = context_prefix(metadata, chunk_type=chunk_type, extra_lines=extra_prefix_lines)
    chunk_text = clean_text_block(prefix + "\n\n" + clean_body)
    content_hash = stable_hash(chunk_text)
    metadata = dict(metadata)
    metadata.update(
        {
            "chunk_type": chunk_type,
            "body_chars": len(clean_body),
            "chunk_chars": len(chunk_text),
            "content_hash": content_hash,
        }
    )
    return {
        "chunk_id": f"chunk_{chunk_no:08d}_{content_hash}",
        "chunk_type": chunk_type,
        "chunk_text": chunk_text,
        "metadata": metadata,
    }


def cells_to_lines(cells: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    seen_values: set[str] = set()
    for key, value in cells.items():
        clean_key = normalize_inline(key)
        clean_value = normalize_inline(value)
        if not clean_value or clean_value == clean_key:
            continue

        value_key = compact_key(clean_value)
        if clean_key.startswith("col_") and value_key in seen_values:
            continue
        seen_values.add(value_key)

        if clean_key.startswith("col_"):
            lines.append(clean_value)
        else:
            lines.append(f"{clean_key}: {clean_value}")
    return lines


def table_row_text(row: dict[str, Any]) -> str:
    cells = row.get("cells")
    if isinstance(cells, dict) and cells:
        lines = cells_to_lines(cells)
        if lines:
            return "\n".join(lines)
    return clean_text_block(row.get("text", ""))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        body = table_row_text(row)
        key = compact_key(body)
        if not key or key in seen:
            continue
        seen.add(key)
        copied = dict(row)
        copied["_rag_row_text"] = body
        deduped.append(copied)
    return deduped


def row_group_label(rows: list[dict[str, Any]]) -> str:
    indices = [str(row.get("row_index")) for row in rows if row.get("row_index") is not None]
    if not indices:
        return ""
    if len(indices) == 1:
        return indices[0]
    return f"{indices[0]}-{indices[-1]}"


def group_table_rows(
    rows: list[dict[str, Any]], *, table_chunk_size: int, max_rows_per_chunk: int
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    current_rows: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_len = 0

    def row_part(row: dict[str, Any]) -> str:
        label = row.get("row_index")
        prefix = f"행 {label}\n" if label is not None else ""
        return clean_text_block(prefix + str(row["_rag_row_text"]))

    def flush() -> None:
        nonlocal current_rows, current_parts, current_len
        if current_parts:
            grouped.append(("\n\n".join(current_parts), current_rows))
        current_rows = []
        current_parts = []
        current_len = 0

    for row in rows:
        part = row_part(row)
        if not part:
            continue
        part_len = len(part)
        if current_parts and (
            current_len + 2 + part_len > table_chunk_size or len(current_rows) >= max_rows_per_chunk
        ):
            flush()
        if part_len > table_chunk_size:
            flush()
            for piece in split_oversized_text(part, chunk_size=table_chunk_size, overlap=80):
                single = dict(row)
                single["_rag_row_text"] = piece
                grouped.append((piece, [single]))
            continue
        current_rows.append(row)
        current_parts.append(part)
        current_len += (2 if current_parts else 0) + part_len
    flush()
    return grouped


def table_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = base_metadata(record)
    table = record.get("table") or {}
    metadata.update(
        {
            "table_id": record.get("table_id", ""),
            "table_type": record.get("table_type", ""),
            "table_shape": table.get("shape"),
            "table_rows": table.get("rows"),
            "table_cols": table.get("cols"),
            "table_cell_count": table.get("cell_count"),
        }
    )
    if table.get("nested_table_count") is not None:
        metadata["nested_table_count"] = table.get("nested_table_count")
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}


def chunks_from_table(
    record: dict[str, Any],
    *,
    next_chunk_no: int,
    table_chunk_size: int,
    max_rows_per_chunk: int,
    pending_context_note: str = "",
) -> tuple[list[dict[str, Any]], int]:
    table = record.get("table") or {}
    metadata = table_metadata(record)
    table_type = str(metadata.get("table_type", "table"))
    prefix_lines: list[str] = []
    if pending_context_note:
        prefix_lines.append(f"직전본문: {pending_context_note}")
        prefix_lines.append("표위치: 위 직전본문 다음에 이 표가 이어짐")

    chunks: list[dict[str, Any]] = []

    rows_data = table.get("rows_data")
    if isinstance(rows_data, list) and rows_data:
        rows = dedupe_rows(rows_data)
        for body, grouped_rows in group_table_rows(
            rows, table_chunk_size=table_chunk_size, max_rows_per_chunk=max_rows_per_chunk
        ):
            row_indices = [row.get("row_index") for row in grouped_rows if row.get("row_index") is not None]
            row_metadata = dict(metadata)
            row_metadata.update(
                {
                    "row_indices": row_indices,
                    "row_range": row_group_label(grouped_rows),
                    "deduped_table_rows": len(rows),
                }
            )
            chunks.append(
                make_chunk(
                    next_chunk_no,
                    chunk_type=f"table_rows/{table_type}",
                    body=body,
                    metadata=row_metadata,
                    extra_prefix_lines=prefix_lines,
                )
            )
            next_chunk_no += 1
        return chunks, next_chunk_no

    row_groups = table.get("row_groups")
    if isinstance(row_groups, list) and row_groups:
        for group in row_groups:
            body = clean_text_block(group.get("text", ""))
            if not body:
                continue
            group_metadata = dict(metadata)
            group_metadata["row_range"] = group.get("row_range", "")
            for piece_index, piece in enumerate(
                split_oversized_text(body, chunk_size=table_chunk_size, overlap=80), start=1
            ):
                piece_metadata = dict(group_metadata)
                if piece_index > 1:
                    piece_metadata["split_part"] = piece_index
                chunks.append(
                    make_chunk(
                        next_chunk_no,
                        chunk_type=f"table_group/{table_type}",
                        body=piece,
                        metadata=piece_metadata,
                        extra_prefix_lines=prefix_lines,
                    )
                )
                next_chunk_no += 1
        return chunks, next_chunk_no

    summary_lines = table.get("summary_lines")
    if isinstance(summary_lines, list) and summary_lines:
        current: list[str] = []
        current_len = 0
        group_index = 1

        def flush_summary() -> None:
            nonlocal current, current_len, group_index, next_chunk_no
            if not current:
                return
            body = "\n".join(current)
            summary_metadata = dict(metadata)
            summary_metadata["summary_group_index"] = group_index
            chunks.append(
                make_chunk(
                    next_chunk_no,
                    chunk_type=f"table_summary/{table_type}",
                    body=body,
                    metadata=summary_metadata,
                    extra_prefix_lines=prefix_lines,
                )
            )
            next_chunk_no += 1
            group_index += 1
            current = []
            current_len = 0

        for line in summary_lines:
            clean_line = clean_text_block(line)
            if not clean_line:
                continue
            if current and current_len + 1 + len(clean_line) > table_chunk_size:
                flush_summary()
            if len(clean_line) > table_chunk_size:
                flush_summary()
                for piece_index, piece in enumerate(
                    split_oversized_text(clean_line, chunk_size=table_chunk_size, overlap=80), start=1
                ):
                    summary_metadata = dict(metadata)
                    summary_metadata["summary_group_index"] = group_index
                    summary_metadata["split_part"] = piece_index
                    chunks.append(
                        make_chunk(
                            next_chunk_no,
                            chunk_type=f"table_summary/{table_type}",
                            body=piece,
                            metadata=summary_metadata,
                            extra_prefix_lines=prefix_lines,
                        )
                    )
                    next_chunk_no += 1
                group_index += 1
                continue
            current.append(clean_line)
            current_len += len(clean_line) + 1
        flush_summary()
        return chunks, next_chunk_no

    return chunks, next_chunk_no


def flush_text_buffer(
    text_records: list[dict[str, Any]],
    *,
    next_chunk_no: int,
    text_chunk_size: int,
    text_overlap: int,
    min_text_chars: int,
    short_context_chars: int,
    as_table_context: bool,
    following_table: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    if not text_records:
        return [], next_chunk_no, ""

    body = clean_text_block("\n\n".join(str(record.get("text", "")) for record in text_records))
    if not body:
        return [], next_chunk_no, ""

    first = text_records[0]
    if is_low_value_text(body, content_type=str(first.get("content_type", ""))):
        return [], next_chunk_no, ""

    if as_table_context and len(body) <= short_context_chars:
        return [], next_chunk_no, body

    metadata = base_metadata(first)
    metadata["source_record_indices"] = [record.get("_source_record_index") for record in text_records]
    metadata["merged_record_count"] = len(text_records)
    chunk_type = "cover_text" if first.get("content_type") == "cover_text" else "section_text"

    chunks: list[dict[str, Any]] = []
    for part_index, part in enumerate(
        split_section_text(body, chunk_size=text_chunk_size, overlap=text_overlap, min_chars=min_text_chars), start=1
    ):
        if len(part) < min_text_chars and chunks:
            previous = chunks[-1]
            previous["chunk_text"] = clean_text_block(previous["chunk_text"] + "\n\n" + part)
            previous["metadata"]["body_chars"] += len(part)
            previous["metadata"]["chunk_chars"] = len(previous["chunk_text"])
            continue
        part_metadata = dict(metadata)
        part_metadata["part_index"] = part_index
        chunks.append(make_chunk(next_chunk_no, chunk_type=chunk_type, body=part, metadata=part_metadata))
        next_chunk_no += 1
    if chunks and following_table:
        marker = table_following_marker(following_table)
        if marker:
            chunks[-1]["chunk_text"] = clean_text_block(chunks[-1]["chunk_text"] + "\n\n" + marker)
            chunks[-1]["metadata"]["next_table_id"] = following_table.get("table_id", "")
            chunks[-1]["metadata"]["next_table_type"] = following_table.get("table_type", "")
            chunks[-1]["metadata"]["next_table_shape"] = (following_table.get("table") or {}).get("shape", "")
    return chunks, next_chunk_no, ""


def table_following_marker(record: dict[str, Any]) -> str:
    table = record.get("table") or {}
    table_id = record.get("table_id", "")
    table_type = record.get("table_type", "")
    table_shape_value = table.get("shape", "")
    size = ""
    if table.get("rows") and table.get("cols"):
        size = f", {table.get('rows')}행x{table.get('cols')}열"
    if not table_id:
        return ""
    parts = [f"[다음 표: {table_id}"]
    if table_type:
        parts.append(f", 유형={table_type}")
    if table_shape_value:
        parts.append(f", 형태={table_shape_value}")
    if size:
        parts.append(size)
    parts.append("]")
    return "".join(parts)


def same_text_bucket(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("file_name") == right.get("file_name")
        and left.get("content_type") == right.get("content_type")
        and (left.get("section_path") or []) == (right.get("section_path") or [])
    )


def build_rag_chunks(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    text_buffer: list[dict[str, Any]] = []
    next_chunk_no = 1

    def extend_text_buffer(record: dict[str, Any]) -> None:
        nonlocal text_buffer
        if text_buffer and not same_text_bucket(text_buffer[-1], record):
            flush_to_chunks(as_table_context=False)
        text_buffer.append(record)

    def flush_to_chunks(*, as_table_context: bool, following_table: dict[str, Any] | None = None) -> str:
        nonlocal text_buffer, next_chunk_no, chunks
        emitted, next_chunk_no, context_note = flush_text_buffer(
            text_buffer,
            next_chunk_no=next_chunk_no,
            text_chunk_size=args.text_chunk_size,
            text_overlap=args.text_overlap,
            min_text_chars=args.min_text_chars,
            short_context_chars=args.short_context_chars,
            as_table_context=as_table_context,
            following_table=following_table,
        )
        chunks.extend(emitted)
        text_buffer = []
        return context_note

    for record in records:
        content_type = record.get("content_type")
        if content_type == "section_text" or (content_type == "cover_text" and args.include_cover):
            extend_text_buffer(record)
            continue

        if content_type == "table":
            context_note = flush_to_chunks(as_table_context=True, following_table=record)
            table_chunks, next_chunk_no = chunks_from_table(
                record,
                next_chunk_no=next_chunk_no,
                table_chunk_size=args.table_chunk_size,
                max_rows_per_chunk=args.max_table_rows,
                pending_context_note=context_note,
            )
            chunks.extend(table_chunks)
            continue

        if content_type == "toc" and args.include_toc:
            context_note = flush_to_chunks(as_table_context=False)
            toc_record = dict(record)
            toc_record["content_type"] = "section_text"
            toc_record["text"] = clean_text_block(record.get("text", ""))
            toc_chunks, next_chunk_no, _ = flush_text_buffer(
                [toc_record],
                next_chunk_no=next_chunk_no,
                text_chunk_size=args.text_chunk_size,
                text_overlap=0,
                min_text_chars=args.min_text_chars,
                short_context_chars=args.short_context_chars,
                as_table_context=False,
            )
            if context_note:
                for chunk in toc_chunks:
                    chunk["chunk_text"] = clean_text_block(f"직전본문: {context_note}\n\n{chunk['chunk_text']}")
            chunks.extend(toc_chunks)
            continue

        flush_to_chunks(as_table_context=False)

    flush_to_chunks(as_table_context=False)
    return chunks


def write_summary_csv(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_type = Counter(chunk["chunk_type"] for chunk in chunks)
    by_file = Counter(chunk["metadata"].get("file_name", "") for chunk in chunks)
    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    rows = [
        {"metric": "total_chunks", "value": len(chunks)},
        {"metric": "files", "value": len(by_file)},
        {"metric": "short_chunks_lt_200", "value": sum(length < 200 for length in lengths)},
        {"metric": "long_chunks_gt_1500", "value": sum(length > 1500 for length in lengths)},
        {"metric": "max_chunk_chars", "value": max(lengths) if lengths else 0},
        {"metric": "avg_chunk_chars", "value": round(sum(lengths) / len(lengths), 1) if lengths else 0},
    ]
    for chunk_type, count in by_type.most_common():
        rows.append({"metric": f"chunk_type:{chunk_type}", "value": count})

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_sample_jsonl(path: Path, chunks: list[dict[str, Any]], *, sample_size: int) -> None:
    if sample_size <= 0:
        return
    selected: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for chunk in chunks:
        chunk_type = chunk["chunk_type"]
        if chunk_type in seen_types:
            continue
        selected.append(chunk)
        seen_types.add(chunk_type)
        if len(selected) >= sample_size:
            break
    if len(selected) < sample_size:
        selected.extend(chunks[: max(0, sample_size - len(selected))])
    write_jsonl(path, selected[:sample_size])


def compact_output_metadata(chunks: list[dict[str, Any]]) -> None:
    """Keep only metadata that is useful for retrieval filters and citations."""
    common_keys = ["file_name", "section_path", "section_type", "heading"]
    table_keys = ["table_type", "table_shape", "table_id", "row_range"]
    link_keys = ["next_table_id", "next_table_type", "next_table_shape"]
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        compact: dict[str, Any] = {}
        for key in common_keys:
            value = metadata.get(key)
            if value not in ("", None, []):
                compact[key] = value
        if str(chunk.get("chunk_type", "")).startswith("table_"):
            for key in table_keys:
                value = metadata.get(key)
                if value not in ("", None, []):
                    compact[key] = value
        else:
            for key in link_keys:
                value = metadata.get(key)
                if value not in ("", None, []):
                    compact[key] = value
        chunk["metadata"] = compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pre-chunk JSONL into RAG-ready chunks.")
    parser.add_argument("--input", type=Path, default=Path("eda/hwp_prechunk_all.jsonl"), help="pre-chunk JSONL path")
    parser.add_argument("--output", type=Path, default=Path("eda/hwp_rag_chunks_all.jsonl"), help="RAG chunk JSONL path")
    parser.add_argument("--summary-output", type=Path, default=None, help="summary CSV path")
    parser.add_argument("--sample-output", type=Path, default=None, help="sample JSONL path")
    parser.add_argument("--sample-size", type=int, default=20, help="number of sample chunks to write")
    parser.add_argument("--text-chunk-size", type=int, default=900, help="target body chars for section text chunks")
    parser.add_argument("--text-overlap", type=int, default=180, help="section text overlap chars")
    parser.add_argument("--table-chunk-size", type=int, default=1000, help="target body chars for table chunks")
    parser.add_argument("--max-table-rows", type=int, default=6, help="max table rows per chunk")
    parser.add_argument("--min-text-chars", type=int, default=40, help="minimum body chars for standalone text chunks")
    parser.add_argument(
        "--short-context-chars",
        type=int,
        default=140,
        help="short text before a table is attached to the table instead of emitted alone",
    )
    parser.add_argument("--include-cover", action="store_true", default=True, help="include cover_text chunks")
    parser.add_argument("--exclude-cover", action="store_false", dest="include_cover", help="skip cover_text chunks")
    parser.add_argument("--include-toc", action="store_true", help="include toc chunks; default skips them")
    parser.add_argument(
        "--include-debug-metadata",
        action="store_true",
        help="keep QA/debug fields such as source indices, lengths, hashes, and split parts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    chunks = build_rag_chunks(records, args)
    if not args.include_debug_metadata:
        compact_output_metadata(chunks)

    write_jsonl(args.output, chunks)
    summary_output = args.summary_output or default_summary_output(args.output)
    sample_output = args.sample_output or default_sample_output(args.output)
    write_summary_csv(summary_output, chunks)
    write_sample_jsonl(sample_output, chunks, sample_size=args.sample_size)

    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    by_type = Counter(chunk["chunk_type"] for chunk in chunks)
    print(f"input_records: {len(records)}")
    print(f"output_chunks: {len(chunks)}")
    print(f"chunk_types: {dict(by_type.most_common())}")
    if lengths:
        print(f"chunk_chars_avg: {round(sum(lengths) / len(lengths), 1)}")
        print(f"chunk_chars_max: {max(lengths)}")
        print(f"short_chunks_lt_200: {sum(length < 200 for length in lengths)}")
        print(f"long_chunks_gt_1500: {sum(length > 1500 for length in lengths)}")
    print(f"output: {args.output}")
    print(f"summary_output: {summary_output}")
    print(f"sample_output: {sample_output}")


if __name__ == "__main__":
    main()
