from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.jsonl import read_jsonl, write_jsonl


def convert(input_path: Path, output_path: Path, doc_id: str | None = None) -> int:
    rows = read_jsonl(input_path)
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        text = _extract_text(row)
        if not text:
            continue

        rid = row.get("chunk_id") or row.get("section_id") or row.get("table_id") or row.get("id") or f"row_{idx:05d}"
        resolved_doc_id = doc_id or row.get("doc_id") or row.get("file_name") or "unknown_doc"

        metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {}
        metadata.setdefault("content_type", row.get("content_type", ""))
        metadata.setdefault("heading_path", row.get("heading_path") or row.get("section_path") or [])
        metadata.setdefault("section_type", row.get("section_type", ""))
        metadata.setdefault("table_type", row.get("table_type", ""))

        out.append({"id": rid, "doc_id": str(resolved_doc_id), "text": text, "metadata": metadata})

    write_jsonl(output_path, out)
    return len(out)


def _extract_text(row: dict[str, Any]) -> str:
    # New chunking output
    direct = str(row.get("chunk_text", "")).strip()
    if direct:
        return direct

    # Old prechunk output
    direct = str(row.get("text", "")).strip()
    if direct:
        return direct

    table = row.get("table")
    if not isinstance(table, dict):
        return ""

    lines: list[str] = []
    for key in ("rows_data", "row_groups", "items"):
        for item in table.get(key, []):
            if isinstance(item, dict) and item.get("text"):
                lines.append(str(item["text"]).strip())
    for item in table.get("summary_lines", []):
        if item:
            lines.append(str(item).strip())
    return "\n".join(line for line in lines if line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert prechunk/chunk JSONL to embedding input JSONL")
    parser.add_argument("--input", required=True, help="Path to prechunk/chunk JSONL")
    parser.add_argument("--output", required=True, help="Path to embedding input JSONL")
    parser.add_argument("--doc-id", required=False, default=None, help="Optional override document id")
    args = parser.parse_args()

    count = convert(Path(args.input), Path(args.output), args.doc_id)
    print(f"Converted {count} rows -> {args.output}")


if __name__ == "__main__":
    main()
