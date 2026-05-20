#!/usr/bin/env python3
"""Parse HWP files into one prechunk JSONL only.

Standalone parsing-only script. It does not import other local project .py files.

Output records are prechunk records:
- cover_text
- section_text
- table
- toc
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import rhwp
    from rhwp.ir.nodes import ListItemBlock, ParagraphBlock, TableBlock
except ImportError:  # pragma: no cover
    rhwp = None

    class ParagraphBlock:  # type: ignore[no-redef]
        pass

    class ListItemBlock:  # type: ignore[no-redef]
        pass

    class TableBlock:  # type: ignore[no-redef]
        pass


DEFAULT_OUTPUT = Path("eda/hwp_prechunk_all.jsonl")

BARE_SECTION_RE = re.compile(
    r"^(사업개요|사업목표|사업유형|사업추진계획|과업의\s*범위|과업의\s*내용|"
    r"과업의\s*일반사항|과업의\s*개요|요구사항\s*상세|공모\s*추진\s*일정|"
    r"입찰참가자격|입찰시\s*고려사항|제안서\s*제출방법(?:\s*및\s*제출서류)?|"
    r"일반사항|제안서의\s*효력|주관사업자\s*선정방식|평가항목\s*및\s*배점\s*한도|"
    r"제안서\s*규격\s*및\s*작성요령|유의사항)$"
)
REQUIREMENT_RE = re.compile(r"요구사항|요구\s*ID|SFR|SER|DAR|DIR|기능요구|보안요구", re.I)
EVALUATION_RE = re.compile(r"평가|배점|점수|평점|기술평가|가격평가|정량|정성|협상대상")
SCHEDULE_RE = re.compile(r"20\d{2}(?:년)?|[1-4]/4|M\+\d+|월별|분기|착수|완료")
SCHEDULE_CONTEXT_RE = re.compile(r"추진\s*일정|공모\s*추진\s*일정|사업\s*기간|수행\s*기간|일정표|월별|연차별|착수|완료")
FORM_RE = re.compile(r"신청서|서약서|각서|기관명|대표자|주소|인감|사업자등록번호")
<<<<<<< HEAD
=======
MENU_CONTEXT_RE = re.compile(r"기능\s*메뉴도|메뉴\s*구성|사이트맵|화면\s*구성|메뉴\s*구조")
ORG_CONTEXT_RE = re.compile(r"추진\s*체계|수행\s*체계|사업수행\s*체계|조직도|역할\s*분담|분담\s*사항")
COMPLIANCE_RE = re.compile(r"기술\s*적용\s*계획|적용계획/결과|부분적용|미적용|준수\s*여부|검토\s*결과")

>>>>>>> 8f9d0e9859aa2ddfb0a0f4c65c4a29dced204ba2
ROMAN_TOKEN_RE = r"(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|IX|IV|V(?:I{0,3})?|X|I{1,3})"
ROMAN_RE = re.compile(rf"^{ROMAN_TOKEN_RE}[.)]?$")
ROMAN_HEADING_RE = re.compile(rf"^({ROMAN_TOKEN_RE})(?:[.)]\s+|\s*/\s*|\s+)(.+)$")
NUMBER_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)([.)])\s*(.+)$")
KOREAN_HEADING_RE = re.compile(r"^([가-하])([.)])\s*(.+)$")
CIRCLED_HEADING_RE = re.compile(r"^([①-⑳])\s*(.+)$")
MAJOR_SECTION_TITLE_RE = re.compile(r"^(사업안내|과업안내|공모사항|제안서\s*작성|붙\s*임)$")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("氠瑢", " ").replace("漠杳", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[\ue000-\uf8ff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_heading_text(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"붙\s*임", "붙임", text)


def block_text(block: Any) -> str:
    if isinstance(block, ParagraphBlock):
        return normalize_text(block.text)
    if isinstance(block, ListItemBlock):
        marker = normalize_text(getattr(block, "marker", ""))
        text = normalize_text(block.text)
        if marker and marker not in {"-", "•", "1."}:
            return normalize_text(f"{marker} {text}")
        return text
    if isinstance(block, TableBlock):
        return normalize_text(block.text)
    return normalize_text(getattr(block, "text", ""))


def cell_text(cell: Any, *, include_nested_table_text: bool = False) -> str:
    parts: list[str] = []
    for inner in getattr(cell, "blocks", []):
        if isinstance(inner, TableBlock) and not include_nested_table_text:
            continue
        text = block_text(inner)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def detect_heading(text: str) -> tuple[int, str] | None:
    short = normalize_text(text)
    if not short or len(short) > 120:
        return None

    match = ROMAN_HEADING_RE.match(short)
    if match:
        roman, title = match.groups()
        title = normalize_heading_text(title)
        if title and len(title) <= 70:
            return 1, normalize_text(f"{roman}. {title}")

    match = NUMBER_HEADING_RE.match(short)
    if match:
        number, marker, title = match.groups()
        title = normalize_text(title)
        if not title:
            return None
        if marker == ")" and len(title) > 55:
            return None
        if marker == "." and len(title) > 90:
            return None
        heading = f"{number}. {title}" if marker == "." else f"{number}) {title}"
        return 2, normalize_text(heading)

    match = KOREAN_HEADING_RE.match(short)
    if match:
        letter, marker, title = match.groups()
        title = normalize_text(title)
        if marker == "." and re.search(r"[:：]", title):
            return None
        if title and len(title) <= 70:
            return 3, normalize_text(f"{letter}{marker} {title}")

    match = CIRCLED_HEADING_RE.match(short)
    if match:
        marker, title = match.groups()
        title = normalize_text(title)
        if title and len(title) <= 70:
            return 4, normalize_text(f"{marker} {title}")

    if BARE_SECTION_RE.match(short):
        return 2, short
    return None


def is_deferred_heading(text: str) -> bool:
    short = normalize_text(text)
    number_match = NUMBER_HEADING_RE.match(short)
    if number_match:
        return number_match.group(2) == ")"
    if KOREAN_HEADING_RE.match(short):
        return True
    if CIRCLED_HEADING_RE.match(short):
        return True
    return False


def is_table_context_heading(text: str) -> bool:
    short = normalize_text(text)
    match = KOREAN_HEADING_RE.match(short)
    if not match:
        return False
    _, marker, title = match.groups()
    return marker == "." and not re.search(r"[:：]", title)


def update_stack(stack: list[dict[str, Any]], level: int, heading: str) -> list[dict[str, Any]]:
    kept = [item for item in stack if int(item["level"]) < level]
    kept.append({"level": level, "heading": heading})
    return kept


def stack_section_path(stack: list[dict[str, Any]]) -> list[str]:
    return [str(item["heading"]) for item in stack]


def section_type(path_items: list[str]) -> str:
    text = " ".join(path_items)
    if re.search(r"사업\s*개요|사업안내|목표|추진\s*배경", text):
        return "overview"
    if re.search(r"과업|요구사항|제안\s*요청|수행\s*범위", text):
        return "requirements"
    if re.search(r"보안|개인정보|접근권한|암호화", text):
        return "security"
    if re.search(r"평가|배점|협상", text):
        return "evaluation"
    if re.search(r"입찰|계약|공모|제출", text):
        return "bid_contract"
    if re.search(r"붙임|서식|양식", text):
        return "appendix_form"
    return "body"


def has_nested_table(table: TableBlock) -> bool:
    return any(isinstance(inner, TableBlock) for cell in table.cells for inner in cell.blocks)


def table_grid(table: TableBlock, *, fill_spans: bool) -> list[list[str]]:
    grid = [["" for _ in range(table.cols)] for _ in range(table.rows)]
    for cell in table.cells:
        text = cell_text(cell)
        row_end = min(table.rows, cell.row + max(1, cell.row_span))
        col_end = min(table.cols, cell.col + max(1, cell.col_span))
        for row in range(cell.row, row_end):
            for col in range(cell.col, col_end):
                if row == cell.row and col == cell.col:
                    grid[row][col] = text
                elif fill_spans:
                    grid[row][col] = text
    return grid


def unique_values(row: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in row:
        clean = normalize_text(value)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        values.append(clean)
    return values


def compact_grid_lines(grid: list[list[str]], *, max_rows: int = 12) -> list[str]:
    lines: list[str] = []
    previous = ""
    for row in grid[:max_rows]:
        values = unique_values(row)
        if not values:
            continue
        line = " / ".join(values)
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return lines


def markdown_cell(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\n", "<br>")


def grid_to_markdown(grid: list[list[str]]) -> str:
    """Render a parsed HWP table grid as GitHub-flavored Markdown.

    Markdown cannot express row/column spans, so the parser uses the span-filled
    grid. That gives retrieval and review a readable approximation while keeping
    the structured payload for chunking.
    """
    non_empty_rows = [row for row in grid if any(normalize_text(value) for value in row)]
    if not non_empty_rows:
        return ""

    col_count = max(len(row) for row in non_empty_rows)
    normalized_rows = [row + [""] * (col_count - len(row)) for row in non_empty_rows]

    header = [markdown_cell(value) for value in normalized_rows[0]]
    separator = ["---"] * col_count
    body_rows = normalized_rows[1:] or [[""] * col_count]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body_rows:
        lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def table_shape(grid: list[list[str]]) -> str:
    rows = [row for row in grid if any(normalize_text(value) for value in row)]
    if not rows:
        return "empty"
    row_count = len(rows)
    col_count = max(len(row) for row in rows)
    first_col_filled = sum(1 for row in rows if row and normalize_text(row[0]))
    first_row_filled = sum(1 for value in rows[0] if normalize_text(value))

    if row_count == 1 or col_count == 1:
        return "freeform"
    if col_count <= 3 and first_col_filled >= max(2, int(row_count * 0.6)):
        return "vertical_key_value"
    if col_count >= 5 and row_count >= 5:
        return "matrix"
    if row_count <= 4 and col_count >= 4:
        return "horizontal_matrix"
    if first_row_filled >= max(2, int(col_count * 0.5)):
        return "header_rows"
    return "mixed"


def layout_heading_from_table(table: TableBlock, grid: list[list[str]]) -> tuple[int, str] | None:
    values: list[str] = []
    for row in grid:
        values.extend(unique_values(row))
    values = [value for value in values if value]
    if not values or len(values) > 4 or table.rows > 8:
        return None

    combined = normalize_text(" ".join(values))
    direct = detect_heading(combined)
    if direct:
        return direct

    if len(values) >= 2 and ROMAN_RE.match(values[0]):
        roman = values[0].rstrip(".)")
        heading = normalize_text(f"{roman}. {' '.join(values[1:])}")
        if len(heading) <= 80:
            return 1, heading

    if len(values) >= 2 and re.fullmatch(r"\d{1,2}[.)]?", values[0]):
        number = values[0].rstrip(")")
        if not number.endswith("."):
            number += "."
        heading = normalize_text(f"{number} {' '.join(values[1:])}")
        if len(heading) <= 90:
            return 2, heading
    return None


def classify_table(table: TableBlock, grid: list[list[str]]) -> str:
    flat = normalize_text(" ".join(value for row in grid for value in row if value))
    filled_slots = sum(1 for row in grid for value in row if value.strip())
    total_slots = max(1, table.rows * table.cols)
    empty_ratio = 1 - filled_slots / total_slots
    span_count = sum(1 for cell in table.cells if cell.row_span > 1 or cell.col_span > 1)
    span_ratio = span_count / max(1, len(table.cells))
    last_values = [row[-1].strip() for row in grid if row and row[-1].strip()]
    page_like = sum(1 for value in last_values if re.fullmatch(r"\d{1,3}", value))
    page_ratio = page_like / max(1, len(last_values))

    if "목차" in flat or (
        page_ratio >= 0.45 and span_ratio >= 0.25 and re.search(r"[ⅠⅡⅢⅣⅤ]|사업안내|과업안내|제안서", flat)
    ):
        return "toc_table"
    if table.rows == 1 and table.cols == 1 and not FORM_RE.search(flat):
        return "note_table"
    if REQUIREMENT_RE.search(flat):
        return "requirement_table"
    if COMPLIANCE_RE.search(flat):
        return "compliance_table"
    if has_nested_table(table):
        return "nested_table"
    if EVALUATION_RE.search(flat):
        return "evaluation_table"
    if SCHEDULE_RE.search(flat) and table.cols >= 5:
        return "schedule_table"
    if FORM_RE.search(flat):
        return "form_table"
    if table.rows <= 2 or (table.cols <= 2 and empty_ratio >= 0.5):
        return "layout_table"
    return "data_table"


def refine_table_type_by_context(table_type: str, path_items: list[str]) -> str:
    """Use the surrounding heading path to fix table type when table text is ambiguous."""
    context = normalize_text(" ".join(path_items))
    if table_type in {"form_table", "note_table", "toc_table"}:
        return table_type
    if MENU_CONTEXT_RE.search(context):
        return "menu_table"
    if ORG_CONTEXT_RE.search(context):
        return "organization_table"
    if SCHEDULE_CONTEXT_RE.search(context):
        return "schedule_table"
    if table_type == "requirement_table":
        return table_type
    if COMPLIANCE_RE.search(context):
        return "compliance_table"
    return table_type


def header_value_rows(grid: list[list[str]]) -> list[dict[str, Any]]:
    if not grid:
        return []
    headers = [normalize_text(value) or f"col_{idx}" for idx, value in enumerate(grid[0])]
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(grid[1:], start=1):
        cells: dict[str, str] = {}
        for col_index, value in enumerate(row):
            clean = normalize_text(value)
            if not clean:
                continue
            header = headers[col_index] if col_index < len(headers) else f"col_{col_index}"
            if clean == header:
                continue
            cells[header] = clean
        if cells:
            rows.append(
                {
                    "row_index": row_index,
                    "text": " / ".join(f"{key}: {value}" for key, value in cells.items()),
                    "cells": cells,
                }
            )
    return rows


def generic_rows(grid: list[list[str]], *, start_row: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(grid[start_row:], start=start_row):
        values = unique_values(row)
        if values:
            rows.append({"row_index": row_index, "text": " / ".join(values), "values": values})
    return rows


def group_rows(rows: list[dict[str, Any]], *, group_size: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for start in range(0, len(rows), group_size):
        chunk = rows[start : start + group_size]
        if chunk:
            groups.append(
                {
                    "row_range": f"{chunk[0]['row_index']}-{chunk[-1]['row_index']}",
                    "text": "\n".join(str(row["text"]) for row in chunk),
                }
            )
    return groups


<<<<<<< HEAD
def table_payload(table: TableBlock, table_type: str, grid: list[list[str]], *, group_size: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"rows": table.rows, "cols": table.cols, "cell_count": len(table.cells)}
    if table_type == "toc_table":
        payload["items"] = generic_rows(grid)
        return payload
    if table_type in {"layout_table", "form_table", "nested_table"}:
=======
def table_payload(
    table: TableBlock,
    table_type: str,
    grid: list[list[str]],
    *,
    group_size: int,
    markdown_grid: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Build compact table payload for pre-chunk JSONL."""
    payload: dict[str, Any] = {
        "rows": table.rows,
        "cols": table.cols,
        "cell_count": len(table.cells),
        "shape": table_shape(markdown_grid or grid),
        "markdown": grid_to_markdown(markdown_grid or grid),
    }

    if table_type == "toc_table":
        payload["items"] = generic_rows(grid)
        return payload

    if table_type in {"layout_table", "form_table", "nested_table", "note_table"}:
>>>>>>> 8f9d0e9859aa2ddfb0a0f4c65c4a29dced204ba2
        payload["summary_lines"] = compact_grid_lines(grid, max_rows=16)
        if table_type == "nested_table":
            payload["nested_table_count"] = sum(
                1 for cell in table.cells for inner in cell.blocks if isinstance(inner, TableBlock)
            )
        return payload
    if table_type in {"requirement_table", "evaluation_table"}:
        payload["rows_data"] = header_value_rows(grid)
        return payload
    if table_type == "schedule_table":
        payload["row_groups"] = group_rows(generic_rows(grid, start_row=2), group_size=group_size)
        return payload

    rows_data = header_value_rows(grid)
    if len(rows_data) <= 20:
        payload["rows_data"] = rows_data
    else:
        payload["row_groups"] = group_rows(generic_rows(grid, start_row=1), group_size=group_size)
    return payload


<<<<<<< HEAD
def table_text_for_record(payload: dict[str, Any]) -> str:
=======
def table_payload_has_content(payload: dict[str, Any]) -> bool:
    if payload.get("markdown"):
        return True
    for key in ("rows_data", "row_groups", "summary_lines", "items"):
        values = payload.get(key)
        if isinstance(values, list) and values:
            return True
    return False


def table_text_for_record(table_type: str, payload: dict[str, Any]) -> str:
>>>>>>> 8f9d0e9859aa2ddfb0a0f4c65c4a29dced204ba2
    if "rows_data" in payload:
        return "\n".join(row["text"] for row in payload["rows_data"][:20])
    if "row_groups" in payload:
        return "\n".join(group["text"] for group in payload["row_groups"][:3])
    if "items" in payload:
        return "\n".join(item["text"] for item in payload["items"][:30])
    if "summary_lines" in payload:
        return "\n".join(payload["summary_lines"])
    return ""


def new_base_record(source_file: str) -> dict[str, Any]:
    return {"file_name": source_file}


def build_prechunk_records(
    input_path: Path,
    *,
    group_size: int = 8,
    debug_headings_path: Path | None = None,
) -> list[dict[str, Any]]:
    _ = debug_headings_path
    if rhwp is None:
        raise SystemExit(
            "rhwp-python is required. In WSL, run this inside the venv where rhwp is installed."
        )

    doc = rhwp.parse(str(input_path))
    ir = doc.to_ir()
    source_file = input_path.name
    records: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    text_buffer: list[str] = []
    pending_table_title: tuple[int, str, bool] | None = None
    pending_table_context: str | None = None
    active_table_context: str | None = None
    pending_major: str | None = None
    section_index = 0
    table_index = 0

    def flush_text() -> None:
        nonlocal section_index, text_buffer
        text = "\n".join(line for line in text_buffer if normalize_text(line)).strip()
        text_buffer = []
        if not normalize_text(text):
            return
        path_items = stack_section_path(stack)
        section_index += 1
        record = new_base_record(source_file)
        record.update(
            {
                "content_type": "section_text" if path_items else "cover_text",
                "section_id": f"section_{section_index:04d}",
                "section_path": path_items,
                "section_type": section_type(path_items),
                "heading": path_items[-1] if path_items else "",
                "text": text,
            }
        )
        records.append(record)

    def flush_pending_table_context_as_text() -> None:
        nonlocal pending_table_context
        if pending_table_context:
            text_buffer.append(pending_table_context)
            pending_table_context = None

    def flush_pending_table_title_as_text() -> None:
        nonlocal pending_table_title, active_table_context
        flush_pending_table_context_as_text()
        if pending_table_title:
            text_buffer.append(pending_table_title[1])
            pending_table_title = None
            active_table_context = None

    def take_pending_table_path_items() -> list[str]:
        nonlocal pending_table_title, pending_table_context, active_table_context
        extras: list[str] = []
        if pending_table_context:
            active_table_context = pending_table_context
            pending_table_context = None
        if active_table_context:
            extras.append(active_table_context)
        if not pending_table_title:
            return extras
        heading_text = pending_table_title[1]
        attach_to_table = pending_table_title[2]
        pending_table_title = None
        if not attach_to_table:
            text_buffer.append(heading_text)
            active_table_context = None
            return []
        extras.append(heading_text)
        return extras

    def flush_pending_major_as_text() -> None:
        nonlocal pending_major
        if pending_major:
            text_buffer.append(pending_major)
            pending_major = None

    def clear_table_context() -> None:
        nonlocal active_table_context
        active_table_context = None

    for block in ir.body:
        if isinstance(block, (ParagraphBlock, ListItemBlock)):
            text = block_text(block)
            if ROMAN_RE.match(text):
                flush_pending_table_title_as_text()
                flush_pending_major_as_text()
                flush_text()
                clear_table_context()
                pending_major = text.rstrip(".)")
                continue
            if pending_major and MAJOR_SECTION_TITLE_RE.match(text):
                flush_pending_table_title_as_text()
                flush_text()
                clear_table_context()
                stack = update_stack(stack, 1, normalize_text(f"{pending_major}. {text}"))
                pending_major = None
                continue
            flush_pending_major_as_text()

            detected = detect_heading(text)
            if detected:
                level, heading = detected
                if is_deferred_heading(text):
                    if is_table_context_heading(text):
                        flush_pending_table_title_as_text()
                        clear_table_context()
                        pending_table_context = heading
                    else:
                        if pending_table_title:
                            flush_pending_table_title_as_text()
                        attach_to_table = bool(pending_table_context or active_table_context) or not text_buffer
                        pending_table_title = (level, heading, attach_to_table)
                else:
                    flush_pending_table_title_as_text()
                    flush_text()
                    clear_table_context()
                    stack = update_stack(stack, level, heading)
            elif text:
                flush_pending_table_title_as_text()
                clear_table_context()
                text_buffer.append(text)
            continue

        if isinstance(block, TableBlock):
            grid_no_fill = table_grid(block, fill_spans=False)
            table_type_initial = classify_table(block, grid_no_fill)
            heading = layout_heading_from_table(block, grid_no_fill)
            if heading:
                flush_pending_table_title_as_text()
                flush_pending_major_as_text()
                flush_text()
                clear_table_context()
                level, heading_text = heading
                stack = update_stack(stack, level, heading_text)
                continue

            grid = table_grid(block, fill_spans=table_type_initial not in {"toc_table", "layout_table"})
            table_type = classify_table(block, grid)
            if table_type == "layout_table":
                lines = compact_grid_lines(grid_no_fill, max_rows=8)
                if lines:
                    if stack:
                        flush_pending_major_as_text()
                        flush_pending_table_title_as_text()
                        clear_table_context()
                        text_buffer.extend(lines)
                    else:
                        record = new_base_record(source_file)
                        record.update(
                            {
                                "content_type": "cover_text",
                                "section_path": [],
                                "section_type": "cover",
                                "heading": "",
                                "text": "\n".join(lines),
                            }
                        )
                        records.append(record)
                continue

            flush_pending_major_as_text()
            table_path_items = take_pending_table_path_items()
            flush_text()
            table_index += 1
            table_id = f"table_{table_index:04d}"
<<<<<<< HEAD
            payload = table_payload(block, table_type, grid, group_size=group_size)
            path_items = stack_section_path(stack)
            if table_path_items:
                path_items = [*path_items, *table_path_items]
=======
            path_items = section_path(stack)
            if table_path_items:
                path_items = [*path_items, *table_path_items]
            table_type = refine_table_type_by_context(table_type, path_items)
            payload = table_payload(block, table_type, grid, group_size=group_size, markdown_grid=grid_no_fill)
            if not table_payload_has_content(payload):
                add_debug("empty_table_skipped", table_type, table_id=table_id, section_path=path_items)
                continue
            add_debug(
                "table",
                table_type,
                table_id=table_id,
                table_title=table_path_items[-1] if table_path_items else "",
                table_context=table_path_items[0] if len(table_path_items) > 1 else "",
                section_path=path_items,
            )
>>>>>>> 8f9d0e9859aa2ddfb0a0f4c65c4a29dced204ba2
            record = new_base_record(source_file)
            content_type = "toc" if table_type == "toc_table" else "table"
            record.update(
                {
                    "content_type": content_type,
                    "table_id": table_id,
                    "table_type": table_type,
                    "section_path": path_items,
                    "section_type": section_type(path_items),
                    "table": payload,
                }
            )
            if content_type == "toc":
                record["text"] = table_text_for_record(payload)
            records.append(record)

    flush_pending_table_title_as_text()
    flush_pending_major_as_text()
    flush_text()
    return records


def discover_hwp_files(input_dir: Path, *, glob_pattern: str, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern)
    return sorted(
        [path for path in iterator if path.is_file() and path.suffix.lower() == ".hwp"],
        key=lambda path: str(path).casefold(),
    )


def write_jsonl(path: Path, records: list[dict[str, Any]], *, limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = records if limit is None else records[:limit]
    with path.open("w", encoding="utf-8") as file:
        for record in selected:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_hwp_files(input_files: list[Path], *, group_size: int, stop_on_error: bool) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    error_count = 0
    total = len(input_files)
    for index, input_path in enumerate(input_files, start=1):
        print(f"[{index}/{total}] parsing: {input_path.name}")
        try:
            records.extend(build_prechunk_records(input_path, group_size=group_size))
        except Exception as exc:
            error_count += 1
            print(f"  error: {type(exc).__name__}: {exc}")
            if stop_on_error:
                raise
    return records, error_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse HWP files into one prechunk JSONL only.")
    parser.add_argument("--input-dir", type=Path, default=Path("files"), help="directory containing HWP files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output prechunk JSONL")
    parser.add_argument("--glob", default="*.hwp", help="HWP filename glob")
    parser.add_argument("--recursive", action="store_true", help="search input directory recursively")
    parser.add_argument("--limit-files", type=int, default=0, help="parse only first N files; 0 means all")
    parser.add_argument("--group-size", type=int, default=8, help="row count per group for large/schedule tables")
    parser.add_argument("--stop-on-error", action="store_true", help="stop when one file fails")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    input_files = discover_hwp_files(args.input_dir, glob_pattern=args.glob, recursive=args.recursive)
    if args.limit_files:
        input_files = input_files[: args.limit_files]
    if not input_files:
        raise SystemExit(f"No HWP files found: {args.input_dir} ({args.glob})")

    records, error_count = parse_hwp_files(input_files, group_size=args.group_size, stop_on_error=args.stop_on_error)
    write_jsonl(args.output, records)
    print(f"target_files: {len(input_files)}")
    print(f"error_files: {error_count}")
    print(f"written_records: {len(records)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
