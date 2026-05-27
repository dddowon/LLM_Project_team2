#!/usr/bin/env python3
"""Parse PDF files into the same prechunk JSONL shape used by the HWP parser.

The parser prefers PyMuPDF when available because it can preserve layout order
and detect tables. It falls back to pypdf for text-only extraction.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


ROMAN_TOKEN_RE = r"(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|IX|IV|V(?:I{0,3})?|X|I{1,3})"
ROMAN_ONLY_RE = re.compile(rf"^{ROMAN_TOKEN_RE}[.)]?$")
ROMAN_HEADING_RE = re.compile(rf"^({ROMAN_TOKEN_RE})(?:[.)]\s+|\s+)(.+)$")
NUMBER_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)([.)])\s*(.+)$")
KOREAN_HEADING_RE = re.compile(r"^([가-힣])([.)])\s*(.+)$")
CIRCLED_HEADING_RE = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩])\s*(.+)$")
BARE_SECTION_RE = re.compile(
    r"^(사업\s*개요|사업안내|사업목표|사업유형|추진\s*배경(?:\s*및\s*방향)?|"
    r"과업\s*개요|과업\s*범위|과업의\s*범위|과업\s*내용|과업의\s*내용|"
    r"요구사항\s*상세|제안\s*요청\s*내용|평가\s*항목(?:\s*및\s*배점)?|"
    r"입찰\s*참가\s*자격|제안서\s*작성|제출\s*서류|붙임|별지\s*서식|"
    r"보안\s*요구사항|개인정보\s*보호)$"
)

TOC_LINE_RE = re.compile(r".{2,}\s(?:\.{2,}|·{2,}|-{1,}|\s{3,})\s*\d{1,4}$")
PAGE_MARKER_RE = re.compile(r"^\s*(?:-?\s*)?\d{1,4}\s*(?:-)?\s*$")
SPACE_RE = re.compile(r"[ \t\u00a0]+")
FIELD_VALUE_RE = re.compile(
    r"(사업\s*명|사업\s*기간|사업\s*예산|입찰\s*방식|계약\s*방법|"
    r"발주\s*기관|수요\s*기관|담당자|주소|전화|이메일|대표자|금액|기간)\s*[:：]"
)
SENTENCE_END_RE = re.compile(r"(한다|있다|없다|한다\.|있음|없음|하여야|되어야|제공)$")


@dataclass
class PdfLine:
    text: str
    page_no: int
    y: float
    x: float = 0.0
    font_size: float = 0.0
    bold: bool = False


@dataclass
class PdfTable:
    rows: list[list[str]]
    page_no: int
    y: float
    bbox: tuple[float, float, float, float] | None = None


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[\ue000-\uf8ff]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_line(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"\s+", " ", text).strip()


def path_text(path_items: list[str]) -> str:
    return " > ".join(item for item in path_items if item)


def section_type(path_items: list[str]) -> str:
    text = " ".join(path_items)
    if re.search(r"사업\s*개요|사업안내|목표|추진\s*배경", text):
        return "overview"
    if re.search(r"과업|요구사항|제안\s*요청|수행\s*범위|기능\s*요구", text):
        return "requirements"
    if re.search(r"보안|개인정보|접근권한|암호화", text):
        return "security"
    if re.search(r"평가|배점|협상", text):
        return "evaluation"
    if re.search(r"입찰|계약|공모|제출", text):
        return "bid_contract"
    if re.search(r"붙임|별지|서식|양식", text):
        return "appendix_form"
    return "body"


def update_stack(stack: list[dict[str, Any]], level: int, heading: str) -> list[dict[str, Any]]:
    kept = [item for item in stack if int(item["level"]) < level]
    kept.append({"level": level, "heading": heading})
    return kept


def stack_section_path(stack: list[dict[str, Any]]) -> list[str]:
    return [str(item["heading"]) for item in stack]


def is_value_like_heading(title: str) -> bool:
    text = normalize_line(title)
    if FIELD_VALUE_RE.search(text):
        return True
    if re.search(r"[:：]\s*\S+", text) and len(text) > 12:
        return True
    if len(text) > 48 and SENTENCE_END_RE.search(text):
        return True
    if len(text) > 70:
        return True
    return False


def detect_heading(text: str, *, font_size: float = 0.0, body_font_size: float = 0.0) -> tuple[int, str] | None:
    short = normalize_line(text)
    if not short or len(short) > 140:
        return None
    if PAGE_MARKER_RE.match(short):
        return None

    match = ROMAN_HEADING_RE.match(short)
    if match:
        roman, title = match.groups()
        title = normalize_line(title)
        if title and len(title) <= 80 and not is_value_like_heading(title):
            return 1, f"{roman}. {title}"

    match = NUMBER_HEADING_RE.match(short)
    if match:
        number, marker, title = match.groups()
        title = normalize_line(title)
        if not title or is_value_like_heading(title):
            return None
        if marker == ")" and len(title) > 45:
            return None
        if marker == "." and len(title) > 80:
            return None
        return 2, f"{number}. {title}" if marker == "." else f"{number}) {title}"

    match = KOREAN_HEADING_RE.match(short)
    if match:
        letter, marker, title = match.groups()
        title = normalize_line(title)
        if not title or is_value_like_heading(title):
            return None
        if marker == ")" and len(title) > 35:
            return None
        if len(title) <= 55:
            return 3, f"{letter}{marker} {title}"

    match = CIRCLED_HEADING_RE.match(short)
    if match:
        marker, title = match.groups()
        title = normalize_line(title)
        if title and len(title) <= 55 and not is_value_like_heading(title):
            return 4, f"{marker} {title}"

    if BARE_SECTION_RE.match(short):
        return 2, short

    if body_font_size and font_size >= body_font_size + 2.0 and len(short) <= 45:
        if not is_value_like_heading(short) and not SENTENCE_END_RE.search(short):
            return 2, short
    return None


def looks_like_toc_line(line: str, page_no: int) -> bool:
    text = normalize_line(line)
    if page_no > 10 or len(text) > 130:
        return False
    return bool(TOC_LINE_RE.match(text))


def clean_grid(raw_rows: list[list[Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    max_cols = 0
    for raw_row in raw_rows:
        row = [normalize_line(cell) for cell in raw_row]
        if any(row):
            rows.append(row)
            max_cols = max(max_cols, len(row))
    if not rows or max_cols == 0:
        return []
    return [row + [""] * (max_cols - len(row)) for row in rows]


def unique_values(row: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in row:
        clean = normalize_line(value)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        values.append(clean)
    return values


def table_shape(grid: list[list[str]]) -> str:
    rows = [row for row in grid if any(normalize_line(value) for value in row)]
    if not rows:
        return "empty"
    row_count = len(rows)
    col_count = max(len(row) for row in rows)
    first_col_filled = sum(1 for row in rows if row and normalize_line(row[0]))
    first_row_filled = sum(1 for value in rows[0] if normalize_line(value))
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


def classify_table(grid: list[list[str]], path_items: list[str]) -> str:
    flat = normalize_line(" ".join(value for row in grid for value in row if value))
    context = normalize_line(" ".join(path_items))
    last_values = [row[-1] for row in grid if row and normalize_line(row[-1])]
    page_like = sum(1 for value in last_values if re.fullmatch(r"\d{1,4}", normalize_line(value)))
    if "목차" in flat or (last_values and page_like / max(1, len(last_values)) >= 0.5):
        return "toc_table"
    if re.search(r"요구사항|요구\s*ID|SFR|SER|DAR|DIR|SIR|PER|QUR|PMR|MPR", flat, re.I):
        return "requirement_table"
    if re.search(r"평가|배점|점수|평점|협상", flat + " " + context):
        return "evaluation_table"
    if re.search(r"일정|기간|월별|추진", flat + " " + context) and len(grid[0]) >= 4:
        return "schedule_table"
    if re.search(r"서식|양식|대표자|주소|서명|날인", flat + " " + context):
        return "form_table"
    if len(grid) <= 2 or len(grid[0]) <= 1:
        return "layout_table"
    return "data_table"


def header_value_rows(grid: list[list[str]]) -> list[dict[str, Any]]:
    if len(grid) < 2:
        return []
    headers = [normalize_line(value) or f"col_{idx}" for idx, value in enumerate(grid[0])]
    if sum(1 for header in headers if not header.startswith("col_")) < 2:
        return []
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(grid[1:], start=1):
        cells: dict[str, str] = {}
        for col_index, value in enumerate(row):
            clean = normalize_line(value)
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


def compact_grid_lines(grid: list[list[str]], *, max_rows: int = 16) -> list[str]:
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


def table_payload(grid: list[list[str]], table_type: str, *, group_size: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rows": len(grid),
        "cols": max((len(row) for row in grid), default=0),
        "cell_count": sum(1 for row in grid for value in row if normalize_line(value)),
        "shape": table_shape(grid),
        "grid": grid,
    }
    if table_type == "toc_table":
        payload["items"] = generic_rows(grid)
        return payload
    if table_type in {"layout_table", "form_table"}:
        payload["summary_lines"] = compact_grid_lines(grid)
        return payload
    if table_type in {"requirement_table", "evaluation_table", "data_table"}:
        rows_data = header_value_rows(grid)
        if rows_data and len(rows_data) <= 40:
            payload["rows_data"] = rows_data
        else:
            payload["row_groups"] = group_rows(generic_rows(grid, start_row=1), group_size=group_size)
        return payload
    if table_type == "schedule_table":
        payload["row_groups"] = group_rows(generic_rows(grid, start_row=1), group_size=group_size)
        return payload
    payload["row_groups"] = group_rows(generic_rows(grid), group_size=group_size)
    return payload


def payload_has_content(payload: dict[str, Any]) -> bool:
    for key in ("rows_data", "row_groups", "summary_lines", "items"):
        if isinstance(payload.get(key), list) and payload[key]:
            return True
    grid = payload.get("grid")
    return isinstance(grid, list) and any(
        isinstance(row, list) and any(normalize_line(value) for value in row) for row in grid
    )


def table_text_for_record(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("rows_data"), list):
        return "\n".join(str(row.get("text", "")) for row in payload["rows_data"][:20])
    if isinstance(payload.get("row_groups"), list):
        return "\n".join(str(group.get("text", "")) for group in payload["row_groups"][:3])
    if isinstance(payload.get("items"), list):
        return "\n".join(str(item.get("text", "")) for item in payload["items"][:30])
    if isinstance(payload.get("summary_lines"), list):
        return "\n".join(str(line) for line in payload["summary_lines"])
    return ""


def _point_in_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def _line_in_any_table(line: PdfLine, table_bboxes: list[tuple[float, float, float, float]]) -> bool:
    if not table_bboxes:
        return False
    return any(_point_in_bbox(line.x, line.y, bbox) for bbox in table_bboxes)


def _word_in_any_table(
    word: tuple[float, float, float, float, str],
    table_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    if not table_bboxes:
        return False
    x0, y0, x1, y1, _ = word
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return any(_point_in_bbox(cx, cy, bbox) for bbox in table_bboxes)


def _cluster_words_into_rows(
    words: list[tuple[float, float, float, float, str]],
    *,
    y_tolerance: float = 3.5,
) -> list[list[tuple[float, float, float, float, str]]]:
    rows: list[list[tuple[float, float, float, float, str]]] = []
    current: list[tuple[float, float, float, float, str]] = []
    current_y: float | None = None

    for word in sorted(words, key=lambda item: (((item[1] + item[3]) / 2.0), item[0])):
        y = (word[1] + word[3]) / 2.0
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current.append(word)
            current_y = y if current_y is None else (current_y * (len(current) - 1) + y) / len(current)
            continue
        rows.append(sorted(current, key=lambda item: item[0]))
        current = [word]
        current_y = y

    if current:
        rows.append(sorted(current, key=lambda item: item[0]))
    return rows


def _row_words_to_cells(
    row_words: list[tuple[float, float, float, float, str]],
) -> tuple[list[str], tuple[float, float, float, float]]:
    if not row_words:
        return [], (0.0, 0.0, 0.0, 0.0)

    gaps = [
        max(0.0, row_words[index][0] - row_words[index - 1][2])
        for index in range(1, len(row_words))
    ]
    positive_gaps = [gap for gap in gaps if gap > 0.0]
    gap_threshold = max(18.0, median(positive_gaps) * 2.8) if positive_gaps else 18.0

    cells: list[str] = []
    current_words: list[str] = []
    for index, word in enumerate(row_words):
        if index > 0:
            gap = max(0.0, word[0] - row_words[index - 1][2])
            if gap >= gap_threshold and current_words:
                cells.append(normalize_line(" ".join(current_words)))
                current_words = []
        current_words.append(str(word[4]))
    if current_words:
        cells.append(normalize_line(" ".join(current_words)))

    x0 = min(word[0] for word in row_words)
    y0 = min(word[1] for word in row_words)
    x1 = max(word[2] for word in row_words)
    y1 = max(word[3] for word in row_words)
    return [cell for cell in cells if cell], (x0, y0, x1, y1)


def _looks_like_table_row(cells: list[str]) -> bool:
    if len(cells) < 2:
        return False
    non_empty = [cell for cell in cells if normalize_line(cell)]
    if len(non_empty) < 2:
        return False
    if all(PAGE_MARKER_RE.match(cell) for cell in non_empty):
        return False
    return True


def _merge_bboxes(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _extract_word_table_candidates(
    page: Any,
    *,
    page_no: int,
    existing_bboxes: list[tuple[float, float, float, float]],
) -> list[PdfTable]:
    try:
        raw_words = page.get_text("words")
    except Exception:
        return []

    words: list[tuple[float, float, float, float, str]] = []
    for raw in raw_words:
        if len(raw) < 5:
            continue
        text = normalize_line(raw[4])
        if not text:
            continue
        word = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), text)
        if _word_in_any_table(word, existing_bboxes):
            continue
        words.append(word)

    row_infos: list[dict[str, Any]] = []
    for row_words in _cluster_words_into_rows(words):
        cells, bbox = _row_words_to_cells(row_words)
        if not _looks_like_table_row(cells):
            continue
        row_infos.append(
            {
                "cells": cells,
                "bbox": bbox,
                "y0": bbox[1],
                "y1": bbox[3],
                "cols": len(cells),
            }
        )

    candidates: list[PdfTable] = []
    current: list[dict[str, Any]] = []

    def flush_group() -> None:
        nonlocal current
        if len(current) < 3:
            current = []
            return
        max_cols = max(int(row["cols"]) for row in current)
        if max_cols < 2:
            current = []
            return
        # Avoid converting plain two-column page layouts unless there is a
        # stronger table signal.
        has_strong_row = any(int(row["cols"]) >= 3 for row in current)
        if not has_strong_row and len(current) < 5:
            current = []
            return
        grid = clean_grid([row["cells"] for row in current])
        if len(grid) >= 3 and max((len(row) for row in grid), default=0) >= 2:
            bbox = _merge_bboxes([row["bbox"] for row in current])
            candidates.append(PdfTable(rows=grid, page_no=page_no, y=bbox[1], bbox=bbox))
        current = []

    previous_y1: float | None = None
    previous_cols: int | None = None
    for row in row_infos:
        row_y0 = float(row["y0"])
        row_cols = int(row["cols"])
        vertical_gap = row_y0 - previous_y1 if previous_y1 is not None else 0.0
        compatible = (
            previous_y1 is None
            or (
                vertical_gap <= 26.0
                and (
                    previous_cols is None
                    or abs(row_cols - previous_cols) <= 1
                    or max(row_cols, previous_cols) >= 3
                )
            )
        )
        if not compatible:
            flush_group()
        current.append(row)
        previous_y1 = float(row["y1"])
        previous_cols = row_cols
    flush_group()

    return candidates


def _extract_fitz_events(input_path: Path, *, extract_tables: bool) -> tuple[list[PdfLine | PdfTable], str]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is not installed") from exc

    doc = fitz.open(str(input_path))
    events: list[PdfLine | PdfTable] = []
    all_font_sizes: list[float] = []

    for page_index, page in enumerate(doc, start=1):
        page_tables: list[PdfTable] = []
        table_bboxes: list[tuple[float, float, float, float]] = []
        if extract_tables and hasattr(page, "find_tables"):
            try:
                found = page.find_tables()
                table_objects = getattr(found, "tables", found)
                for table_object in table_objects:
                    rows = table_object.extract() or []
                    grid = clean_grid(rows)
                    if not grid:
                        continue
                    bbox_tuple = tuple(float(value) for value in getattr(table_object, "bbox", (0, 0, 0, 0)))
                    bbox = bbox_tuple if len(bbox_tuple) == 4 else None
                    y = bbox[1] if bbox else 0.0
                    page_tables.append(PdfTable(rows=grid, page_no=page_index, y=y, bbox=bbox))
                    if bbox:
                        table_bboxes.append(bbox)
            except Exception:
                page_tables = []
                table_bboxes = []

        if extract_tables:
            word_tables = _extract_word_table_candidates(
                page,
                page_no=page_index,
                existing_bboxes=table_bboxes,
            )
            for table in word_tables:
                page_tables.append(table)
                if table.bbox:
                    table_bboxes.append(table.bbox)

        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = normalize_line(" ".join(str(span.get("text", "")) for span in spans))
                if not text:
                    continue
                bbox = line.get("bbox", (0, 0, 0, 0))
                x = float(bbox[0]) if len(bbox) >= 2 else 0.0
                y = float(bbox[1]) if len(bbox) >= 2 else 0.0
                font_size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                bold = any("bold" in str(span.get("font", "")).casefold() for span in spans)
                pdf_line = PdfLine(text=text, page_no=page_index, y=y, x=x, font_size=font_size, bold=bold)
                if _line_in_any_table(pdf_line, table_bboxes):
                    continue
                all_font_sizes.append(font_size)
                events.append(pdf_line)
        events.extend(page_tables)

    body_size = median([size for size in all_font_sizes if size > 0.0]) if all_font_sizes else 0.0
    for event in events:
        if isinstance(event, PdfLine) and not event.font_size:
            event.font_size = body_size
    events.sort(key=lambda item: (item.page_no, item.y, getattr(item, "x", 0.0), 1 if isinstance(item, PdfTable) else 0))
    return events, "pymupdf"


def _extract_pypdf_events(input_path: Path) -> tuple[list[PdfLine], str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required for PDF text fallback") from exc

    reader = PdfReader(str(input_path))
    events: list[PdfLine] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for line_index, line in enumerate(text.splitlines(), start=1):
            clean = normalize_line(line)
            if clean:
                events.append(PdfLine(text=clean, page_no=page_index, y=float(line_index)))
    return events, "pypdf"


def extract_pdf_events(
    input_path: Path,
    *,
    backend: str = "auto",
    extract_tables: bool = True,
) -> tuple[list[PdfLine | PdfTable], str]:
    backend = backend.lower()
    if backend not in {"auto", "pymupdf", "pypdf"}:
        raise ValueError("backend must be one of: auto, pymupdf, pypdf")
    if backend in {"auto", "pymupdf"}:
        try:
            return _extract_fitz_events(input_path, extract_tables=extract_tables)
        except Exception:
            if backend == "pymupdf":
                raise
    return _extract_pypdf_events(input_path)


def build_prechunk_records(
    input_path: Path,
    *,
    group_size: int = 8,
    debug_headings_path: Path | None = None,
    backend: str = "auto",
    extract_tables: bool = True,
) -> list[dict[str, Any]]:
    events, used_backend = extract_pdf_events(input_path, backend=backend, extract_tables=extract_tables)
    source_file = input_path.name
    font_sizes = [event.font_size for event in events if isinstance(event, PdfLine) and event.font_size > 0.0]
    body_font_size = median(font_sizes) if font_sizes else 0.0

    records: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    text_buffer: list[str] = []
    toc_buffer: list[str] = []
    heading_debug: list[dict[str, Any]] = []
    pending_major: str | None = None
    section_index = 0
    table_index = 0
    buffer_start_page: int | None = None
    buffer_end_page: int | None = None

    def flush_toc() -> None:
        nonlocal toc_buffer
        text = "\n".join(toc_buffer).strip()
        toc_buffer = []
        if not text:
            return
        records.append(
            {
                "file_name": source_file,
                "source_format": "pdf",
                "pdf_backend": used_backend,
                "content_type": "toc",
                "section_path": [],
                "section_type": "toc",
                "heading": "toc",
                "text": text,
            }
        )

    def flush_text() -> None:
        nonlocal section_index, text_buffer, buffer_start_page, buffer_end_page
        text = "\n".join(line for line in text_buffer if normalize_line(line)).strip()
        text_buffer = []
        start_page = buffer_start_page
        end_page = buffer_end_page
        buffer_start_page = None
        buffer_end_page = None
        if not normalize_line(text):
            return
        path_items = stack_section_path(stack)
        section_index += 1
        records.append(
            {
                "file_name": source_file,
                "source_format": "pdf",
                "pdf_backend": used_backend,
                "content_type": "section_text" if path_items else "cover_text",
                "section_id": f"section_{section_index:04d}",
                "section_path": path_items,
                "section_type": section_type(path_items) if path_items else "cover",
                "heading": path_items[-1] if path_items else "",
                "page_start": start_page,
                "page_end": end_page,
                "text": text,
            }
        )

    def append_text(line: str, page_no: int) -> None:
        nonlocal buffer_start_page, buffer_end_page
        if buffer_start_page is None:
            buffer_start_page = page_no
        buffer_end_page = page_no
        text_buffer.append(line)

    def flush_pending_major_as_text(page_no: int) -> None:
        nonlocal pending_major
        if pending_major:
            append_text(pending_major, page_no)
            pending_major = None

    for event in events:
        if isinstance(event, PdfTable):
            flush_pending_major_as_text(event.page_no)
            flush_text()
            path_items = stack_section_path(stack)
            table_type = classify_table(event.rows, path_items)
            payload = table_payload(event.rows, table_type, group_size=group_size)
            if not payload_has_content(payload):
                continue
            table_index += 1
            record: dict[str, Any] = {
                "file_name": source_file,
                "source_format": "pdf",
                "pdf_backend": used_backend,
                "content_type": "toc" if table_type == "toc_table" else "table",
                "table_id": f"table_{table_index:04d}",
                "table_type": table_type,
                "section_path": path_items,
                "section_type": section_type(path_items),
                "page_start": event.page_no,
                "page_end": event.page_no,
                "table": payload,
            }
            if record["content_type"] == "toc":
                record["text"] = table_text_for_record(payload)
            records.append(record)
            continue

        line = normalize_line(event.text)
        if not line:
            continue
        if looks_like_toc_line(line, event.page_no):
            toc_buffer.append(line)
            continue
        if toc_buffer and event.page_no > 10:
            flush_toc()

        if ROMAN_ONLY_RE.match(line):
            flush_pending_major_as_text(event.page_no)
            flush_text()
            pending_major = line.rstrip(".)")
            continue

        if pending_major:
            major_title = normalize_line(line)
            if BARE_SECTION_RE.match(major_title) or (len(major_title) <= 35 and not is_value_like_heading(major_title)):
                flush_text()
                heading = f"{pending_major}. {major_title}"
                stack = update_stack(stack, 1, heading)
                heading_debug.append(
                    {"file_name": source_file, "page_no": event.page_no, "level": 1, "heading": heading}
                )
                pending_major = None
                continue
            flush_pending_major_as_text(event.page_no)

        detected = detect_heading(line, font_size=event.font_size, body_font_size=body_font_size)
        if detected:
            level, heading = detected
            flush_text()
            stack = update_stack(stack, level, heading)
            heading_debug.append(
                {"file_name": source_file, "page_no": event.page_no, "level": level, "heading": heading}
            )
        else:
            append_text(line, event.page_no)

    flush_pending_major_as_text(events[-1].page_no if events else 1)
    flush_text()
    flush_toc()

    if debug_headings_path:
        write_jsonl(debug_headings_path, heading_debug)
    return records


def discover_pdf_files(input_dir: Path, *, glob_pattern: str = "*.pdf", recursive: bool = False) -> list[Path]:
    iterator = input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern)
    return sorted(
        [path for path in iterator if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda path: str(path).casefold(),
    )


def parse_pdf_files(
    input_files: list[Path],
    *,
    group_size: int,
    stop_on_error: bool,
    backend: str = "auto",
    extract_tables: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    error_count = 0
    total = len(input_files)
    for index, input_path in enumerate(input_files, start=1):
        print(f"[{index}/{total}] parsing: {input_path.name}")
        try:
            records.extend(
                build_prechunk_records(
                    input_path,
                    group_size=group_size,
                    backend=backend,
                    extract_tables=extract_tables,
                )
            )
        except Exception as exc:
            error_count += 1
            print(f"  error: {type(exc).__name__}: {exc}")
            if stop_on_error:
                raise
    return records, error_count


def write_jsonl(path: Path, records: list[dict[str, Any]], *, limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = records if limit is None else records[:limit]
    with path.open("w", encoding="utf-8") as file:
        for record in selected:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDF files into prechunk JSONL.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/v1/raw"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", default="*.pdf")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--backend", choices=("auto", "pymupdf", "pypdf"), default="auto")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.input:
        records = build_prechunk_records(
            args.input,
            group_size=args.group_size,
            backend=args.backend,
            extract_tables=not args.no_tables,
        )
        error_count = 0
        target_count = 1
    else:
        files = discover_pdf_files(args.input_dir, glob_pattern=args.glob, recursive=args.recursive)
        if args.limit_files:
            files = files[: args.limit_files]
        records, error_count = parse_pdf_files(
            files,
            group_size=args.group_size,
            stop_on_error=args.stop_on_error,
            backend=args.backend,
            extract_tables=not args.no_tables,
        )
        target_count = len(files)
    write_jsonl(args.output, records)
    print(f"target_files: {target_count}")
    print(f"error_files: {error_count}")
    print(f"written_records: {len(records)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
