from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def normalize_text(text: object) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: object) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", normalize_text(text))


_TABLE_HTML_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_CELL_WITH_ATTR_PATTERN = re.compile(r"<(td|th)\b([^>]*)>(.*?)</(?:td|th)>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_KV_LINE_PATTERN = re.compile(r"^\s*([^\n\r:：]{1,40})\s*[:：]\s*(.+)\s*$")
_SPAN_PATTERN = re.compile(r'\b(colspan|rowspan)\s*=\s*["\']?(\d+)["\']?', re.IGNORECASE)
_HEADER_HINT_PATTERN = re.compile(r"(항목|의견|기간|구분|내용|비고|코드|수량|금액|일자|번호|명)", re.IGNORECASE)
_DURATION_TOKEN_PATTERN = re.compile(
    r"(\d+\s*(?:개월|달|년|주|일|시간|분|초|month|months|year|years|week|weeks|day|days))",
    re.IGNORECASE,
)


def _strip_html_tags(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    text = _TAG_PATTERN.sub(" ", text)
    text = html.unescape(text)
    return normalize_space(text)


def _extract_table_html_candidates(text: str) -> list[str]:
    if not text:
        return []
    return [candidate.strip() for candidate in _TABLE_HTML_PATTERN.findall(text) if candidate.strip()]


def _parse_table_rows(table_html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_html in _ROW_PATTERN.findall(table_html):
        cells = [_strip_html_tags(cell) for cell in _CELL_PATTERN.findall(row_html)]
        cleaned = [cell for cell in cells if cell]
        if cleaned:
            rows.append(cleaned)
    return rows


def _parse_span(attrs: str, name: str, default: int = 1) -> int:
    for key, value in _SPAN_PATTERN.findall(attrs or ""):
        if key.lower() != name.lower():
            continue
        try:
            parsed = int(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default
    return default


def _normalize_row_cells(row: list[str]) -> list[str]:
    normalized = [normalize_space(cell) for cell in row]
    while normalized and not normalized[-1]:
        normalized.pop()
    return normalized


def _table_to_grid(table_html: str) -> list[list[str]]:
    row_html_list = _ROW_PATTERN.findall(table_html)
    parsed_rows: list[list[tuple[str, str, str]]] = []
    for row_html in row_html_list:
        parsed_rows.append(_CELL_WITH_ATTR_PATTERN.findall(row_html))

    if not parsed_rows:
        return []

    max_cols = 0
    rowspans: dict[int, int] = {}
    for row_cells in parsed_rows:
        col = 0
        upcoming_rowspans: dict[int, int] = {}
        for _, attrs, _ in row_cells:
            while rowspans.get(col, 0) > 0:
                col += 1
            colspan = _parse_span(attrs, "colspan", 1)
            rowspan = _parse_span(attrs, "rowspan", 1)
            for offset in range(colspan):
                if rowspan > 1:
                    upcoming_rowspans[col + offset] = max(upcoming_rowspans.get(col + offset, 0), rowspan - 1)
            col += colspan

        max_cols = max(max_cols, col)
        next_rowspans: dict[int, int] = {}
        for c_idx, remaining in rowspans.items():
            if remaining - 1 > 0:
                next_rowspans[c_idx] = remaining - 1
        for c_idx, remaining in upcoming_rowspans.items():
            next_rowspans[c_idx] = max(next_rowspans.get(c_idx, 0), remaining)
        rowspans = next_rowspans

    if max_cols <= 0:
        return []

    grid = [[""] * max_cols for _ in parsed_rows]
    rowspans = {}
    for r_idx, row_cells in enumerate(parsed_rows):
        col = 0
        upcoming_rowspans: dict[int, int] = {}
        for _, attrs, inner in row_cells:
            while rowspans.get(col, 0) > 0:
                col += 1

            colspan = _parse_span(attrs, "colspan", 1)
            rowspan = _parse_span(attrs, "rowspan", 1)
            value = _strip_html_tags(inner)

            # Keep anchor text only (top-left) to avoid value duplication artifacts.
            if col < max_cols:
                grid[r_idx][col] = value
            for offset in range(colspan):
                if rowspan > 1:
                    upcoming_rowspans[col + offset] = max(upcoming_rowspans.get(col + offset, 0), rowspan - 1)
            col += colspan

        next_rowspans: dict[int, int] = {}
        for c_idx, remaining in rowspans.items():
            if remaining - 1 > 0:
                next_rowspans[c_idx] = remaining - 1
        for c_idx, remaining in upcoming_rowspans.items():
            next_rowspans[c_idx] = max(next_rowspans.get(c_idx, 0), remaining)
        rowspans = next_rowspans

    return [_normalize_row_cells(row) for row in grid]


def _compress_consecutive_cells(row: list[str]) -> list[tuple[str, int, int]]:
    if not row:
        return []
    grouped: list[tuple[str, int, int]] = []
    start = 0
    prev = normalize_space(row[0])
    for idx in range(1, len(row)):
        cur = normalize_space(row[idx])
        if cur == prev:
            continue
        grouped.append((prev, start, idx - 1))
        start = idx
        prev = cur
    grouped.append((prev, start, len(row) - 1))
    return grouped


def _guess_header_row_index(grid: list[list[str]]) -> int:
    if not grid:
        return -1
    for idx, row in enumerate(grid):
        non_empty = [normalize_space(cell) for cell in row if normalize_space(cell)]
        if len(non_empty) < 2:
            continue
        unique = []
        for cell in non_empty:
            if cell not in unique:
                unique.append(cell)
        if len(unique) < 2:
            continue

        avg_len = sum(len(cell) for cell in unique) / len(unique)
        has_hint = any(_HEADER_HINT_PATTERN.search(cell) for cell in unique)
        if avg_len > 18 and not has_hint:
            continue

        next_non_empty = []
        if idx + 1 < len(grid):
            next_non_empty = [normalize_space(cell) for cell in grid[idx + 1] if normalize_space(cell)]
        if next_non_empty and has_hint:
            return idx
        if has_hint and len(unique) >= 3:
            return idx
    return -1


def _parse_table_layout(table_html: str) -> tuple[list[list[dict[str, int | str]]], int]:
    row_html_list = _ROW_PATTERN.findall(table_html)
    parsed_rows: list[list[tuple[str, str, str]]] = []
    for row_html in row_html_list:
        parsed_rows.append(_CELL_WITH_ATTR_PATTERN.findall(row_html))

    layout: list[list[dict[str, int | str]]] = []
    max_cols = 0
    rowspans: dict[int, int] = {}
    for row_cells in parsed_rows:
        row_layout: list[dict[str, int | str]] = []
        col = 0
        upcoming_rowspans: dict[int, int] = {}
        for _, attrs, inner in row_cells:
            while rowspans.get(col, 0) > 0:
                col += 1
            colspan = _parse_span(attrs, "colspan", 1)
            rowspan = _parse_span(attrs, "rowspan", 1)
            start = col
            end = col + colspan - 1
            row_layout.append(
                {
                    "text": _strip_html_tags(inner),
                    "start": start,
                    "end": end,
                }
            )
            if rowspan > 1:
                for cc in range(start, end + 1):
                    upcoming_rowspans[cc] = max(upcoming_rowspans.get(cc, 0), rowspan - 1)
            col = end + 1

        max_cols = max(max_cols, col)
        layout.append(row_layout)

        next_rowspans: dict[int, int] = {}
        for c_idx, remaining in rowspans.items():
            if remaining - 1 > 0:
                next_rowspans[c_idx] = remaining - 1
        for c_idx, remaining in upcoming_rowspans.items():
            next_rowspans[c_idx] = max(next_rowspans.get(c_idx, 0), remaining)
        rowspans = next_rowspans

    return layout, max_cols


def _guess_header_row_index_from_layout(layout: list[list[dict[str, int | str]]], max_cols: int) -> int:
    best_idx = -1
    best_score = -1.0
    for idx, row_cells in enumerate(layout):
        values = [normalize_space(str(cell.get("text", ""))) for cell in row_cells]
        values = [value for value in values if value]
        if len(values) < 2:
            continue
        unique_values: list[str] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        if len(unique_values) < 2:
            continue

        hint_count = sum(1 for value in unique_values if _HEADER_HINT_PATTERN.search(value))
        short_count = sum(1 for value in unique_values if len(value) <= 12)
        avg_len = sum(len(value) for value in unique_values) / len(unique_values)
        long_count = sum(1 for value in unique_values if len(value) >= 24)
        cell_count = len(row_cells)
        covered_cols = sum(max(0, int(cell.get("end", 0)) - int(cell.get("start", 0)) + 1) for cell in row_cells)
        coverage_ratio = covered_cols / max(1, max_cols)

        score = 0.0
        score += hint_count * 4.0
        score += short_count * 1.0
        score += cell_count * 2.0
        if cell_count >= 3:
            score += 2.0
        if avg_len > 20:
            score -= 2.0
        if long_count > 0 and cell_count <= 2:
            score -= 3.0
        if coverage_ratio < 0.6:
            score -= 2.0

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def _extract_duration_token(text: str) -> str:
    if not text:
        return ""
    candidates: list[str] = []
    has_full_date = bool(re.search(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", text))
    for match in _DURATION_TOKEN_PATTERN.finditer(text):
        token = normalize_space(match.group(1))
        # Avoid taking day-of-month token from full date strings.
        if has_full_date and token.endswith("일"):
            continue
        candidates.append(token)
    return candidates[-1] if candidates else ""


def _extract_row_maps_from_table_html(table_html: str) -> list[dict[str, str]]:
    layout, max_cols = _parse_table_layout(table_html)
    if not layout or max_cols <= 0:
        return []
    header_idx = _guess_header_row_index_from_layout(layout, max_cols)
    if header_idx < 0 or header_idx >= len(layout):
        return []

    header_groups: list[tuple[str, int, int]] = []
    for cell in layout[header_idx]:
        header_name = normalize_space(str(cell.get("text", "")))
        if not header_name:
            continue
        start = int(cell.get("start", 0))
        end = int(cell.get("end", start))
        header_groups.append((header_name, start, end))
    if len(header_groups) < 2:
        return []

    row_maps: list[dict[str, str]] = []
    for row_cells in layout[header_idx + 1 :]:
        if not row_cells:
            continue
        row_map: dict[str, str] = {}
        for header_name, start, end in header_groups:
            best_value = ""
            best_score = -1.0
            header_width = max(1, end - start + 1)
            for cell in row_cells:
                cell_text = normalize_space(str(cell.get("text", "")))
                if not cell_text:
                    continue
                cell_start = int(cell.get("start", 0))
                cell_end = int(cell.get("end", cell_start))
                overlap = max(0, min(end, cell_end) - max(start, cell_start) + 1)
                if overlap <= 0:
                    continue

                cell_width = max(1, cell_end - cell_start + 1)
                coverage = overlap / header_width
                focus = overlap / cell_width
                score = (coverage * 2.0) + focus
                if score > best_score:
                    best_score = score
                    best_value = cell_text
            row_map[header_name] = best_value

        if any(value for value in row_map.values()):
            # If 기간-like column is empty but 의견-like column contains duration token,
            # recover a minimal value without per-doc hardcoding.
            duration_headers = [key for key in row_map if _HEADER_HINT_PATTERN.search(key) and "기간" in key]
            opinion_headers = [key for key in row_map if "의견" in key]
            for d_key in duration_headers:
                duration_value = row_map.get(d_key, "")
                if duration_value and opinion_headers:
                    same_as_opinion = any(duration_value == row_map.get(o_key, "") for o_key in opinion_headers)
                    if same_as_opinion or len(duration_value) > 20:
                        reduced = _extract_duration_token(duration_value)
                        if reduced:
                            row_map[d_key] = reduced
                        elif same_as_opinion:
                            row_map[d_key] = ""

                if row_map.get(d_key):
                    continue

                candidate = ""
                for o_key in opinion_headers:
                    candidate = _extract_duration_token(row_map.get(o_key, ""))
                    if candidate:
                        break
                if candidate:
                    row_map[d_key] = candidate
            row_maps.append(row_map)
    return row_maps


def _find_header_key(row_map: dict[str, str], keywords: tuple[str, ...]) -> str | None:
    if not row_map:
        return None
    normalized_targets = [compact_text(word) for word in keywords if word]
    for key in row_map:
        key_compact = compact_text(key)
        if all(target in key_compact for target in normalized_targets):
            return key
    return None


def _extract_fields_from_row_maps(row_maps: list[dict[str, str]]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for row_map in row_maps:
        item_key = (
            _find_header_key(row_map, ("검토", "항목"))
            or _find_header_key(row_map, ("구분",))
            or _find_header_key(row_map, ("항목",))
        )
        opinion_key = (
            _find_header_key(row_map, ("검토", "의견"))
            or _find_header_key(row_map, ("의견",))
            or _find_header_key(row_map, ("내용",))
        )
        period_key = (
            _find_header_key(row_map, ("사업", "기간"))
            or _find_header_key(row_map, ("추정", "기간"))
            or _find_header_key(row_map, ("기간",))
        )

        if item_key and opinion_key:
            item_value = normalize_space(row_map.get(item_key, ""))
            # For review-style tables, keep only real item rows.
            if not item_value or not _looks_like_bullet_item(item_value):
                continue

        if item_key and row_map.get(item_key):
            _append_field(fields, "검토항목", _strip_leading_marker(row_map[item_key]))
        if opinion_key and row_map.get(opinion_key):
            _append_field(fields, "검토의견", row_map[opinion_key])
        if period_key and row_map.get(period_key):
            _append_field(fields, "추정 사업기간", row_map[period_key])

    return fields


def _parse_labeled_block(block_text: str) -> tuple[str, str] | None:
    label_match = re.search(r"label:\s*([^\n\r]+)", block_text, flags=re.IGNORECASE)
    content_match = re.search(r"content:\s*(.*)", block_text, flags=re.IGNORECASE | re.DOTALL)
    if not label_match or not content_match:
        return None
    label = normalize_space(label_match.group(1)).lower()
    content = normalize_space(content_match.group(1).replace("#################", " "))
    if not label or not content:
        return None
    return label, content


def _extract_labeled_blocks(pred_raw_item: dict) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    raw_output = pred_raw_item.get("raw_pipeline_output")
    if isinstance(raw_output, list):
        for page in raw_output:
            if not isinstance(page, dict):
                continue
            parsing_items = page.get("parsing_res_list")
            if not isinstance(parsing_items, list):
                continue
            for item in parsing_items:
                if isinstance(item, str):
                    parsed = _parse_labeled_block(item)
                    if parsed:
                        blocks.append(parsed)
    if blocks:
        return blocks

    # Fallback: parse label/content pairs from ocr_lines plain text.
    current_label = ""
    for line in pred_raw_item.get("ocr_lines", []):
        if not isinstance(line, dict):
            continue
        text = normalize_space(str(line.get("text", "")))
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("label:"):
            current_label = normalize_space(text.split(":", 1)[1]).lower()
            continue
        if lowered.startswith("content:"):
            content = normalize_space(text.split(":", 1)[1])
            if current_label and content:
                blocks.append((current_label, content))
            current_label = ""
    return blocks


def _strip_leading_marker(text: str) -> str:
    stripped = re.sub(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", text)
    stripped = re.sub(r"^\s*\d+[.)]\s*", "", stripped)
    stripped = re.sub(r"^\s*[•·\-]\s*", "", stripped)
    return normalize_space(stripped)


def _canonicalize_key(text: str) -> str:
    key = _strip_leading_marker(text)
    key = key.strip(":-–— ")
    key = normalize_space(key)
    if not key:
        return ""
    if len(compact_text(key)) < 2:
        return ""
    if len(key) > 40:
        return ""
    if re.fullmatch(r"[#*=\-_/\\\s]+", key):
        return ""
    return key


def _looks_like_bullet_item(text: str) -> bool:
    return bool(re.match(r"^\s*([①②③④⑤⑥⑦⑧⑨⑩]|\d+[.)]|[•·\-])\s*", text))


def _append_field(structure: dict[str, list[str]], key: str, value: str) -> None:
    canonical_key = _canonicalize_key(key)
    value = normalize_space(value)
    if not canonical_key or not value:
        return
    values = structure.setdefault(canonical_key, [])
    if value not in values:
        values.append(value)


def _extract_fields_from_table_rows(rows: list[list[str]]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    pending_single_key: str | None = None
    for row in rows:
        if not row:
            continue
        if len(row) == 1:
            single_line = normalize_space(row[0])
            if re.search(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", single_line):
                _append_field(fields, "작성일", single_line)
            if single_line.endswith("귀하"):
                _append_field(fields, "수신", single_line)

        if len(row) >= 2:
            # Ignore likely header rows (column names only).
            if all(len(normalize_space(cell)) <= 12 for cell in row) and any(
                token in normalize_space(" ".join(row)) for token in ("항목", "의견", "구분", "내용")
            ):
                pending_single_key = None
                continue
            key = _canonicalize_key(row[0])
            value = normalize_space(" ".join(row[1:]))
            if key and value and compact_text(key) != compact_text(value):
                _append_field(fields, key, value)
            pending_single_key = None
            if _looks_like_bullet_item(row[0]):
                _append_field(fields, "검토항목", _strip_leading_marker(row[0]))
            continue

        single = normalize_space(row[0])
        if not single:
            continue
        if _looks_like_bullet_item(single):
            _append_field(fields, "검토항목", _strip_leading_marker(single))
            pending_single_key = None
            continue

        single_key = _canonicalize_key(single)
        if pending_single_key and single_key and compact_text(pending_single_key) != compact_text(single):
            _append_field(fields, pending_single_key, single)
            pending_single_key = None
            continue

        if single_key and len(single_key) <= 20:
            pending_single_key = single_key
        else:
            pending_single_key = None
    return fields


def build_pred_structure_from_ocr(pred_text: str, pred_raw_item: dict) -> dict[str, list[str]]:
    structure: dict[str, list[str]] = {}
    blocks = _extract_labeled_blocks(pred_raw_item)

    table_html_candidates: list[str] = []
    title_candidates: list[str] = []
    footnote_candidates: list[str] = []
    free_text_candidates: list[str] = []

    for label, content in blocks:
        label_key = normalize_space(label).lower()
        if "table" in label_key:
            table_html_candidates.extend(_extract_table_html_candidates(content))
            continue
        cleaned = _strip_html_tags(content)
        if not cleaned:
            continue
        if "title" in label_key:
            title_candidates.append(cleaned)
        elif "footnote" in label_key:
            footnote_candidates.append(cleaned)
        else:
            free_text_candidates.append(cleaned)

    # Fallback when parsing_res_list blocks are unavailable.
    if not table_html_candidates:
        table_html_candidates.extend(_extract_table_html_candidates(pred_text))

    if title_candidates:
        for title in dedupe_keep_order(title_candidates):
            _append_field(structure, "문서명", title)

    for table_html in dedupe_keep_order(table_html_candidates):
        rows = _parse_table_rows(table_html)
        table_fields = _extract_fields_from_table_rows(rows)
        for key, values in table_fields.items():
            for value in values:
                _append_field(structure, key, value)

        row_maps = _extract_row_maps_from_table_html(table_html)
        row_map_fields = _extract_fields_from_row_maps(row_maps)
        for key, values in row_map_fields.items():
            for value in values:
                _append_field(structure, key, value)

    searchable_texts = free_text_candidates + footnote_candidates
    for text in searchable_texts:
        if not text:
            continue
        for raw_line in re.split(r"[\n\r]+", text):
            line = normalize_space(raw_line)
            if not line:
                continue
            kv_match = _KV_LINE_PATTERN.match(line)
            if kv_match:
                _append_field(structure, kv_match.group(1), kv_match.group(2))

    return structure


def load_item_by_id(path: Path, target_id: str) -> dict:
    payload = _load_json_or_jsonl(path)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("id") == target_id:
                return item
        raise ValueError(f"id not found in {path}: {target_id}")
    if isinstance(payload, dict):
        if payload.get("id") == target_id:
            return payload
        raise ValueError(f"id mismatch in {path}: expected={target_id}, found={payload.get('id')}")
    raise ValueError(f"Unsupported JSON shape in {path}: {type(payload).__name__}")


def _load_json_or_jsonl(path: Path) -> dict | list:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return []
    decoder = json.JSONDecoder()
    pos = 0
    top_level: list[dict | list] = []
    while pos < len(raw):
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos >= len(raw):
            break
        item, next_pos = decoder.raw_decode(raw, pos)
        top_level.append(item)
        pos = next_pos

    if len(top_level) == 1:
        only = top_level[0]
        if isinstance(only, (dict, list)):
            return only
        raise ValueError(f"Unsupported JSON shape in {path}: {type(only).__name__}")

    rows: list[dict] = []
    for idx, block in enumerate(top_level, start=1):
        if isinstance(block, dict):
            rows.append(block)
            continue
        if isinstance(block, list):
            for row in block:
                if isinstance(row, dict):
                    rows.append(row)
                    continue
                raise ValueError(f"Unsupported item type in block {idx} at {path}: {type(row).__name__}")
            continue
        raise ValueError(f"Unsupported block type {idx} in {path}: {type(block).__name__}")
    return rows


def load_pred_raw(path: Path, target_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("id") and payload.get("id") != target_id:
            raise ValueError(f"id mismatch in {path}: expected={target_id}, found={payload.get('id')}")
        return payload
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("id") == target_id:
                return item
        raise ValueError(f"id not found in {path}: {target_id}")
    raise ValueError(f"Unsupported JSON shape in {path}: {type(payload).__name__}")


def build_pred_structured(gt_item: dict, pred_raw_item: dict, score_threshold: float) -> dict:
    if "ocr_lines" not in pred_raw_item:
        raise ValueError("Pred raw JSON missing key: ocr_lines")

    all_lines = pred_raw_item["ocr_lines"]
    if not isinstance(all_lines, list):
        raise ValueError("Pred raw JSON key `ocr_lines` must be a list")

    kept: list[str] = []
    for line in all_lines:
        if not isinstance(line, dict):
            continue
        text = normalize_space(str(line.get("text", "")))
        if not text:
            continue
        score = line.get("score")
        if score is not None and float(score) < score_threshold:
            continue
        kept.append(text)

    kept = dedupe_keep_order(kept)
    pred_text = " ".join(kept).strip()
    meta_keys = [
        "id",
        "source_file_name",
        "source_file_type",
        "source_doc_key",
        "original_image_file_name",
        "image_file_name",
        "image_path",
        "image_type",
        "use_eval",
    ]
    result = {key: gt_item.get(key) for key in meta_keys}
    result["pred_text"] = pred_text
    image_type = str(gt_item.get("image_type", "unknown")).strip().lower()
    pred_structure = build_pred_structure_from_ocr(pred_text, pred_raw_item)
    # [Design Intent]
    # Ensure non-empty structured output for chart/diagram/table families.
    # This prevents invalid eval states (pred_total=0) when OCR text exists
    # but rule-based structure extraction misses the shape.
    if not pred_structure and pred_text:
        if image_type == "chart":
            pred_structure = {"차트_텍스트": [pred_text]}
        elif image_type == "diagram":
            pred_structure = {"다이어그램_텍스트": [pred_text]}
        elif image_type in {"table", "table_form"}:
            pred_structure = {"테이블_텍스트": [pred_text]}
        else:
            pred_structure = {"텍스트": [pred_text]}
    result["pred_structure"] = pred_structure
    result["type"] = gt_item.get("image_type", "unknown")
    result["status"] = pred_raw_item.get("status", "success" if kept else "empty")
    result["latency_ms"] = pred_raw_item.get("latency_ms")
    result["model"] = pred_raw_item.get("model", "paddleocr")
    result["meta"] = {
        "score_threshold": score_threshold,
        "kept_line_count": len(kept),
        "raw_line_count": len(all_lines),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OCR raw JSON to GT-aligned structured prediction JSON")
    parser.add_argument("--gt", required=True, help="GT path (.json/.jsonl)")
    parser.add_argument("--pred-raw", required=True, help="Raw OCR JSON path from run_paddle_ocr.py")
    parser.add_argument("--id", required=True, help="Target item id")
    parser.add_argument("--output", required=True, help="Output pred_structured JSON path")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum OCR confidence threshold")
    args = parser.parse_args()

    gt_item = load_item_by_id(Path(args.gt), args.id)
    pred_raw_item = load_pred_raw(Path(args.pred_raw), args.id)
    pred_structured = build_pred_structured(gt_item, pred_raw_item, score_threshold=args.score_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pred_structured, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output: {output}")
    print(f"pred_text: {pred_structured['pred_text']}")
    print(f"pred_structure: {json.dumps(pred_structured['pred_structure'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
