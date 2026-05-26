from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Callable

_TABLE_HTML_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(r"<(th|td)\b([^>]*)>(.*?)</(?:th|td)>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_SPAN_PATTERN = re.compile(r'\b(colspan|rowspan)\s*=\s*["\']?(\d+)["\']?', re.IGNORECASE)
_DURATION_TOKEN_PATTERN = re.compile(
    r"(\d+\s*(?:개월|달|년|주|일|시간|분|초|month|months|year|years|week|weeks|day|days))",
    re.IGNORECASE,
)
_PERIOD_LABEL_PATTERN = re.compile(r"(사업기간|추정\s*사업기간|적정\s*사업기간|기간)", re.IGNORECASE)


def _normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def _strip_html_tags(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    text = _TAG_PATTERN.sub(" ", text)
    text = html.unescape(text)
    return _normalize_space(text)


def _extract_table_html_from_text(text: str) -> list[str]:
    if not text:
        return []
    return [candidate.strip() for candidate in _TABLE_HTML_PATTERN.findall(text) if candidate.strip()]


def _extract_table_html_from_obj(value: object) -> list[str]:
    tables: list[str] = []
    if value is None:
        return tables
    if isinstance(value, str):
        tables.extend(_extract_table_html_from_text(value))
        return tables
    if isinstance(value, dict):
        for nested in value.values():
            tables.extend(_extract_table_html_from_obj(nested))
        return tables
    if isinstance(value, (list, tuple)):
        for nested in value:
            tables.extend(_extract_table_html_from_obj(nested))
        return tables
    return tables


def _dedupe_table_html(candidates: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = re.sub(r"\s+", " ", candidate).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate.strip())
    return deduped


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _parse_labeled_block(block_text: str) -> tuple[str, str] | None:
    label_match = re.search(r"label:\s*([^\n\r]+)", block_text, flags=re.IGNORECASE)
    content_match = re.search(r"content:\s*(.*)", block_text, flags=re.IGNORECASE | re.DOTALL)
    if not label_match or not content_match:
        return None
    label = _normalize_space(label_match.group(1)).lower()
    content = _normalize_space(content_match.group(1).replace("#################", " "))
    if not label or not content:
        return None
    return label, content


def _extract_table_footnotes(pred_raw: dict[str, Any], pred_structured: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    raw_output = pred_raw.get("raw_pipeline_output")
    if isinstance(raw_output, list):
        for page in raw_output:
            if not isinstance(page, dict):
                continue
            parsing_items = page.get("parsing_res_list")
            if not isinstance(parsing_items, list):
                continue
            for item in parsing_items:
                if not isinstance(item, str):
                    continue
                parsed = _parse_labeled_block(item)
                if not parsed:
                    continue
                label, content = parsed
                if "footnote" not in label:
                    continue
                cleaned = _strip_html_tags(content)
                if cleaned and cleaned not in notes:
                    notes.append(cleaned)

    if notes:
        return notes

    pred_text = str(pred_structured.get("pred_text", ""))
    for match in re.finditer(
        r"label:\s*([a-zA-Z0-9_\- ]+)\s+bbox:.*?content:\s*(.*?)(?=\s+label:|\s+number\s+footnote|$)",
        pred_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        label = _normalize_space(match.group(1)).lower()
        if "footnote" not in label:
            continue
        cleaned = _strip_html_tags(match.group(2))
        if cleaned and cleaned not in notes:
            notes.append(cleaned)
    return notes


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


def _extract_duration_token(text: str) -> str:
    if not text:
        return ""
    has_full_date = bool(re.search(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", text))
    candidates: list[str] = []
    for match in _DURATION_TOKEN_PATTERN.finditer(text):
        token = _normalize_space(match.group(1))
        if has_full_date and token.endswith("일"):
            continue
        candidates.append(token)
    return candidates[-1] if candidates else ""


def _looks_like_period_header(name: str) -> bool:
    return "기간" in name


def _looks_like_opinion_header(name: str) -> bool:
    return ("의견" in name) or ("내용" in name)


def _looks_like_item_header(name: str) -> bool:
    return ("항목" in name) or ("구분" in name)


def _is_review_item_row_text(text: str) -> bool:
    value = _normalize_space(text)
    if not value:
        return False
    # Keep carry-forward only for list-like review rows, not signature/footer rows.
    return bool(re.match(r"^\s*([①②③④⑤⑥⑦⑧⑨⑩]|\d+[.)])\s*", value))


def _looks_like_period_label_text(text: str) -> bool:
    value = _normalize_space(text)
    if not value:
        return False
    if len(value) > 20:
        return False
    if _extract_duration_token(value):
        return False
    if not _PERIOD_LABEL_PATTERN.search(value):
        return False
    return True


def _looks_like_compact_period_cell(text: str) -> bool:
    value = _normalize_space(text)
    if not value:
        return False
    if len(value) > 16:
        return False
    # Compact period cells are typically short (e.g., "7개월", "6 개월")
    if value.count(" ") > 2:
        return False
    return True


def _postprocess_row_records_period(
    *,
    records: list[dict[str, Any]],
    header_groups: list[tuple[str, int, int]],
) -> None:
    header_names = [name for name, _, _ in header_groups]
    period_headers = [name for name in header_names if _looks_like_period_header(name)]
    if not period_headers:
        return
    opinion_headers = [name for name in header_names if _looks_like_opinion_header(name)]
    item_headers = [name for name in header_names if _looks_like_item_header(name)]

    last_known_period = ""
    for record in records:
        if "_section" in record:
            continue
        value_source = record.setdefault("_value_source", {})
        period_raw = record.setdefault("_period_raw", {})
        period_label = record.setdefault("_period_label", {})
        item_values = [_normalize_space(str(record.get(name, ""))) for name in item_headers]
        has_item_value = any(item_values)
        has_review_item_marker = any(_is_review_item_row_text(value) for value in item_values)

        for p_name in period_headers:
            raw_period = _normalize_space(str(record.get(p_name, "")))
            period_raw[p_name] = raw_period
            direct_token = _extract_duration_token(raw_period)
            period_token = direct_token
            source = "direct_cell" if direct_token else "empty"

            if direct_token and not _looks_like_compact_period_cell(raw_period):
                source = "inferred_token"

            if raw_period and not direct_token and _looks_like_period_label_text(raw_period):
                period_label[p_name] = raw_period

            if not period_token:
                for o_name in opinion_headers:
                    opinion_text = _normalize_space(str(record.get(o_name, "")))
                    token_from_opinion = _extract_duration_token(opinion_text)
                    period_token = token_from_opinion
                    if period_token:
                        source = "inferred_token"
                        break

            if not period_token and has_item_value and has_review_item_marker and last_known_period:
                period_token = last_known_period
                source = "carried_forward"

            record[p_name] = period_token
            value_source[p_name] = source
            if period_token and has_review_item_marker:
                last_known_period = period_token


def _parse_table_layout(table_html: str) -> tuple[list[list[dict[str, int | str]]], int]:
    row_html_list = _ROW_PATTERN.findall(table_html)
    parsed_rows: list[list[tuple[str, str, str]]] = []
    for row_html in row_html_list:
        parsed_rows.append(_CELL_PATTERN.findall(row_html))

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
            row_layout.append({"text": _strip_html_tags(inner), "start": start, "end": end})

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


def _guess_header_row_index(layout: list[list[dict[str, int | str]]], max_cols: int) -> int:
    best_idx = -1
    best_score = -1.0
    for idx, row_cells in enumerate(layout):
        values = [_normalize_space(str(cell.get("text", ""))) for cell in row_cells]
        values = [value for value in values if value]
        if len(values) < 2:
            continue
        unique_values: list[str] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        if len(unique_values) < 2:
            continue

        header_hint_count = sum(1 for value in unique_values if re.search(r"(항목|의견|기간|구분|내용|명|번호)", value))
        short_count = sum(1 for value in unique_values if len(value) <= 12)
        long_count = sum(1 for value in unique_values if len(value) >= 24)
        cell_count = len(row_cells)
        covered_cols = sum(max(0, int(cell.get("end", 0)) - int(cell.get("start", 0)) + 1) for cell in row_cells)
        coverage_ratio = covered_cols / max(1, max_cols)

        score = 0.0
        score += header_hint_count * 4.0
        score += short_count * 1.0
        score += cell_count * 2.0
        if cell_count >= 3:
            score += 2.0
        if long_count > 0 and cell_count <= 2:
            score -= 3.0
        if coverage_ratio < 0.6:
            score -= 2.0

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def _unique_headers(header_names: list[str]) -> list[str]:
    used: dict[str, int] = {}
    output: list[str] = []
    for idx, raw_name in enumerate(header_names, start=1):
        base = _normalize_space(raw_name) or f"col_{idx}"
        count = used.get(base, 0)
        used[base] = count + 1
        output.append(base if count == 0 else f"{base}_{count + 1}")
    return output


def _build_row_records_from_table(table_html: str, table_index: int) -> list[dict[str, Any]]:
    layout, max_cols = _parse_table_layout(table_html)
    if not layout or max_cols <= 0:
        return []

    header_idx = _guess_header_row_index(layout, max_cols)
    header_groups: list[tuple[str, int, int]] = []
    body_start_idx = 0

    if 0 <= header_idx < len(layout):
        header_names: list[str] = []
        header_ranges: list[tuple[int, int]] = []
        for cell in layout[header_idx]:
            name = _normalize_space(str(cell.get("text", "")))
            start = int(cell.get("start", 0))
            end = int(cell.get("end", start))
            if not name:
                continue
            header_names.append(name)
            header_ranges.append((start, end))

        if header_names:
            uniq_names = _unique_headers(header_names)
            header_groups = [
                (name, start, end) for name, (start, end) in zip(uniq_names, header_ranges, strict=False)
            ]
            body_start_idx = header_idx + 1

    if not header_groups:
        header_groups = [(f"col_{i + 1}", i, i) for i in range(max_cols)]
        body_start_idx = 0

    records: list[dict[str, Any]] = []
    for row_idx, row_cells in enumerate(layout[body_start_idx:], start=body_start_idx + 1):
        if not row_cells:
            continue

        # Keep section rows (single full-width cell) separately.
        if len(row_cells) == 1:
            section_text = _normalize_space(str(row_cells[0].get("text", "")))
            section_start = int(row_cells[0].get("start", 0))
            section_end = int(row_cells[0].get("end", section_start))
            if section_text and section_start == 0 and section_end >= max_cols - 1:
                records.append(
                    {
                        "table_index": table_index,
                        "row_index": row_idx,
                        "_section": section_text,
                    }
                )
                continue

        record: dict[str, Any] = {
            "table_index": table_index,
            "row_index": row_idx,
        }
        non_empty_value_count = 0
        for header_name, start, end in header_groups:
            best_value = ""
            best_score = -1.0
            header_width = max(1, end - start + 1)
            for cell in row_cells:
                cell_text = _normalize_space(str(cell.get("text", "")))
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
                score = coverage * 2.0 + focus
                if score > best_score:
                    best_score = score
                    best_value = cell_text
            record[header_name] = best_value
            if best_value:
                non_empty_value_count += 1

        if non_empty_value_count > 0:
            records.append(record)

    _postprocess_row_records_period(
        records=records,
        header_groups=header_groups,
    )

    return records


def _iter_data_headers(row: dict[str, Any]) -> list[str]:
    headers: list[str] = []
    for key in row.keys():
        if key in {"table_index", "row_index"}:
            continue
        if key.startswith("_"):
            continue
        headers.append(key)
    return headers


def _pick_first_header(headers: list[str], matcher: Callable[[str], bool]) -> str:
    for name in headers:
        if matcher(name):
            return name
    return ""


def _source_rank(source: str) -> int:
    return {
        "direct_cell": 3,
        "inferred_token": 2,
        "carried_forward": 1,
        "empty": 0,
    }.get(source, 0)


def _build_logical_row(
    row: dict[str, Any],
    *,
    item_header: str,
    opinion_header: str,
    period_header: str,
) -> dict[str, Any]:
    headers = _iter_data_headers(row)
    cells = {header: _normalize_space(str(row.get(header, ""))) for header in headers}

    period_value = _normalize_space(str(row.get(period_header, ""))) if period_header else ""
    value_source_map = row.get("_value_source", {})
    label_map = row.get("_period_label", {})
    source = "empty"
    if isinstance(value_source_map, dict) and period_header:
        source = _normalize_space(str(value_source_map.get(period_header, ""))) or "empty"
    period_label = ""
    if isinstance(label_map, dict) and period_header:
        period_label = _normalize_space(str(label_map.get(period_header, "")))

    return {
        "row_index": int(row.get("row_index", 0)),
        "cells": cells,
        "검토항목": _normalize_space(str(row.get(item_header, ""))) if item_header else "",
        "검토의견": _normalize_space(str(row.get(opinion_header, ""))) if opinion_header else "",
        "추정 사업기간": {
            "header": period_header,
            "label": period_label,
            "value": period_value,
            "value_source": source,
        },
    }


def _build_table_sections_payload(table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_table: dict[int, list[dict[str, Any]]] = {}
    for row in table_rows:
        if not isinstance(row, dict):
            continue
        table_index = int(row.get("table_index", 1))
        by_table.setdefault(table_index, []).append(row)

    sections_payload: list[dict[str, Any]] = []

    for table_index in sorted(by_table.keys()):
        rows = sorted(by_table[table_index], key=lambda record: int(record.get("row_index", 0)))
        headers: list[str] = []
        for row in rows:
            if "_section" in row:
                continue
            for header in _iter_data_headers(row):
                if header not in headers:
                    headers.append(header)
        if not headers:
            continue

        item_header = _pick_first_header(headers, _looks_like_item_header)
        opinion_header = _pick_first_header(headers, _looks_like_opinion_header)
        period_header = _pick_first_header(headers, _looks_like_period_header)

        current_section_title = "default"
        current_section_rows: list[dict[str, Any]] = []
        section_index = 1
        physical_rows = [row for row in rows if "_section" in row or any(_normalize_space(str(row.get(h, ""))) for h in headers)]

        i = 0
        while i < len(physical_rows):
            row = physical_rows[i]
            if "_section" in row:
                if current_section_rows:
                    sections_payload.append(
                        {
                            "table_index": table_index,
                            "section_index": section_index,
                            "section_title": current_section_title,
                            "rows": current_section_rows,
                        }
                    )
                    current_section_rows = []
                    section_index += 1
                section_text = _normalize_space(str(row.get("_section", "")))
                current_section_title = section_text or f"section_{section_index}"
                i += 1
                continue

            logical = _build_logical_row(
                row,
                item_header=item_header,
                opinion_header=opinion_header,
                period_header=period_header,
            )

            period_obj = logical.get("추정 사업기간", {})
            label = _normalize_space(str(period_obj.get("label", ""))) if isinstance(period_obj, dict) else ""
            value = _normalize_space(str(period_obj.get("value", ""))) if isinstance(period_obj, dict) else ""
            source = _normalize_space(str(period_obj.get("value_source", ""))) if isinstance(period_obj, dict) else "empty"

            has_item = bool(_normalize_space(str(logical.get("검토항목", ""))))
            has_opinion = bool(_normalize_space(str(logical.get("검토의견", ""))))

            # If current row has period label but no value, try to bind the immediate next value-only row.
            if period_header and label and not value and i + 1 < len(physical_rows):
                next_row = physical_rows[i + 1]
                if "_section" not in next_row:
                    next_logical = _build_logical_row(
                        next_row,
                        item_header=item_header,
                        opinion_header=opinion_header,
                        period_header=period_header,
                    )
                    next_period = next_logical.get("추정 사업기간", {})
                    next_value = (
                        _normalize_space(str(next_period.get("value", ""))) if isinstance(next_period, dict) else ""
                    )
                    next_source = (
                        _normalize_space(str(next_period.get("value_source", "")))
                        if isinstance(next_period, dict)
                        else "empty"
                    )
                    next_has_item = bool(_normalize_space(str(next_logical.get("검토항목", ""))))
                    next_has_opinion = bool(_normalize_space(str(next_logical.get("검토의견", ""))))

                    if next_value and not next_has_item and not next_has_opinion:
                        period_obj["value"] = next_value
                        period_obj["value_source"] = next_source
                        logical["추정 사업기간"] = period_obj
                        i += 1

            # If same row has label and inferred value from opinion, keep inferred value but allow
            # a later direct value-only row to override.
            if period_header and label and value and source != "direct_cell" and i + 1 < len(physical_rows):
                next_row = physical_rows[i + 1]
                if "_section" not in next_row:
                    next_logical = _build_logical_row(
                        next_row,
                        item_header=item_header,
                        opinion_header=opinion_header,
                        period_header=period_header,
                    )
                    next_period = next_logical.get("추정 사업기간", {})
                    next_value = (
                        _normalize_space(str(next_period.get("value", ""))) if isinstance(next_period, dict) else ""
                    )
                    next_source = (
                        _normalize_space(str(next_period.get("value_source", "")))
                        if isinstance(next_period, dict)
                        else "empty"
                    )
                    next_has_item = bool(_normalize_space(str(next_logical.get("검토항목", ""))))
                    next_has_opinion = bool(_normalize_space(str(next_logical.get("검토의견", ""))))
                    if next_value and not next_has_item and not next_has_opinion and _source_rank(next_source) > _source_rank(source):
                        period_obj["value"] = next_value
                        period_obj["value_source"] = next_source
                        logical["추정 사업기간"] = period_obj
                        i += 1

            # Drop meaningless value-only rows if their value has been consumed above.
            if has_item or has_opinion or label or value:
                current_section_rows.append(logical)
            i += 1

        if current_section_rows:
            sections_payload.append(
                {
                    "table_index": table_index,
                    "section_index": section_index,
                    "section_title": current_section_title,
                    "rows": current_section_rows,
                }
            )

    return sections_payload


def _build_table_rows_payload(
    *,
    pred_structured: dict[str, Any],
    tables: list[str],
    table_footnotes: list[str],
) -> dict[str, Any]:
    table_rows: list[dict[str, Any]] = []
    for table_index, table_html in enumerate(tables, start=1):
        table_rows.extend(_build_row_records_from_table(table_html, table_index=table_index))
    table_sections = _build_table_sections_payload(table_rows)

    return {
        "schema_version": "ocr_table_rows.v1",
        "id": pred_structured.get("id"),
        "type": pred_structured.get("type"),
        "table_rows": table_rows,
        "table_sections": table_sections,
        "table_footnotes": table_footnotes,
    }


def _build_parsed_rows_preview_sections(table_rows: list[dict[str, Any]]) -> str:
    by_table: dict[int, list[dict[str, Any]]] = {}
    for row in table_rows:
        if not isinstance(row, dict):
            continue
        table_index = int(row.get("table_index", 1))
        by_table.setdefault(table_index, []).append(row)

    sections: list[str] = []
    for table_index in sorted(by_table.keys()):
        rows = by_table[table_index]
        headers: list[str] = []
        for row in rows:
            if "_section" in row:
                continue
            for key in _iter_data_headers(row):
                if key not in headers:
                    headers.append(key)

        if not headers:
            continue

        header_html = "".join(f"<th>{html.escape(col)}</th>" for col in headers)
        body_rows: list[str] = []
        for row in rows:
            if "_section" in row:
                section = html.escape(_normalize_space(str(row.get("_section", ""))))
                if section:
                    body_rows.append(f'<tr class="section-row"><td colspan="{len(headers)}">{section}</td></tr>')
                continue

            cells: list[str] = []
            for col in headers:
                value = html.escape(_normalize_space(str(row.get(col, ""))))
                cells.append(f"<td>{value}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")

        if not body_rows:
            continue

        sections.append(
            f"""
<section class="table-block parsed-block">
  <h2>Parsed Rows (Table {table_index})</h2>
  <table class="parsed-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</section>
"""
        )
    return "".join(sections)


def build_table_preview_html(tables: list[str], table_footnotes: list[str], table_rows: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for idx, table_html in enumerate(tables, start=1):
        sections.append(f'<section class="table-block"><h2>Table {idx}</h2>{table_html}</section>')

    parsed_sections = _build_parsed_rows_preview_sections(table_rows)
    if parsed_sections:
        sections.append(parsed_sections)

    if table_footnotes:
        note_items = "".join(f"<li>{html.escape(note)}</li>" for note in table_footnotes)
        sections.append(
            """
<section class="table-block">
  <h2>Footnotes</h2>
  <ul class="footnote-list">
%s
  </ul>
</section>
"""
            % note_items
        )

    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OCR Table Preview</title>
  <style>
    body {{
      margin: 24px;
      font-family: "Noto Sans KR", "Malgun Gothic", sans-serif;
      color: #111827;
      line-height: 1.45;
      background: #f8fafc;
    }}
    .table-block {{
      margin: 0 0 24px 0;
      background: #ffffff;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }}
    .table-block h2 {{
      margin: 0 0 12px 0;
      font-size: 14px;
      font-weight: 700;
      color: #374151;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: auto;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #6b7280;
      padding: 6px 8px;
      vertical-align: top;
      word-break: break-word;
      white-space: pre-wrap;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 700;
    }}
    .footnote-list {{
      margin: 0;
      padding-left: 20px;
    }}
    .footnote-list li {{
      margin: 0 0 6px 0;
    }}
    .parsed-block {{
      border-color: #93c5fd;
      background: #eff6ff;
    }}
    .parsed-table th {{
      background: #dbeafe;
    }}
    .section-row td {{
      background: #e5e7eb;
      font-weight: 600;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def save_ocr_table_outputs(
    *,
    pred_raw_path: str,
    pred_structured_path: str,
    output_path: str,
) -> dict[str, Any]:
    pred_raw = json.loads(Path(pred_raw_path).read_text(encoding="utf-8"))
    pred_structured = json.loads(Path(pred_structured_path).read_text(encoding="utf-8"))

    candidates: list[str] = []
    pred_text = str(pred_structured.get("pred_text", ""))
    candidates.extend(_extract_table_html_from_text(pred_text))
    candidates.extend(_extract_table_html_from_obj(pred_raw.get("raw_pipeline_output")))
    candidates.extend(_extract_table_html_from_obj(pred_raw.get("ocr_lines")))
    unique_tables = _dedupe_table_html(candidates)

    table_output_path = Path(output_path)
    preview_output_path = table_output_path.with_name("pred_table_rows_human_review.html")
    table_rows_output_path = table_output_path.with_name(f"{table_output_path.stem}_rows.json")

    if not unique_tables:
        _remove_if_exists(table_output_path)
        _remove_if_exists(preview_output_path)
        _remove_if_exists(table_rows_output_path)
        return {
            "saved": False,
            "pred_table_rows_path": str(table_rows_output_path),
        }

    table_output_path.parent.mkdir(parents=True, exist_ok=True)
    table_output_path.write_text("\n\n".join(unique_tables), encoding="utf-8")

    table_footnotes = _extract_table_footnotes(pred_raw, pred_structured)
    table_rows_payload = _build_table_rows_payload(
        pred_structured=pred_structured,
        tables=unique_tables,
        table_footnotes=table_footnotes,
    )
    preview_output_path.write_text(
        build_table_preview_html(
            unique_tables,
            table_footnotes,
            table_rows_payload.get("table_rows", []),
        ),
        encoding="utf-8",
    )
    table_rows_output_path.write_text(json.dumps(table_rows_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "saved": True,
        "pred_table_rows_path": str(table_rows_output_path),
    }
