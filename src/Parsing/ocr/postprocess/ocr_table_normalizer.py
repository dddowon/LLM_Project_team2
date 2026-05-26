from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

_TABLE_HTML_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HAS_CONTENT_PATTERN = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_ROW_PATTERN = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(r"<(th|td)\b([^>]*)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_COLSPAN_PATTERN = re.compile(r'\bcolspan\s*=\s*["\']?(\d+)["\']?', re.IGNORECASE)

_DEFAULT_RULES: dict[str, Any] = {
    "global": {
        "strip_outer_whitespace": True,
        "collapse_inter_tag_whitespace": True,
        "drop_empty_tables": False,
        "header_body_mismatch_ratio_warn": 0.2,
        "header_body_cell_count_mismatch_ratio_warn": 0.2,
        "possible_column_merge_delta": 1,
    },
    "per_type": {
        "table": {"drop_empty_tables": False},
        "diagram": {"drop_empty_tables": True},
        "scan": {"drop_empty_tables": False},
    },
}


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


def load_normalization_rules(rules_path: str | None) -> dict[str, Any]:
    if not rules_path:
        return dict(_DEFAULT_RULES)
    path = Path(rules_path)
    if not path.exists():
        return dict(_DEFAULT_RULES)
    raw = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        return dict(_DEFAULT_RULES)
    merged: dict[str, Any] = dict(_DEFAULT_RULES)
    if isinstance(payload.get("global"), dict):
        merged["global"] = {**_DEFAULT_RULES["global"], **payload["global"]}
    if isinstance(payload.get("per_type"), dict):
        base_per_type = dict(_DEFAULT_RULES["per_type"])
        for key, value in payload["per_type"].items():
            if isinstance(value, dict):
                base_per_type[str(key).lower()] = {**base_per_type.get(str(key).lower(), {}), **value}
        merged["per_type"] = base_per_type
    return merged


def _resolve_effective_rules(rules: dict[str, Any], doc_type: str) -> dict[str, Any]:
    global_rules = rules.get("global", {}) if isinstance(rules.get("global"), dict) else {}
    per_type = rules.get("per_type", {}) if isinstance(rules.get("per_type"), dict) else {}
    doc_type_rules = per_type.get(doc_type.lower(), {}) if isinstance(per_type.get(doc_type.lower()), dict) else {}
    return {**global_rules, **doc_type_rules}


def _has_meaningful_cell_content(table_html: str) -> bool:
    cells = _HAS_CONTENT_PATTERN.findall(table_html)
    if not cells:
        return False
    for raw_cell in cells:
        cell_text = _TAG_PATTERN.sub(" ", raw_cell or "")
        if re.sub(r"\s+", " ", cell_text).strip():
            return True
    return False


def normalize_table_html(table_html: str, *, effective_rules: dict[str, Any]) -> str | None:
    normalized = table_html
    if bool(effective_rules.get("strip_outer_whitespace", True)):
        normalized = normalized.strip()
    if bool(effective_rules.get("collapse_inter_tag_whitespace", True)):
        normalized = re.sub(r">\s+<", "><", normalized)

    if bool(effective_rules.get("drop_empty_tables", False)) and not _has_meaningful_cell_content(normalized):
        return None

    return normalized


def normalize_tables(
    tables: list[str],
    *,
    doc_type: str,
    rules: dict[str, Any],
) -> list[str]:
    effective_rules = _resolve_effective_rules(rules, doc_type=doc_type)
    normalized: list[str] = []
    for table_html in tables:
        item = normalize_table_html(table_html, effective_rules=effective_rules)
        if not item:
            continue
        normalized.append(item)
    return _dedupe_table_html(normalized)


def _parse_colspan(cell_attrs: str) -> int:
    match = _COLSPAN_PATTERN.search(cell_attrs or "")
    if not match:
        return 1
    try:
        value = int(match.group(1))
    except ValueError:
        return 1
    return value if value > 0 else 1


def _row_shape(row_html: str) -> tuple[int, int, bool]:
    cells = _CELL_PATTERN.findall(row_html)
    if not cells:
        return 0, 0, False
    col_count = 0
    cell_count = len(cells)
    has_header_cell = False
    for tag, attrs, _ in cells:
        col_count += _parse_colspan(attrs)
        if str(tag).lower() == "th":
            has_header_cell = True
    return col_count, cell_count, has_header_cell


def analyze_table_structure(
    tables: list[str],
    *,
    rules: dict[str, Any],
    doc_type: str,
) -> dict[str, Any]:
    effective_rules = _resolve_effective_rules(rules, doc_type=doc_type)
    warn_ratio = float(effective_rules.get("header_body_mismatch_ratio_warn", 0.2))
    cell_warn_ratio = float(effective_rules.get("header_body_cell_count_mismatch_ratio_warn", 0.2))
    merge_delta = int(effective_rules.get("possible_column_merge_delta", 1))

    per_table: list[dict[str, Any]] = []
    warning_items: list[dict[str, Any]] = []
    warning_codes: list[str] = []

    for table_index, table_html in enumerate(tables, start=1):
        row_html_list = _ROW_PATTERN.findall(table_html)
        row_stats: list[dict[str, Any]] = []
        for row_index, row_html in enumerate(row_html_list, start=1):
            col_count, cell_count, is_header_row = _row_shape(row_html)
            row_stats.append(
                {
                    "row_index": row_index,
                    "col_count": col_count,
                    "cell_count": cell_count,
                    "row_type": "header" if is_header_row else "body",
                }
            )

        header_rows = [item for item in row_stats if item["row_type"] == "header" and int(item["col_count"]) > 0]
        if header_rows:
            expected_cols = max(int(item["col_count"]) for item in header_rows)
        else:
            positive_rows = [item for item in row_stats if int(item["col_count"]) > 0]
            expected_cols = max((int(item["col_count"]) for item in positive_rows), default=0)

        if header_rows:
            expected_header_cells = max(int(item["cell_count"]) for item in header_rows)
        else:
            rows_with_expected_cols = [item for item in row_stats if int(item["col_count"]) == expected_cols]
            expected_header_cells = max((int(item["cell_count"]) for item in rows_with_expected_cols), default=0)

        if expected_header_cells <= 0:
            positive_rows = [item for item in row_stats if int(item["cell_count"]) > 0]
            expected_header_cells = max((int(item["cell_count"]) for item in positive_rows), default=0)

        header_anchor_index = 1
        for item in row_stats:
            if int(item["col_count"]) == expected_cols and int(item["cell_count"]) == expected_header_cells:
                header_anchor_index = int(item["row_index"])
                break

        body_rows_all = [
            item
            for item in row_stats
            if int(item["row_index"]) > header_anchor_index and int(item["col_count"]) > 0
        ]
        body_rows = [
            item
            for item in body_rows_all
            if not (int(item["cell_count"]) == 1 and int(item["col_count"]) == expected_cols)
        ]

        mismatch_rows = [item for item in body_rows if int(item["col_count"]) != expected_cols]
        mismatch_ratio = (len(mismatch_rows) / len(body_rows)) if body_rows else 0.0
        cell_mismatch_rows = [
            item
            for item in body_rows
            if int(item["col_count"]) == expected_cols and int(item["cell_count"]) < expected_header_cells
        ]
        cell_mismatch_ratio = (len(cell_mismatch_rows) / len(body_rows)) if body_rows else 0.0

        table_warning_codes: list[str] = []
        if expected_cols > 0 and body_rows and mismatch_rows and mismatch_ratio >= warn_ratio:
            table_warning_codes.append("header_body_col_mismatch")
        if expected_header_cells > 0 and body_rows and cell_mismatch_rows and cell_mismatch_ratio >= cell_warn_ratio:
            table_warning_codes.append("header_body_cell_count_mismatch")

        merge_rows = [
            item for item in mismatch_rows if expected_cols > 0 and int(item["col_count"]) <= (expected_cols - merge_delta)
        ]
        merge_rows_by_cell = [
            item
            for item in cell_mismatch_rows
            if expected_header_cells > 0 and int(item["cell_count"]) <= (expected_header_cells - merge_delta)
        ]
        if merge_rows or merge_rows_by_cell:
            table_warning_codes.append("possible_column_merge")

        if table_warning_codes:
            for code in table_warning_codes:
                warning_items.append(
                    {
                        "table_index": table_index,
                        "code": code,
                        "expected_cols": expected_cols,
                        "expected_header_cells": expected_header_cells,
                        "body_rows": len(body_rows),
                        "mismatch_rows": [int(item["row_index"]) for item in mismatch_rows],
                        "mismatch_ratio": round(mismatch_ratio, 4),
                        "cell_mismatch_rows": [int(item["row_index"]) for item in cell_mismatch_rows],
                        "cell_mismatch_ratio": round(cell_mismatch_ratio, 4),
                    }
                )
            warning_codes.extend(table_warning_codes)

        per_table.append(
            {
                "table_index": table_index,
                "expected_cols": expected_cols,
                "expected_header_cells": expected_header_cells,
                "header_anchor_index": header_anchor_index,
                "body_rows": len(body_rows),
                "body_rows_all": len(body_rows_all),
                "mismatch_rows": [int(item["row_index"]) for item in mismatch_rows],
                "mismatch_ratio": round(mismatch_ratio, 4),
                "cell_mismatch_rows": [int(item["row_index"]) for item in cell_mismatch_rows],
                "cell_mismatch_ratio": round(cell_mismatch_ratio, 4),
                "warning_codes": table_warning_codes,
                "row_stats": row_stats,
            }
        )

    unique_codes = sorted(set(warning_codes))
    return {
        "warning_codes": unique_codes,
        "warning_items": warning_items,
        "per_table": per_table,
        "table_count": len(tables),
    }


def build_table_preview_html(tables: list[str]) -> str:
    sections: list[str] = []
    for idx, table_html in enumerate(tables, start=1):
        sections.append(
            f'<section class="table-block"><h2>Table {idx}</h2>{table_html}</section>'
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
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def save_ocr_table_outputs(
    *,
    pred_raw_path: str,
    pred_structured_path: str,
    output_path: str,
    table_normalization_enabled: bool,
    table_normalization_rules_path: str | None,
) -> dict[str, Any]:
    pred_raw = json.loads(Path(pred_raw_path).read_text(encoding="utf-8"))
    pred_structured = json.loads(Path(pred_structured_path).read_text(encoding="utf-8"))
    doc_type = str(pred_structured.get("type", "table")).lower()

    candidates: list[str] = []
    pred_text = str(pred_structured.get("pred_text", ""))
    candidates.extend(_extract_table_html_from_text(pred_text))
    candidates.extend(_extract_table_html_from_obj(pred_raw.get("raw_pipeline_output")))
    candidates.extend(_extract_table_html_from_obj(pred_raw.get("ocr_lines")))
    unique_tables = _dedupe_table_html(candidates)

    table_output_path = Path(output_path)
    preview_output_path = table_output_path.with_name(f"{table_output_path.stem}_preview{table_output_path.suffix}")
    normalized_output_path = table_output_path.with_name(f"{table_output_path.stem}_normalized{table_output_path.suffix}")
    diagnostics_output_path = table_output_path.with_name(
        f"{table_output_path.stem}_structure_diagnostics.json"
    )
    if not unique_tables:
        _remove_if_exists(table_output_path)
        _remove_if_exists(preview_output_path)
        _remove_if_exists(normalized_output_path)
        _remove_if_exists(diagnostics_output_path)
        return {
            "saved": False,
            "table_structure_warning": [],
            "table_structure_diagnostics_path": str(diagnostics_output_path),
        }

    table_output_path.parent.mkdir(parents=True, exist_ok=True)
    table_output_path.write_text("\n\n".join(unique_tables), encoding="utf-8")

    rules = load_normalization_rules(table_normalization_rules_path)
    diagnostics = analyze_table_structure(
        unique_tables,
        rules=rules,
        doc_type=doc_type,
    )
    diagnostics_output_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_source = unique_tables
    if table_normalization_enabled:
        normalized_tables = normalize_tables(
            unique_tables,
            doc_type=doc_type,
            rules=rules,
        )
        if normalized_tables:
            normalized_output_path.write_text("\n\n".join(normalized_tables), encoding="utf-8")
            preview_source = normalized_tables
        else:
            _remove_if_exists(normalized_output_path)
    else:
        _remove_if_exists(normalized_output_path)

    preview_output_path.write_text(build_table_preview_html(preview_source), encoding="utf-8")
    return {
        "saved": True,
        "table_structure_warning": diagnostics.get("warning_codes", []),
        "table_structure_diagnostics_path": str(diagnostics_output_path),
    }
