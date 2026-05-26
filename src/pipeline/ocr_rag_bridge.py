from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.utils.jsonl import write_jsonl

_WHITESPACE_RE = re.compile(r"\s+")
_SIGNATURE_TOKEN_RE = re.compile(r"^(?:위\s*원(?:장)?|\(?서명\)?|날인)$")


def _normalize_space(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        output = [_normalize_space(item) for item in value if _normalize_space(item)]
        return output
    normalized = _normalize_space(value)
    return [normalized] if normalized else []


def _format_pred_structure_text(pred_structure: dict[str, Any]) -> str:
    lines: list[str] = []
    for field, raw_values in pred_structure.items():
        values = _to_str_list(raw_values)
        if not values:
            continue
        lines.append(f"- {field}: {' | '.join(values)}")
    return "\n".join(lines).strip()


def _period_value_cell(row: dict[str, Any]) -> tuple[str, str, str]:
    period = row.get("추정 사업기간", "")
    if isinstance(period, dict):
        value = _normalize_space(period.get("value", ""))
        source = _normalize_space(period.get("value_source", "empty")) or "empty"
        label = _normalize_space(period.get("label", ""))
        return value, source, label
    return _normalize_space(period), "unknown", ""


def _is_signature_row(item: str, opinion: str, period_value: str) -> bool:
    if period_value:
        return False
    item_norm = _normalize_space(item)
    opinion_norm = _normalize_space(opinion)
    if not item_norm and not opinion_norm:
        return False
    item_ok = (not item_norm) or bool(_SIGNATURE_TOKEN_RE.match(item_norm))
    opinion_ok = (not opinion_norm) or bool(_SIGNATURE_TOKEN_RE.match(opinion_norm))
    return item_ok and opinion_ok


def _format_table_section_text(section: dict[str, Any]) -> str:
    title = _normalize_space(section.get("section_title", ""))
    rows = section.get("rows", [])
    if not isinstance(rows, list):
        rows = []

    lines: list[str] = []
    row_line_count = 0
    if title:
        lines.append(f"[섹션] {title}")
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = _normalize_space(row.get("검토항목", ""))
        opinion = _normalize_space(row.get("검토의견", ""))
        period_value, period_source, period_label = _period_value_cell(row)
        if _is_signature_row(item, opinion, period_value):
            continue
        row_index = _normalize_space(row.get("row_index", ""))
        row_head = f"행 {row_index}" if row_index else "행"
        parts = [
            f"{row_head}",
            f"검토항목={item}" if item else "검토항목=",
            f"검토의견={opinion}" if opinion else "검토의견=",
            f"추정 사업기간={period_value}" if period_value else "추정 사업기간=",
            f"period_source={period_source}",
        ]
        if period_label:
            parts.append(f"period_label={period_label}")
        lines.append(" | ".join(parts))
        row_line_count += 1
    if row_line_count == 0:
        return ""
    return "\n".join(lines).strip()


def _format_table_rows_fallback_text(table_rows_payload: dict[str, Any], max_rows: int = 20) -> str:
    rows = table_rows_payload.get("table_rows", [])
    if not isinstance(rows, list):
        return ""
    lines: list[str] = []
    count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "_section" in row:
            section = _normalize_space(row.get("_section", ""))
            if section:
                lines.append(f"[섹션] {section}")
            continue
        count += 1
        if count > max_rows:
            break
        item = _normalize_space(row.get("검토항목", ""))
        opinion = _normalize_space(row.get("검토의견", ""))
        period = _normalize_space(row.get("추정 사업기간", ""))
        if _is_signature_row(item, opinion, period):
            continue
        lines.append(f"행 {row.get('row_index', '')} | 검토항목={item} | 검토의견={opinion} | 추정 사업기간={period}")
    return "\n".join(lines).strip()


def _format_table_footnotes(table_rows_payload: dict[str, Any]) -> str:
    notes = table_rows_payload.get("table_footnotes", [])
    if not isinstance(notes, list):
        return ""
    lines = [_normalize_space(note) for note in notes if _normalize_space(note)]
    if not lines:
        return ""
    return "\n".join(f"- {note}" for note in lines)


def _build_chunk_row(
    *,
    chunk_id: str,
    doc_id: str,
    chunk_type: str,
    chunk_text: str,
    metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_type": chunk_type,
        "chunk_text": chunk_text,
        "metadata": metadata,
    }


def _iter_engine_dirs(ocr_eval_root: Path, engine: str | None) -> list[Path]:
    if engine:
        target = ocr_eval_root / engine
        return [target] if target.exists() else []
    return sorted([path for path in ocr_eval_root.iterdir() if path.is_dir()], key=lambda p: p.name)


def _iter_image_dirs(engine_dir: Path, doc_key: str | None) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    doc_dirs = sorted([path for path in engine_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    if doc_key:
        doc_dirs = [path for path in doc_dirs if path.name == doc_key]
    for one_doc in doc_dirs:
        image_dirs = sorted([path for path in one_doc.iterdir() if path.is_dir()], key=lambda p: p.name)
        for image_dir in image_dirs:
            if (image_dir / "pred_structured.json").exists():
                output.append((one_doc.name, image_dir))
    return output


def export_ocr_eval_to_rag_inputs(
    *,
    ocr_eval_root: Path,
    output_manifest: Path,
    output_chunks: Path,
    engine: str | None = None,
    doc_key: str | None = None,
    include_review_required: bool = True,
    include_html_chunk: bool = False,
    html_chunk_max_chars: int = 1200,
) -> tuple[int, int]:
    manifest_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    engine_dirs = _iter_engine_dirs(ocr_eval_root, engine)

    global_chunk_index = 0
    for engine_dir in engine_dirs:
        engine_name = engine_dir.name
        for one_doc_key, image_dir in _iter_image_dirs(engine_dir, doc_key):
            pred_structured_path = image_dir / "pred_structured.json"
            pred_table_rows_path = image_dir / "pred_table_rows.json"
            pred_table_html_path = image_dir / "pred_table_raw.html"
            pred_table_preview_path = image_dir / "pred_table_rows_human_review.html"
            eval_summary_path = image_dir / "eval_summary.json"

            pred_structured = _load_json(pred_structured_path)
            if not pred_structured:
                continue
            table_rows_payload = _load_json(pred_table_rows_path)
            eval_summary = _load_json(eval_summary_path)

            item_id = _normalize_space(pred_structured.get("id")) or f"{one_doc_key}_{image_dir.name}"
            image_stem = image_dir.name
            image_type = _normalize_space(pred_structured.get("type", "unknown"))
            review_required = bool(eval_summary.get("review_required", False))
            review_reasons = eval_summary.get("review_reasons", [])
            if not isinstance(review_reasons, list):
                review_reasons = []

            if review_required and not include_review_required:
                continue

            manifest_row: dict[str, Any] = {
                "id": item_id,
                "doc_key": one_doc_key,
                "image_stem": image_stem,
                "type": image_type,
                "ocr_engine": engine_name,
                "review_required": review_required,
                "review_reasons": review_reasons,
                "paths": {
                    "pred_structured_path": str(pred_structured_path),
                    "pred_table_rows_path": str(pred_table_rows_path),
                    "pred_table_html_path": str(pred_table_html_path),
                    "pred_table_preview_path": str(pred_table_preview_path),
                    "eval_summary_path": str(eval_summary_path),
                },
                "table_sections_count": len(table_rows_payload.get("table_sections", []))
                if isinstance(table_rows_payload.get("table_sections"), list)
                else 0,
                "table_rows_count": len(table_rows_payload.get("table_rows", []))
                if isinstance(table_rows_payload.get("table_rows"), list)
                else 0,
            }
            manifest_rows.append(manifest_row)

            common_metadata = {
                "file_name": one_doc_key,
                "doc_key": one_doc_key,
                "image_stem": image_stem,
                "ocr_item_id": item_id,
                "ocr_engine": engine_name,
                "ocr_type": image_type,
                "pred_structured_path": str(pred_structured_path),
                "pred_table_rows_path": str(pred_table_rows_path),
                "pred_table_html_path": str(pred_table_html_path),
                "eval_summary_path": str(eval_summary_path),
                "review_required": str(review_required),
                "review_reasons": "|".join(str(reason) for reason in review_reasons),
                "source": "ocr",
            }

            pred_structure = pred_structured.get("pred_structure", {})
            if isinstance(pred_structure, dict):
                structure_text = _format_pred_structure_text(pred_structure)
                if structure_text:
                    global_chunk_index += 1
                    seed = f"{item_id}|pred_structure|{engine_name}|{structure_text[:120]}"
                    chunk_rows.append(
                        _build_chunk_row(
                            chunk_id=f"ocr_chunk_{global_chunk_index:08d}_{_stable_hash(seed)}",
                            doc_id=one_doc_key,
                            chunk_type="ocr_structured",
                            chunk_text=(
                                f"문서키: {one_doc_key}\n이미지: {image_stem}\nOCR ID: {item_id}\n"
                                f"자료유형: {image_type}\n\n[Pred Structure]\n{structure_text}"
                            ),
                            metadata={**common_metadata, "chunk_scope": "pred_structure"},
                        )
                    )

            table_sections = table_rows_payload.get("table_sections", [])
            if isinstance(table_sections, list) and table_sections:
                for section_idx, section in enumerate(table_sections, start=1):
                    if not isinstance(section, dict):
                        continue
                    section_text = _format_table_section_text(section)
                    if not section_text:
                        continue
                    section_title = _normalize_space(section.get("section_title", "")) or f"section_{section_idx}"
                    global_chunk_index += 1
                    seed = f"{item_id}|table_section|{section_idx}|{section_title}|{engine_name}"
                    chunk_rows.append(
                        _build_chunk_row(
                            chunk_id=f"ocr_chunk_{global_chunk_index:08d}_{_stable_hash(seed)}",
                            doc_id=one_doc_key,
                            chunk_type="ocr_table_section",
                            chunk_text=(
                                f"문서키: {one_doc_key}\n이미지: {image_stem}\nOCR ID: {item_id}\n"
                                f"섹션제목: {section_title}\n\n{section_text}"
                            ),
                            metadata={
                                **common_metadata,
                                "chunk_scope": "table_section",
                                "table_section_title": section_title,
                                "table_section_index": str(section_idx),
                            },
                        )
                    )
            else:
                fallback_text = _format_table_rows_fallback_text(table_rows_payload)
                if fallback_text:
                    global_chunk_index += 1
                    seed = f"{item_id}|table_rows_fallback|{engine_name}"
                    chunk_rows.append(
                        _build_chunk_row(
                            chunk_id=f"ocr_chunk_{global_chunk_index:08d}_{_stable_hash(seed)}",
                            doc_id=one_doc_key,
                            chunk_type="ocr_table_rows",
                            chunk_text=(
                                f"문서키: {one_doc_key}\n이미지: {image_stem}\nOCR ID: {item_id}\n\n"
                                f"[Parsed Table Rows]\n{fallback_text}"
                            ),
                            metadata={**common_metadata, "chunk_scope": "table_rows_fallback"},
                        )
                    )

            footnote_text = _format_table_footnotes(table_rows_payload)
            if footnote_text:
                global_chunk_index += 1
                seed = f"{item_id}|table_footnotes|{engine_name}"
                chunk_rows.append(
                    _build_chunk_row(
                        chunk_id=f"ocr_chunk_{global_chunk_index:08d}_{_stable_hash(seed)}",
                        doc_id=one_doc_key,
                        chunk_type="ocr_table_footnotes",
                        chunk_text=(
                            f"문서키: {one_doc_key}\n이미지: {image_stem}\nOCR ID: {item_id}\n\n"
                            f"[Table Footnotes]\n{footnote_text}"
                        ),
                        metadata={**common_metadata, "chunk_scope": "table_footnotes"},
                    )
                )

            if include_html_chunk and pred_table_html_path.exists():
                html_text = pred_table_html_path.read_text(encoding="utf-8")
                html_text = _normalize_space(html_text)
                if html_text:
                    html_snippet = html_text[:html_chunk_max_chars]
                    global_chunk_index += 1
                    seed = f"{item_id}|table_html|{engine_name}|{html_snippet[:120]}"
                    chunk_rows.append(
                        _build_chunk_row(
                            chunk_id=f"ocr_chunk_{global_chunk_index:08d}_{_stable_hash(seed)}",
                            doc_id=one_doc_key,
                            chunk_type="ocr_table_html",
                            chunk_text=(
                                f"문서키: {one_doc_key}\n이미지: {image_stem}\nOCR ID: {item_id}\n"
                                f"[Table HTML Snippet]\n{html_snippet}"
                            ),
                            metadata={**common_metadata, "chunk_scope": "table_html_snippet"},
                        )
                    )

    # Attach per-image chunk counts to manifest rows.
    chunk_count_map: dict[str, int] = {}
    for row in chunk_rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        item_id = _normalize_space(metadata.get("ocr_item_id", ""))
        if not item_id:
            continue
        chunk_count_map[item_id] = chunk_count_map.get(item_id, 0) + 1
    for row in manifest_rows:
        item_id = _normalize_space(row.get("id", ""))
        row["chunk_count"] = chunk_count_map.get(item_id, 0)

    write_jsonl(output_manifest, manifest_rows)
    write_jsonl(output_chunks, chunk_rows)
    return len(manifest_rows), len(chunk_rows)
