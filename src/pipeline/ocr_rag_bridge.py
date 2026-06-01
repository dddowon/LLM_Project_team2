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


def _is_ocr_image_dir(image_dir: Path, *, allow_inference_only: bool) -> bool:
    if (image_dir / "eval" / "gt_pred_structured.json").exists():
        return True
    if allow_inference_only and (image_dir / "inference" / "pred_raw.json").exists():
        return True
    return False


def _is_doc_dir(doc_dir: Path, *, allow_inference_only: bool) -> bool:
    image_dirs = [path for path in doc_dir.iterdir() if path.is_dir()]
    return any(_is_ocr_image_dir(path, allow_inference_only=allow_inference_only) for path in image_dirs)


def _iter_image_dirs(
    engine_dir: Path,
    doc_key: str | None,
    *,
    allow_inference_only: bool,
    images_tag: str | None = None,
) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []

    def collect_from_doc_dir(one_doc: Path) -> None:
        if doc_key and one_doc.name != doc_key:
            return
        image_dirs = sorted([path for path in one_doc.iterdir() if path.is_dir()], key=lambda p: p.name)
        for image_dir in image_dirs:
            if _is_ocr_image_dir(image_dir, allow_inference_only=allow_inference_only):
                output.append((one_doc.name, image_dir))

    first_level_dirs = sorted([path for path in engine_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    for first_level in first_level_dirs:
        if _is_doc_dir(first_level, allow_inference_only=allow_inference_only):
            # Legacy layout: <engine>/<doc_key>/<image_stem>/...
            # When images_tag is explicitly requested, skip legacy layout to avoid mixing tags.
            if images_tag:
                continue
            collect_from_doc_dir(first_level)
            continue

        # Versioned layout: <engine>/<images_tag>/<doc_key>/<image_stem>/...
        if images_tag and first_level.name != images_tag:
            continue
        second_level_dirs = sorted([path for path in first_level.iterdir() if path.is_dir()], key=lambda p: p.name)
        for second_level in second_level_dirs:
            if _is_doc_dir(second_level, allow_inference_only=allow_inference_only):
                collect_from_doc_dir(second_level)
    return output


def _format_pred_raw_text(pred_raw: dict[str, Any], *, max_lines: int = 160) -> str:
    """Inference-only fallback: derive readable text from pred_raw.json."""
    lines: list[str] = []

    ocr_lines = pred_raw.get("ocr_lines")
    if isinstance(ocr_lines, list):
        for item in ocr_lines:
            if len(lines) >= max_lines:
                break
            if isinstance(item, dict):
                text = _normalize_space(item.get("text", ""))
            else:
                text = _normalize_space(item)
            if text:
                lines.append(text)

    if not lines:
        for key in ("text", "texts", "markdown", "content", "result", "output", "prediction"):
            value = pred_raw.get(key)
            if isinstance(value, list):
                for item in value[:max_lines]:
                    text = _normalize_space(item)
                    if text:
                        lines.append(text)
            else:
                text = _normalize_space(value)
                if text:
                    lines.append(text)
            if lines:
                break

    return "\n".join(lines).strip()


def _resolve_curated_table_rows_path(
    *,
    curated_root: Path,
    engine_name: str,
    doc_key: str,
    image_stem: str,
    curated_file_name: str,
    images_tag: str | None,
    curated_version: str | None,
) -> Path:
    candidates: list[Path] = []
    normalized_images_tag = _normalize_space(images_tag)
    normalized_curated_version = _normalize_space(curated_version)

    if normalized_curated_version:
        candidates.extend(
            [
                curated_root / engine_name / normalized_curated_version / doc_key / image_stem / curated_file_name,
                curated_root / normalized_curated_version / engine_name / doc_key / image_stem / curated_file_name,
                curated_root / normalized_curated_version / doc_key / image_stem / curated_file_name,
            ]
        )
    if normalized_images_tag:
        candidates.extend(
            [
                curated_root / engine_name / normalized_images_tag / doc_key / image_stem / curated_file_name,
                curated_root / normalized_images_tag / engine_name / doc_key / image_stem / curated_file_name,
                curated_root / normalized_images_tag / doc_key / image_stem / curated_file_name,
            ]
        )

    candidates.extend(
        [
            curated_root / engine_name / doc_key / image_stem / curated_file_name,
            curated_root / doc_key / image_stem / curated_file_name,
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


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
    allow_inference_only: bool = False,
    images_tag: str | None = None,
    curated_root: Path | None = None,
    curated_file_name: str = "pred_table_layout.curated.json",
    input_version: str | None = None,
    ocr_engine_version: str | None = None,
    ocr_output_version: str | None = None,
    ocr_curated_version: str | None = None,
    rag_index_version: str | None = None,
) -> tuple[int, int]:
    manifest_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    engine_dirs = _iter_engine_dirs(ocr_eval_root, engine)

    global_chunk_index = 0
    for engine_dir in engine_dirs:
        engine_name = engine_dir.name
        for one_doc_key, image_dir in _iter_image_dirs(
            engine_dir,
            doc_key,
            allow_inference_only=allow_inference_only,
            images_tag=images_tag,
        ):
            pred_structured_path = image_dir / "eval" / "gt_pred_structured.json"
            pred_table_rows_path = image_dir / "inference" / "pred_table_layout.json"
            pred_table_html_path = image_dir / "inference" / "pred_table_raw.html"
            pred_table_preview_path = image_dir / "inference" / "pred_table_layout.html"
            eval_summary_path = image_dir / "eval" / "gt_eval_summary.json"
            pred_raw_path = image_dir / "inference" / "pred_raw.json"
            curated_table_rows_path = None
            if curated_root:
                curated_table_rows_path = _resolve_curated_table_rows_path(
                    curated_root=curated_root,
                    engine_name=engine_name,
                    doc_key=one_doc_key,
                    image_stem=image_dir.name,
                    curated_file_name=curated_file_name,
                    images_tag=images_tag,
                    curated_version=ocr_curated_version,
                )

            pred_structured = _load_json(pred_structured_path)
            is_inference_only = not bool(pred_structured)
            if is_inference_only and not allow_inference_only:
                continue

            pred_raw = _load_json(pred_raw_path) if is_inference_only else {}
            table_source = "raw"
            table_rows_source_path = pred_table_rows_path
            if curated_table_rows_path and curated_table_rows_path.exists():
                table_source = "curated"
                table_rows_source_path = curated_table_rows_path
            table_rows_payload = _load_json(table_rows_source_path)
            eval_summary = _load_json(eval_summary_path)

            item_id = (
                _normalize_space((pred_structured or {}).get("id"))
                or _normalize_space(pred_raw.get("id"))
                or f"{one_doc_key}_{image_dir.name}"
            )
            image_stem = image_dir.name
            image_type = _normalize_space((pred_structured or {}).get("type", "")) or _normalize_space(
                pred_raw.get("type", "unknown")
            )
            review_required = False if is_inference_only else bool(eval_summary.get("review_required", False))
            review_reasons = [] if is_inference_only else eval_summary.get("review_reasons", [])
            if not isinstance(review_reasons, list):
                review_reasons = []

            if review_required and not include_review_required:
                continue

            resolved_input_version = _normalize_space(input_version)
            resolved_ocr_engine_version = _normalize_space(ocr_engine_version) or engine_name
            resolved_ocr_output_version = _normalize_space(ocr_output_version) or _normalize_space(images_tag)
            resolved_ocr_curated_version = _normalize_space(ocr_curated_version)
            resolved_rag_index_version = _normalize_space(rag_index_version)

            manifest_row: dict[str, Any] = {
                "id": item_id,
                "doc_key": one_doc_key,
                "image_stem": image_stem,
                "type": image_type,
                "ocr_engine": engine_name,
                "input_version": resolved_input_version,
                "ocr_engine_version": resolved_ocr_engine_version,
                "ocr_output_version": resolved_ocr_output_version,
                "ocr_curated_version": resolved_ocr_curated_version,
                "rag_index_version": resolved_rag_index_version,
                "review_required": review_required,
                "review_reasons": review_reasons,
                "inference_only": is_inference_only,
                "paths": {
                    "pred_structured_path": str(pred_structured_path),
                    "pred_raw_path": str(pred_raw_path),
                    "pred_table_rows_path": str(pred_table_rows_path),
                    "curated_table_rows_path": str(curated_table_rows_path) if curated_table_rows_path else "",
                    "table_rows_source_path": str(table_rows_source_path),
                    "pred_table_html_path": str(pred_table_html_path),
                    "pred_table_preview_path": str(pred_table_preview_path),
                    "eval_summary_path": str(eval_summary_path),
                },
                "table_source": table_source,
                "fallback_used": table_source != "curated",
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
                "input_version": resolved_input_version,
                "ocr_engine_version": resolved_ocr_engine_version,
                "ocr_output_version": resolved_ocr_output_version,
                "ocr_curated_version": resolved_ocr_curated_version,
                "rag_index_version": resolved_rag_index_version,
                "pred_structured_path": str(pred_structured_path),
                "pred_raw_path": str(pred_raw_path),
                "pred_table_rows_path": str(pred_table_rows_path),
                "curated_table_rows_path": str(curated_table_rows_path) if curated_table_rows_path else "",
                "table_rows_source_path": str(table_rows_source_path),
                "pred_table_html_path": str(pred_table_html_path),
                "eval_summary_path": str(eval_summary_path),
                "review_required": str(review_required),
                "review_reasons": "|".join(str(reason) for reason in review_reasons),
                "inference_only": str(is_inference_only),
                "source": "ocr",
                "table_source": table_source,
                "source_priority": "curated>raw",
                "fallback_used": str(table_source != "curated"),
            }

            if pred_structured:
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
            elif pred_raw:
                raw_text = _format_pred_raw_text(pred_raw)
                if raw_text:
                    global_chunk_index += 1
                    seed = f"{item_id}|pred_raw|{engine_name}|{raw_text[:120]}"
                    chunk_rows.append(
                        _build_chunk_row(
                            chunk_id=f"ocr_chunk_{global_chunk_index:08d}_{_stable_hash(seed)}",
                            doc_id=one_doc_key,
                            chunk_type="ocr_raw_text",
                            chunk_text=(
                                f"문서키: {one_doc_key}\n이미지: {image_stem}\nOCR ID: {item_id}\n"
                                f"자료유형: {image_type}\n\n[OCR Raw Text]\n{raw_text}"
                            ),
                            metadata={**common_metadata, "chunk_scope": "pred_raw"},
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
