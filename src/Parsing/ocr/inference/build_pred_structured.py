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
_TAG_PATTERN = re.compile(r"<[^>]+>")
_KV_LINE_PATTERN = re.compile(r"^\s*([^\n\r:：]{1,40})\s*[:：]\s*(.+)\s*$")


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
    result["pred_structure"] = build_pred_structure_from_ocr(pred_text, pred_raw_item)
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
