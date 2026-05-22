from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
import re
from pathlib import Path


def normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def is_english_like(text: str) -> bool:
    if not re.search(r"[A-Za-z]", text):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9\s.,&'()/_-]+", text))


def extract_hangul_segments(text: str) -> list[str]:
    segments = re.findall(r"[가-힣][가-힣\s]*[가-힣]|[가-힣]", text)
    return [normalize_space(seg) for seg in segments if normalize_space(seg)]


def extract_english_segments(text: str) -> list[str]:
    segments = re.findall(r"[A-Za-z][A-Za-z0-9\s.,&'()/_-]*[A-Za-z0-9)]", text)
    cleaned = [normalize_space(seg) for seg in segments if normalize_space(seg)]
    return [seg for seg in cleaned if is_english_like(seg)]


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


def is_value_matched(expected: object, pred_text: str, threshold: float = 0.65) -> tuple[bool, float]:
    expected_norm = compact_text(expected)
    pred_norm = compact_text(pred_text)
    if not expected_norm:
        return False, 0.0
    if expected_norm in pred_norm:
        return True, 1.0
    score = SequenceMatcher(None, expected_norm, pred_norm).ratio()
    return score >= threshold, score


def build_pred_structure_from_gt_schema(gt_obj: object, pred_text: str, threshold: float = 0.65) -> object:
    if isinstance(gt_obj, dict):
        return {
            str(key): build_pred_structure_from_gt_schema(value, pred_text, threshold=threshold)
            for key, value in gt_obj.items()
        }
    if isinstance(gt_obj, list):
        matched_values: list[str] = []
        for value in gt_obj:
            matched, _ = is_value_matched(value, pred_text, threshold=threshold)
            if matched:
                matched_values.append(str(value))
        return matched_values
    value = str(gt_obj).strip()
    if not value:
        return ""
    matched, _ = is_value_matched(value, pred_text, threshold=threshold)
    return value if matched else ""


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
    result["pred_structure"] = build_pred_structure_from_gt_schema(
        gt_item.get("gt_structure", {}),
        pred_text,
        threshold=0.65,
    )
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
