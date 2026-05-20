from __future__ import annotations

import argparse
import json
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


def load_item_by_id(path: Path, target_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    # [Design Intent] VLM often returns mixed Korean/English in one line.
    # Split mixed lines into language-specific segments before mapping to structured fields.
    kor_candidates: list[str] = []
    eng_candidates: list[str] = []
    for text in kept:
        kor_candidates.extend(extract_hangul_segments(text))
        eng_candidates.extend(extract_english_segments(text))
    kor_candidates = dedupe_keep_order(kor_candidates)
    eng_candidates = dedupe_keep_order(eng_candidates)

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
    result["pred_structure"] = {
        "기관명": kor_candidates[:1],
        "영문명": eng_candidates[:1],
        "로고": kept,
    }
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
    parser.add_argument("--gt", required=True, help="GT JSON path")
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
