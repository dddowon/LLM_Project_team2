from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def levenshtein_tokens(a_tokens: list[str], b_tokens: list[str]) -> int:
    if a_tokens == b_tokens:
        return 0
    if not a_tokens:
        return len(b_tokens)
    if not b_tokens:
        return len(a_tokens)
    prev = list(range(len(b_tokens) + 1))
    for i, a_token in enumerate(a_tokens, 1):
        curr = [i]
        for j, b_token in enumerate(b_tokens, 1):
            cost = 0 if a_token == b_token else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def load_item_by_id(path: Path, target_id: str) -> dict:
    payload = _load_json_or_jsonl(path)
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


def compute_set_metrics(gt_values: list[str], pred_values: list[str]) -> dict:
    gt_norm = {normalize(value): value for value in gt_values if normalize(value)}
    pred_norm = {normalize(value): value for value in pred_values if normalize(value)}
    gt_set = set(gt_norm.keys())
    pred_set = set(pred_norm.keys())
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "exact_match": gt_set == pred_set,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": tp,
        "gt_total": len(gt_set),
        "pred_total": len(pred_set),
        "missing": [gt_norm[key] for key in sorted(gt_set - pred_set)],
        "extra": [pred_norm[key] for key in sorted(pred_set - gt_set)],
    }


def normalize_text_relaxed(text: object) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: object) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", normalize_text_relaxed(text))


def flatten_structure(obj: object, path: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            rows.extend(flatten_structure(value, next_path))
        return rows
    if isinstance(obj, list):
        for value in obj:
            rows.extend(flatten_structure(value, path))
        return rows
    value = str(obj).strip()
    if value:
        rows.append({"field_path": path, "expected_value": value})
    return rows


def value_match_score(expected: object, pred_text: str, threshold: float = 0.65) -> tuple[float, bool]:
    expected_norm = compact_text(expected)
    pred_norm = compact_text(pred_text)
    if not expected_norm:
        return 0.0, False
    if expected_norm in pred_norm:
        return 1.0, True
    score = SequenceMatcher(None, expected_norm, pred_norm).ratio()
    return score, score >= threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GT JSON against pred_structured JSON")
    parser.add_argument("--gt", required=True, help="GT path (.json/.jsonl)")
    parser.add_argument("--pred-structured", required=True, help="Pred structured JSON path")
    parser.add_argument("--id", required=True, help="Target id")
    parser.add_argument("--output", required=True, help="Output report path")
    args = parser.parse_args()

    gt_item = load_item_by_id(Path(args.gt), args.id)
    pred_item = load_item_by_id(Path(args.pred_structured), args.id)
    if "pred_structure" not in pred_item:
        raise ValueError("Pred structured JSON missing key: pred_structure")

    gt_text = str(gt_item.get("gt_text", ""))
    pred_text = str(pred_item.get("pred_text", ""))
    gt_norm = normalize(gt_text)
    pred_norm = normalize(pred_text)
    text_dist = levenshtein(gt_norm, pred_norm)
    text_cer = text_dist / max(1, len(gt_norm))
    char_similarity = max(0.0, 1.0 - (text_dist / max(1, len(gt_norm), len(pred_norm)))) * 100.0
    gt_words = gt_norm.split() if gt_norm else []
    pred_words = pred_norm.split() if pred_norm else []
    word_dist = levenshtein_tokens(gt_words, pred_words)
    text_wer = word_dist / max(1, len(gt_words))

    gt_structure = gt_item.get("gt_structure", {}) if isinstance(gt_item.get("gt_structure", {}), dict) else {}
    pred_structure = (
        pred_item.get("pred_structure", {}) if isinstance(pred_item.get("pred_structure", {}), dict) else {}
    )

    expected_fields = flatten_structure(gt_structure)
    matched_total = 0
    for field in expected_fields:
        _, matched = value_match_score(field["expected_value"], pred_text, threshold=0.65)
        if matched:
            matched_total += 1
    field_match_rate = matched_total / len(expected_fields) if expected_fields else 0.0

    report = {
        "id": args.id,
        "type": pred_item.get("type", gt_item.get("image_type", "unknown")),
        "status": pred_item.get("status", "success"),
        "latency_ms": pred_item.get("latency_ms"),
        "text": {
            "gt_text": gt_text,
            "pred_text": pred_text,
            "exact_match": gt_norm == pred_norm,
            "cer": text_cer,
            "wer": text_wer,
            "char_similarity_pct": round(char_similarity, 2),
        },
        "structure": {
            "mode": "gt_schema_value_match",
            "gt_structure": gt_structure,
            "pred_structure": pred_structure,
        },
        "field_match": {
            "rate_pct": round(field_match_rate * 100.0, 2),
            "matched": matched_total,
            "total": len(expected_fields),
            "threshold": 0.65,
        },
        "macro_f1": field_match_rate,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
