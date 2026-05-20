from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GT JSON against pred_structured JSON")
    parser.add_argument("--gt", required=True, help="GT JSON path")
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

    fields = ["기관명", "영문명", "로고"]
    structure_report: dict[str, dict] = {}
    f1_scores: list[float] = []
    matched_total = 0
    gt_total = 0
    gt_structure = gt_item.get("gt_structure", {}) if isinstance(gt_item.get("gt_structure", {}), dict) else {}
    pred_structure = (
        pred_item.get("pred_structure", {}) if isinstance(pred_item.get("pred_structure", {}), dict) else {}
    )

    for field in fields:
        gt_values = gt_structure.get(field, [])
        pred_values = pred_structure.get(field, [])
        if not isinstance(gt_values, list):
            gt_values = []
        if not isinstance(pred_values, list):
            pred_values = []
        metrics = compute_set_metrics([str(v) for v in gt_values], [str(v) for v in pred_values])
        structure_report[field] = metrics
        f1_scores.append(metrics["f1"])
        matched_total += metrics["matched"]
        gt_total += metrics["gt_total"]

    field_match_rate = (matched_total / gt_total * 100.0) if gt_total else 0.0

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
        "structure": structure_report,
        "field_match": {
            "rate_pct": round(field_match_rate, 2),
            "matched": matched_total,
            "total": gt_total,
        },
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
