from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


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


def normalize_text_relaxed(text: object) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: object) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", normalize_text_relaxed(text))


def value_match_score(expected: object, predicted: object, threshold: float = 0.65) -> tuple[float, bool]:
    expected_norm = compact_text(expected)
    pred_norm = compact_text(predicted)
    if not expected_norm:
        return 0.0, False
    if expected_norm in pred_norm:
        return 1.0, True
    score = SequenceMatcher(None, expected_norm, pred_norm).ratio()
    return score, score >= threshold


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = compact_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(value).strip())
    return deduped


def flatten_structure_values(obj: object, path: str = "", acc: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    if acc is None:
        acc = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            flatten_structure_values(value, next_path, acc)
        return acc

    if isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                flatten_structure_values(value, path, acc)
                continue
            text = str(value).strip()
            if text:
                acc.setdefault(path, []).append(text)
        return acc

    text = str(obj).strip()
    if text:
        acc.setdefault(path, []).append(text)
    return acc


def match_field_values(gt_values: list[str], pred_values: list[str], threshold: float) -> dict[str, Any]:
    expected = _dedupe_values(gt_values)
    predicted = _dedupe_values(pred_values)

    candidate_pairs: list[tuple[float, int, int]] = []
    for expected_idx, expected_value in enumerate(expected):
        for predicted_idx, predicted_value in enumerate(predicted):
            score, matched = value_match_score(expected_value, predicted_value, threshold=threshold)
            if matched:
                candidate_pairs.append((score, expected_idx, predicted_idx))
    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    matched_pairs: list[dict[str, Any]] = []
    for score, expected_idx, predicted_idx in candidate_pairs:
        if expected_idx in used_expected or predicted_idx in used_predicted:
            continue
        used_expected.add(expected_idx)
        used_predicted.add(predicted_idx)
        matched_pairs.append(
            {
                "gt_value": expected[expected_idx],
                "pred_value": predicted[predicted_idx],
                "score": round(score, 4),
            }
        )

    matched = len(matched_pairs)
    gt_total = len(expected)
    pred_total = len(predicted)
    precision = matched / pred_total if pred_total else (1.0 if gt_total == 0 else 0.0)
    recall = matched / gt_total if gt_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    missing = [value for idx, value in enumerate(expected) if idx not in used_expected]
    extra = [value for idx, value in enumerate(predicted) if idx not in used_predicted]
    return {
        "exact_match": not missing and not extra,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "matched": matched,
        "gt_total": gt_total,
        "pred_total": pred_total,
        "missing": missing,
        "extra": extra,
        "matched_pairs": matched_pairs,
    }


def evaluate_structure(
    gt_structure: dict[str, Any], pred_structure: dict[str, Any], threshold: float
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    gt_map = flatten_structure_values(gt_structure)
    pred_map = flatten_structure_values(pred_structure)
    field_paths = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    field_metrics: list[dict[str, Any]] = []
    matched_total = 0
    gt_total = 0
    pred_total = 0
    macro_f1_values: list[float] = []

    for field_path in field_paths:
        per_field = match_field_values(gt_map.get(field_path, []), pred_map.get(field_path, []), threshold)
        per_field["field_path"] = field_path
        field_metrics.append(per_field)

        matched_total += int(per_field["matched"])
        gt_total += int(per_field["gt_total"])
        pred_total += int(per_field["pred_total"])
        macro_f1_values.append(float(per_field["f1"]))

    micro_precision = matched_total / pred_total if pred_total else (1.0 if gt_total == 0 else 0.0)
    micro_recall = matched_total / gt_total if gt_total else 1.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    macro_f1 = sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0

    aggregate = {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "matched": matched_total,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "field_count": len(field_paths),
    }
    return field_metrics, aggregate


def _build_gt_structure_from_typed_gt(gt_item: dict[str, Any]) -> dict[str, Any]:
    image_type = str(gt_item.get("image_type", "")).strip().lower()
    if image_type in {"table", "table_form"}:
        table_gt = gt_item.get("table_gt")
        if isinstance(table_gt, dict):
            grid_text = table_gt.get("grid_text")
            if isinstance(grid_text, list):
                flat_cells: list[str] = []
                for row in grid_text:
                    if not isinstance(row, list):
                        continue
                    for cell in row:
                        text = str(cell).strip()
                        if text:
                            flat_cells.append(text)
                if flat_cells:
                    return {"table_text": flat_cells}
            return {}
    if image_type == "chart":
        chart_gt = gt_item.get("chart_gt")
        if not isinstance(chart_gt, dict):
            return {}
        ticks = ((chart_gt.get("x_axis") or {}).get("ticks") or [])
        series = chart_gt.get("series") or []
        structure: dict[str, Any] = {}
        if ticks:
            structure["컬럼"] = [str(x) for x in ticks]
        stat_map: dict[str, dict[str, str]] = {}
        for one_series in series:
            if not isinstance(one_series, dict):
                continue
            name = str(one_series.get("name", "")).strip()
            values = one_series.get("values") or []
            if not name:
                continue
            stat_map[name] = {str(year): str(val) for year, val in zip(ticks, values)}
        if stat_map:
            structure["통계"] = stat_map
        if structure:
            return structure
        derived_text = ((gt_item.get("derived") or {}).get("gt_text_generated") or "")
        if derived_text:
            return {"chart_text": [str(derived_text)]}
    if image_type == "diagram":
        diagram_gt = gt_item.get("diagram_gt")
        if isinstance(diagram_gt, dict):
            flat = flatten_structure_values(diagram_gt)
            texts: list[str] = []
            for values in flat.values():
                for v in values:
                    value = str(v).strip()
                    if value:
                        texts.append(value)
            if texts:
                return {"diagram_text": texts}
        derived_text = ((gt_item.get("derived") or {}).get("gt_text_generated") or "")
        if derived_text:
            return {"diagram_text": [str(derived_text)]}
    return {}


def _normalize_pred_structure_by_type(pred_item: dict[str, Any], image_type: str) -> dict[str, Any]:
    pred_structure = pred_item.get("pred_structure", {})
    if not isinstance(pred_structure, dict):
        pred_structure = {}
    pred_text = str(pred_item.get("pred_text", "")).strip()
    t = image_type.lower()
    if t in {"table", "table_form"}:
        values: list[str] = []
        for one in pred_structure.values():
            if isinstance(one, list):
                values.extend(str(v).strip() for v in one if str(v).strip())
            else:
                value = str(one).strip()
                if value:
                    values.append(value)
        if not values and pred_text:
            values = [pred_text]
        return {"table_text": values}
    if t == "chart":
        values = []
        for one in pred_structure.values():
            if isinstance(one, list):
                values.extend(str(v).strip() for v in one if str(v).strip())
            else:
                value = str(one).strip()
                if value:
                    values.append(value)
        if not values and pred_text:
            values = [pred_text]
        return {"chart_text": values}
    if t == "diagram":
        values = []
        for one in pred_structure.values():
            if isinstance(one, list):
                values.extend(str(v).strip() for v in one if str(v).strip())
            else:
                value = str(one).strip()
                if value:
                    values.append(value)
        if not values and pred_text:
            values = [pred_text]
        return {"diagram_text": values}
    return pred_structure


def build_eval_report(
    *,
    item_id: str,
    gt_item: dict[str, Any],
    pred_item: dict[str, Any],
    threshold: float = 0.65,
) -> dict[str, Any]:
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

    # [Design Intent]
    # Use typed GT blocks (chart_gt/table_gt/diagram_gt) as the single source of truth.
    # Do not depend on optional legacy `gt_structure`.
    image_type = str(gt_item.get("image_type", pred_item.get("type", "unknown"))).strip().lower()
    gt_structure = _build_gt_structure_from_typed_gt(gt_item)
    pred_structure = _normalize_pred_structure_by_type(pred_item, image_type)

    field_metrics, aggregate = evaluate_structure(gt_structure, pred_structure, threshold=threshold)

    return {
        "id": item_id,
        "type": pred_item.get("type", gt_item.get("image_type", "unknown")),
        "status": pred_item.get("status", "success"),
        "latency_ms": pred_item.get("latency_ms"),
        "required_fields": gt_item.get("required_fields", []),
        "text": {
            "gt_text": gt_text,
            "pred_text": pred_text,
            "exact_match": gt_norm == pred_norm,
            "cer": text_cer,
            "wer": text_wer,
            "char_similarity_pct": round(char_similarity, 2),
        },
        "structure": {
            "mode": "field_path_value_match",
            "threshold": threshold,
            "gt_structure": gt_structure,
            "pred_structure": pred_structure,
            "field_metrics": field_metrics,
            "aggregate": {
                "field_count": aggregate["field_count"],
                "matched": aggregate["matched"],
                "gt_total": aggregate["gt_total"],
                "pred_total": aggregate["pred_total"],
                "micro_precision": round(aggregate["micro_precision"], 6),
                "micro_recall": round(aggregate["micro_recall"], 6),
                "micro_f1": round(aggregate["micro_f1"], 6),
                "macro_f1": round(aggregate["macro_f1"], 6),
            },
        },
        "structure_micro_recall": round(aggregate["micro_recall"], 6),
        "structure_macro_f1": round(aggregate["macro_f1"], 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GT JSON against pred_structured JSON")
    parser.add_argument("--gt", required=True, help="GT path (.json/.jsonl)")
    parser.add_argument("--pred-structured", required=True, help="Pred structured JSON path")
    parser.add_argument("--id", required=True, help="Target id")
    parser.add_argument("--output", required=True, help="Output report path")
    parser.add_argument("--threshold", type=float, default=0.65, help="Value match threshold")
    args = parser.parse_args()

    gt_item = load_item_by_id(Path(args.gt), args.id)
    pred_item = load_item_by_id(Path(args.pred_structured), args.id)
    report = build_eval_report(
        item_id=args.id,
        gt_item=gt_item,
        pred_item=pred_item,
        threshold=args.threshold,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
