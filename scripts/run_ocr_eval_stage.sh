#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# Re-run GT evaluation only by reusing existing OCR inference artifacts
# (eval/gt_pred_structured.json) without running OCR inference again.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Activate env manually and rerun."
  exit 1
fi

OCR_ENV_NAME="${OCR_ENV_NAME:-ocr_vl15}"
OCR_CONFIG_PATH="${OCR_CONFIG_PATH:-configs/ocr_default.yaml}"
DOC_KEY="${DOC_KEY:-}"
STRUCTURE_THRESHOLD="${STRUCTURE_THRESHOLD:-0.65}"

eval "$(conda shell.bash hook)"
conda activate "${OCR_ENV_NAME}"

# Load runtime paths from OCR config
readarray -t CFG_LINES < <(python - "${OCR_CONFIG_PATH}" <<'PY'
import shlex, sys
from src.config_ocr import load_ocr_config
cfg = load_ocr_config(sys.argv[1])
print(f"IMAGES_ROOT={shlex.quote(str(cfg.paths.images_root))}")
print(f"GT_ROOT={shlex.quote(str(cfg.paths.gt_root))}")
print(f"OCR_OUTPUT_ROOT={shlex.quote(str(cfg.paths.output_root))}")
print(f"OCR_ENGINE={shlex.quote(str(cfg.ocr.engine))}")
PY
)
for line in "${CFG_LINES[@]}"; do
  eval "${line}"
done

infer_images_tag() {
  local images_root="$1"
  local marker="ocr_images/"
  local suffix=""
  local tag=""
  if [[ "${images_root}" == *"${marker}"* ]]; then
    suffix="${images_root#*"${marker}"}"
    tag="${suffix%%/*}"
  fi
  printf "%s" "${tag}"
}

IMAGES_TAG="$(infer_images_tag "${IMAGES_ROOT}")"
ENGINE_OUTPUT_ROOT="${OCR_OUTPUT_ROOT}/${OCR_ENGINE}"
if [[ -n "${IMAGES_TAG}" ]]; then
  ENGINE_OUTPUT_ROOT="${ENGINE_OUTPUT_ROOT}/${IMAGES_TAG}"
fi

echo "[OCR EVAL STAGE] ocr_config=${OCR_CONFIG_PATH}"
echo "[OCR EVAL STAGE] env=${OCR_ENV_NAME}"
echo "[OCR EVAL STAGE] images_root=${IMAGES_ROOT}"
echo "[OCR EVAL STAGE] gt_root=${GT_ROOT}"
echo "[OCR EVAL STAGE] engine_output_root=${ENGINE_OUTPUT_ROOT}"

DOC_COUNT=0
OK_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

shopt -s nullglob
for doc_dir in "${IMAGES_ROOT}"/*; do
  [[ -d "${doc_dir}" ]] || continue
  doc_key="$(basename "${doc_dir}")"
  if [[ -n "${DOC_KEY}" && "${doc_key}" != "${DOC_KEY}" ]]; then
    continue
  fi
  DOC_COUNT=$((DOC_COUNT + 1))

  for image_path in "${doc_dir}"/*; do
    [[ -f "${image_path}" ]] || continue
    case "${image_path,,}" in
      *.jpg|*.jpeg|*.png|*.bmp|*.gif|*.webp) ;;
      *) continue ;;
    esac

    image_name="$(basename "${image_path}")"
    image_stem="${image_name%.*}"
    pred_structured_path="${ENGINE_OUTPUT_ROOT}/${doc_key}/${image_stem}/eval/gt_pred_structured.json"
    out_eval_path="${ENGINE_OUTPUT_ROOT}/${doc_key}/${image_stem}/eval/gt_eval_summary.json"

    if [[ ! -s "${pred_structured_path}" ]]; then
      echo "[SKIP] pred_structured missing: doc_key=${doc_key} image=${image_name} path=${pred_structured_path}"
      SKIP_COUNT=$((SKIP_COUNT + 1))
      continue
    fi

    set +e
    python - "${GT_ROOT}" "${doc_key}" "${image_name}" "${pred_structured_path}" "${out_eval_path}" "${STRUCTURE_THRESHOLD}" <<'PY'
import sys
from pathlib import Path
from src.cli import _resolve_gt_path, _infer_item_id_from_gt
import subprocess

gt_root, doc_key, image_name, pred_structured_path, out_eval_path, threshold = sys.argv[1:]
gt_path = _resolve_gt_path(gt_root, doc_key)
if not gt_path.exists():
    raise FileNotFoundError(f"GT not found for doc_key={doc_key}: {gt_path}")
item_id = _infer_item_id_from_gt(gt_path, image_name=image_name)
subprocess.run([
    sys.executable,
    "-m",
    "src.cli",
    "eval-pred-structured",
    "--gt",
    str(Path(gt_path)),
    "--pred-structured",
    str(Path(pred_structured_path)),
    "--id",
    item_id,
    "--output",
    str(Path(out_eval_path)),
    "--structure-threshold",
    str(float(threshold)),
], check=True)
print(f"[OK] doc_key={doc_key} image={image_name} id={item_id}")
PY
    rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
      OK_COUNT=$((OK_COUNT + 1))
    else
      echo "[FAIL] doc_key=${doc_key} image=${image_name}"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
  done
done

echo "=== OCR Eval Stage Summary ==="
echo "1. docs_scanned: ${DOC_COUNT}"
echo "2. eval_ok: ${OK_COUNT}"
echo "3. eval_skip: ${SKIP_COUNT}"
echo "4. eval_fail: ${FAIL_COUNT}"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
  exit 1
fi

# Rebuild consolidated evaluation summaries for downstream review.
python - "${ENGINE_OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for eval_path in sorted(root.glob("**/eval/gt_eval_summary.json")):
    try:
        result = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    structure = result.get("structure", {}) if isinstance(result, dict) else {}
    aggregate = structure.get("aggregate", {}) if isinstance(structure, dict) else {}
    text = result.get("text", {}) if isinstance(result, dict) else {}
    table_html = result.get("table_html", {}) if isinstance(result, dict) else {}
    table_rows = result.get("table_rows", {}) if isinstance(result, dict) else {}
    review_reasons = result.get("review_reasons", [])
    if not isinstance(review_reasons, list):
        review_reasons = []
    missing_required = result.get("missing_required_fields", [])
    if not isinstance(missing_required, list):
        missing_required = []
    row = {
        "id": result.get("id"),
        "doc_key": eval_path.parents[2].name,
        "type": result.get("type"),
        "status": result.get("status"),
        "text_similarity": (float(text.get("char_similarity_pct", 0.0)) / 100.0) if text.get("char_similarity_pct") is not None else None,
        "cer": text.get("cer"),
        "wer": text.get("wer"),
        "structure_micro_recall": result.get("structure_micro_recall"),
        "structure_macro_f1": result.get("structure_macro_f1"),
        "matched_fields": aggregate.get("matched"),
        "total_fields": aggregate.get("gt_total"),
        "table_html_exists": bool(table_html.get("exists", False)),
        "table_rows_exists": bool(table_rows.get("exists", False)),
        "review_required": bool(result.get("review_required", False)),
        "review_reasons": "|".join(str(x) for x in review_reasons),
        "missing_required_fields": "|".join(str(x) for x in missing_required),
        "latency_ms": result.get("latency_ms"),
    }
    rows.append(row)

if not rows:
    print("[WARN] No gt_eval_summary.json rows found; skip consolidated summaries.")
    raise SystemExit(0)

summary_json_path = root / "ocr_eval_summary.json"
summary_csv_path = root / "ocr_eval_summary.csv"
summary_txt_path = root / "ocr_eval_summary.txt"

summary_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

fieldnames = [
    "id", "doc_key", "type", "status", "text_similarity", "cer", "wer",
    "structure_micro_recall", "structure_macro_f1", "matched_fields", "total_fields",
    "table_html_exists", "table_rows_exists", "review_required", "review_reasons",
    "missing_required_fields", "latency_ms",
]
with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = []
for row in rows:
    similarity_pct = float(row["text_similarity"]) * 100.0 if row.get("text_similarity") is not None else 0.0
    lines.extend([
        "=" * 80,
        str(row.get("id", "")),
        f"type: {row.get('type', '')}",
        f"status: {row.get('status', '')}",
        f"문자 유사도: {round(similarity_pct, 2)} %",
        f"CER: {row.get('cer', 'N/A')}",
        f"WER: {row.get('wer', 'N/A')}",
        f"구조 micro recall: {row.get('structure_micro_recall', 'N/A')} ({row.get('matched_fields')}/{row.get('total_fields')})",
        f"구조 macro f1: {row.get('structure_macro_f1', 'N/A')}",
        f"table_html_exists: {row.get('table_html_exists')}",
        f"table_rows_exists: {row.get('table_rows_exists')}",
    ])
    if row.get("review_required"):
        lines.append(f"review_required: True reasons={row.get('review_reasons')} missing_required={row.get('missing_required_fields')}")

review_count = sum(1 for r in rows if r.get("review_required"))
lines.extend([
    "",
    "[SUMMARY]",
    f"평가 이미지 수: {len(rows)}",
    f"리뷰 큐 건수: {review_count}",
])
summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[OK] saved: {summary_json_path}")
print(f"[OK] saved: {summary_csv_path}")
print(f"[OK] saved: {summary_txt_path}")
PY
