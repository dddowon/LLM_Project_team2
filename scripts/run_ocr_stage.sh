#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# OCR 전용 환경(ocr_vl15)에서 실행하는 Stage-1 파이프라인.
# 1) OCR 추론/평가 산출물 생성 (TABLE_DUAL_PASS=1일 때 표 영역 2-stage 추론 활성화)
# 2) OCR 산출물을 RAG handoff(JSONL)로 export
# 기본값은 전량 export이며, EXCLUDE_REVIEW_REQUIRED=1 일 때만 review_required 건을 제외한다.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAGE_START_TS="$(date +%s)"

format_elapsed() {
  local elapsed="$1"
  local h=$((elapsed / 3600))
  local m=$(((elapsed % 3600) / 60))
  local s=$((elapsed % 60))
  printf "%02dh %02dm %02ds" "${h}" "${m}" "${s}"
}

OCR_ENV_NAME="${OCR_ENV_NAME:-ocr_vl15}"
OCR_ENGINE="${OCR_ENGINE:-paddleocr_vl}"
IMAGES_ROOT="${IMAGES_ROOT:-data/v2/ocr_images}"
GT_ROOT="${GT_ROOT:-data/v2/ocr_outputs/incoming_gt}"
OCR_OUTPUT_ROOT="${OCR_OUTPUT_ROOT:-data/v2/ocr_outputs}"
OCR_CONFIG_PATH="${OCR_CONFIG_PATH:-configs/ocr_default.yaml}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.0}"
STRUCTURE_THRESHOLD="${STRUCTURE_THRESHOLD:-0.65}"
DOC_KEY="${DOC_KEY:-}"

RAG_HANDOFF_DIR="${RAG_HANDOFF_DIR:-data/v2/ocr_rag}"
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-${RAG_HANDOFF_DIR}/ocr_input_manifest.jsonl}"
CHUNKS_OUTPUT="${CHUNKS_OUTPUT:-${RAG_HANDOFF_DIR}/ocr_input_chunks.jsonl}"
EXCLUDE_REVIEW_REQUIRED="${EXCLUDE_REVIEW_REQUIRED:-0}"
INCLUDE_HTML_CHUNK="${INCLUDE_HTML_CHUNK:-0}"
HTML_CHUNK_MAX_CHARS="${HTML_CHUNK_MAX_CHARS:-1200}"
USE_DOC_UNWARPING="${USE_DOC_UNWARPING:-1}"
TABLE_DUAL_PASS="${TABLE_DUAL_PASS:-0}"
OCR_USE_GT="${OCR_USE_GT:-0}"

echo "[OCR STAGE] root=${ROOT_DIR}"
echo "[OCR STAGE] env=${OCR_ENV_NAME}"
echo "[OCR STAGE] engine=${OCR_ENGINE}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Activate '${OCR_ENV_NAME}' manually and rerun."
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${OCR_ENV_NAME}"

OCR_BATCH_ARGS=(
  -m src.cli ocr-run-batch
  --images-root "${IMAGES_ROOT}"
  --gt-root "${GT_ROOT}"
  --output-root "${OCR_OUTPUT_ROOT}"
  --ocr-config "${OCR_CONFIG_PATH}"
  --score-threshold "${SCORE_THRESHOLD}"
  --structure-threshold "${STRUCTURE_THRESHOLD}"
)
if [[ "${USE_DOC_UNWARPING}" == "1" ]]; then
  OCR_BATCH_ARGS+=(--use-doc-unwarping)
fi
if [[ "${TABLE_DUAL_PASS}" == "1" ]]; then
  OCR_BATCH_ARGS+=(--table-dual-pass)
fi
if [[ "${OCR_USE_GT}" == "0" ]]; then
  OCR_BATCH_ARGS+=(--no-gt)
fi
if [[ -n "${DOC_KEY}" ]]; then
  OCR_BATCH_ARGS+=(--doc-key "${DOC_KEY}")
fi

python "${OCR_BATCH_ARGS[@]}"

mkdir -p "${RAG_HANDOFF_DIR}"

EXPORT_ARGS=(
  -m src.cli ocr-export-rag
  --ocr-eval-root "${OCR_OUTPUT_ROOT}"
  --engine "${OCR_ENGINE}"
  --output-manifest "${MANIFEST_OUTPUT}"
  --output-chunks "${CHUNKS_OUTPUT}"
)
if [[ "${OCR_USE_GT}" == "0" ]]; then
  EXPORT_ARGS+=(--allow-inference-only)
fi
if [[ "${EXCLUDE_REVIEW_REQUIRED}" == "1" ]]; then
  EXPORT_ARGS+=(--exclude-review-required)
fi
if [[ "${INCLUDE_HTML_CHUNK}" == "1" ]]; then
  EXPORT_ARGS+=(--include-html-chunk --html-chunk-max-chars "${HTML_CHUNK_MAX_CHARS}")
fi
if [[ -n "${DOC_KEY}" ]]; then
  EXPORT_ARGS+=(--doc-key "${DOC_KEY}")
fi

python "${EXPORT_ARGS[@]}"

STAGE_END_TS="$(date +%s)"
STAGE_ELAPSED_SEC="$((STAGE_END_TS - STAGE_START_TS))"
STAGE_TOTAL_LATENCY_MS="$((STAGE_ELAPSED_SEC * 1000))"
STAGE_TOTAL_LATENCY_HMS="$(format_elapsed "${STAGE_ELAPSED_SEC}")"

ENGINE_SUMMARY_DIR="${OCR_OUTPUT_ROOT}/${OCR_ENGINE}"
mkdir -p "${ENGINE_SUMMARY_DIR}"
STAGE_SUMMARY_TXT="${ENGINE_SUMMARY_DIR}/ocr_stage_summary.txt"
STAGE_SUMMARY_JSON="${ENGINE_SUMMARY_DIR}/ocr_inference_stage_summary.json"

cat <<EOF
=== OCR Stage Summary ===
1. stage: done
2. ocr_use_gt: ${OCR_USE_GT}
3. exclude_review_required: ${EXCLUDE_REVIEW_REQUIRED}
4. use_doc_unwarping: ${USE_DOC_UNWARPING}
5. table_dual_pass: ${TABLE_DUAL_PASS}
6. include_html_chunk: ${INCLUDE_HTML_CHUNK}
7. html_chunk_max_chars: ${HTML_CHUNK_MAX_CHARS}
8. manifest: ${MANIFEST_OUTPUT}
9. chunks: ${CHUNKS_OUTPUT}
10. total_latency_ms: ${STAGE_TOTAL_LATENCY_MS}
11. total_latency_hms: ${STAGE_TOTAL_LATENCY_HMS}
EOF

cat > "${STAGE_SUMMARY_TXT}" <<EOF
=== OCR Stage Summary ===
1. stage: done
2. ocr_use_gt: ${OCR_USE_GT}
3. exclude_review_required: ${EXCLUDE_REVIEW_REQUIRED}
4. use_doc_unwarping: ${USE_DOC_UNWARPING}
5. table_dual_pass: ${TABLE_DUAL_PASS}
6. include_html_chunk: ${INCLUDE_HTML_CHUNK}
7. html_chunk_max_chars: ${HTML_CHUNK_MAX_CHARS}
8. manifest: ${MANIFEST_OUTPUT}
9. chunks: ${CHUNKS_OUTPUT}
10. total_latency_ms: ${STAGE_TOTAL_LATENCY_MS}
11. total_latency_hms: ${STAGE_TOTAL_LATENCY_HMS}
EOF

cat > "${STAGE_SUMMARY_JSON}" <<EOF
{
  "title": "OCR Stage Summary",
  "stage": "done",
  "ocr_use_gt": ${OCR_USE_GT},
  "exclude_review_required": ${EXCLUDE_REVIEW_REQUIRED},
  "use_doc_unwarping": ${USE_DOC_UNWARPING},
  "table_dual_pass": ${TABLE_DUAL_PASS},
  "include_html_chunk": ${INCLUDE_HTML_CHUNK},
  "html_chunk_max_chars": ${HTML_CHUNK_MAX_CHARS},
  "manifest": "${MANIFEST_OUTPUT}",
  "chunks": "${CHUNKS_OUTPUT}",
  "total_latency_ms": ${STAGE_TOTAL_LATENCY_MS},
  "total_latency_hms": "${STAGE_TOTAL_LATENCY_HMS}"
}
EOF

echo "12. output"
echo "12-1. saved_ocr_stage_summary_json: ${STAGE_SUMMARY_JSON}"
echo "12-2. saved_ocr_stage_summary_txt: ${STAGE_SUMMARY_TXT}"
