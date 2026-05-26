#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# OCR 전용 환경(ocr_vl15)에서만 실행하는 Stage-1 파이프라인.
# 1) OCR 추론/평가 산출물 생성
# 2) 품질 게이트(review_required=false) 통과 건만 RAG handoff 파일로 export
# 개발모드에서는 품질 게이트 통과 여부와 상관없이 모두 export

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

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
if [[ "${EXCLUDE_REVIEW_REQUIRED}" == "1" ]]; then
  EXPORT_ARGS+=(--exclude-review-required)
fi
if [[ -n "${DOC_KEY}" ]]; then
  EXPORT_ARGS+=(--doc-key "${DOC_KEY}")
fi

python "${EXPORT_ARGS[@]}"

echo "[OCR STAGE DONE]"
echo "exclude_review_required: ${EXCLUDE_REVIEW_REQUIRED}"
echo "manifest: ${MANIFEST_OUTPUT}"
echo "chunks: ${CHUNKS_OUTPUT}"
