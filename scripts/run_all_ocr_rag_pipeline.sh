#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# OCR + RAG를 한 번에 실행하는 통합 엔트리포인트.
# 내부적으로 기존 stage 스크립트를 호출해 중복 구현을 피하고,
# 디버깅 시에는 stage 스크립트를 개별 실행할 수 있게 유지한다.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_RAG_STAGE="${RUN_RAG_STAGE:-1}" # 1: OCR 후 RAG까지 실행, 0: OCR만 실행

# OCR stage env/args
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
USE_DOC_UNWARPING="${USE_DOC_UNWARPING:-0}"
TABLE_DUAL_PASS="${TABLE_DUAL_PASS:-0}"
OCR_NO_GT="${OCR_NO_GT:-1}"

# RAG stage env/args
RAG_ENV_NAME="${RAG_ENV_NAME:-llm_team2}"
INPUT_CHUNKS="${INPUT_CHUNKS:-${CHUNKS_OUTPUT}}"
OUTPUT_EMBEDDED="${OUTPUT_EMBEDDED:-data/v2/ocr_rag/ocr_input_embedded.jsonl}"
INDEX_DIR="${INDEX_DIR:-data/v2/ocr_rag/chroma_index}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE_REAL="${FORCE_REAL:-0}"

echo "[PIPELINE] root=${ROOT_DIR}"
echo "[PIPELINE] run_rag_stage=${RUN_RAG_STAGE}"
echo "[PIPELINE] doc_key=${DOC_KEY:-<all>}"

OCR_ENV_NAME="${OCR_ENV_NAME}" \
OCR_ENGINE="${OCR_ENGINE}" \
IMAGES_ROOT="${IMAGES_ROOT}" \
GT_ROOT="${GT_ROOT}" \
OCR_OUTPUT_ROOT="${OCR_OUTPUT_ROOT}" \
OCR_CONFIG_PATH="${OCR_CONFIG_PATH}" \
SCORE_THRESHOLD="${SCORE_THRESHOLD}" \
STRUCTURE_THRESHOLD="${STRUCTURE_THRESHOLD}" \
DOC_KEY="${DOC_KEY}" \
RAG_HANDOFF_DIR="${RAG_HANDOFF_DIR}" \
MANIFEST_OUTPUT="${MANIFEST_OUTPUT}" \
CHUNKS_OUTPUT="${CHUNKS_OUTPUT}" \
EXCLUDE_REVIEW_REQUIRED="${EXCLUDE_REVIEW_REQUIRED}" \
INCLUDE_HTML_CHUNK="${INCLUDE_HTML_CHUNK}" \
HTML_CHUNK_MAX_CHARS="${HTML_CHUNK_MAX_CHARS}" \
USE_DOC_UNWARPING="${USE_DOC_UNWARPING}" \
TABLE_DUAL_PASS="${TABLE_DUAL_PASS}" \
OCR_NO_GT="${OCR_NO_GT}" \
./scripts/run_ocr_stage.sh

if [[ "${OCR_NO_GT}" == "1" ]]; then
  echo "[PIPELINE] OCR_NO_GT=1 -> skip RAG stage"
  exit 0
fi

if [[ "${RUN_RAG_STAGE}" != "1" ]]; then
  echo "[PIPELINE] RUN_RAG_STAGE=${RUN_RAG_STAGE} -> skip RAG stage"
  exit 0
fi

if [[ ! -s "${INPUT_CHUNKS}" ]]; then
  echo "[PIPELINE] chunks file is missing or empty: ${INPUT_CHUNKS}"
  echo "[PIPELINE] skip RAG stage"
  exit 0
fi

RAG_ENV_NAME="${RAG_ENV_NAME}" \
INPUT_CHUNKS="${INPUT_CHUNKS}" \
OUTPUT_EMBEDDED="${OUTPUT_EMBEDDED}" \
INDEX_DIR="${INDEX_DIR}" \
EMBED_MODEL="${EMBED_MODEL}" \
BATCH_SIZE="${BATCH_SIZE}" \
FORCE_REAL="${FORCE_REAL}" \
./scripts/run_rag_stage.sh

echo "[PIPELINE DONE]"
