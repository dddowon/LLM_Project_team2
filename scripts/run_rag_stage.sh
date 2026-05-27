#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# RAG 전용 가상환경에서 OCR handoff 입력(JSONL)을 임베딩하고 Chroma 인덱스를 구축한다.
# OCR 코드/가상환경과 분리하여 운영해 충돌을 줄인다.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RAG_ENV_NAME="${RAG_ENV_NAME:-llm_team2}"
INPUT_CHUNKS="${INPUT_CHUNKS:-data/v2/ocr_rag/ocr_input_chunks.jsonl}"
OUTPUT_EMBEDDED="${OUTPUT_EMBEDDED:-data/v2/ocr_rag/ocr_input_embedded.jsonl}"
INDEX_DIR="${INDEX_DIR:-data/v2/ocr_rag/chroma_index}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE_REAL="${FORCE_REAL:-0}"

echo "[RAG STAGE] root=${ROOT_DIR}"
echo "[RAG STAGE] env=${RAG_ENV_NAME}"
echo "[RAG STAGE] input=${INPUT_CHUNKS}"

if [[ ! -f "${INPUT_CHUNKS}" ]]; then
  echo "[ERROR] input chunks not found: ${INPUT_CHUNKS}"
  echo "Run scripts/run_ocr_stage.sh first."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Activate '${RAG_ENV_NAME}' manually and rerun."
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${RAG_ENV_NAME}"

EMBED_ARGS=(
  -m src.cli embed-jsonl
  --input "${INPUT_CHUNKS}"
  --output "${OUTPUT_EMBEDDED}"
  --model "${EMBED_MODEL}"
  --batch-size "${BATCH_SIZE}"
)
if [[ "${FORCE_REAL}" == "1" ]]; then
  EMBED_ARGS+=(--force-real)
fi

python "${EMBED_ARGS[@]}"

python -m src.cli build-chroma \
  --input "${OUTPUT_EMBEDDED}" \
  --index-dir "${INDEX_DIR}"

echo "[RAG STAGE DONE]"
echo "embedded: ${OUTPUT_EMBEDDED}"
echo "index: ${INDEX_DIR}"
