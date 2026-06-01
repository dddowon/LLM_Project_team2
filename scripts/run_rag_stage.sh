#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# RAG 전용 가상환경에서 OCR handoff 입력(JSONL)을 임베딩하고 Chroma 인덱스를 구축한다.
# OCR 코드/가상환경과 분리하여 운영해 충돌을 줄인다.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

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

load_ocr_runtime_from_config() {
  local config_path="$1"
  python - "${config_path}" <<'PY'
import shlex
import sys
from src.config_ocr import load_ocr_config

cfg = load_ocr_config(sys.argv[1])
pairs = {
    "OCR_ENGINE": cfg.ocr.engine,
    "OCR_IMAGES_ROOT": cfg.paths.images_root,
}
for key, value in pairs.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

RAG_ENV_NAME="${RAG_ENV_NAME:-llm_team2}"
OCR_CONFIG_PATH="${OCR_CONFIG_PATH:-configs/ocr_default.yaml}"
INPUT_CHUNKS="${INPUT_CHUNKS:-data/v2/ocr_rag/ocr_input_chunks.jsonl}"
OUTPUT_EMBEDDED="${OUTPUT_EMBEDDED:-}"
INDEX_DIR="${INDEX_DIR:-}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE_REAL="${FORCE_REAL:-0}"

echo "[RAG STAGE] root=${ROOT_DIR}"
echo "[RAG STAGE] env=${RAG_ENV_NAME}"
echo "[RAG STAGE] ocr_config=${OCR_CONFIG_PATH}"

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

eval "$(load_ocr_runtime_from_config "${OCR_CONFIG_PATH}")"
OCR_IMAGES_TAG="$(infer_images_tag "${OCR_IMAGES_ROOT}")"
RAG_VARIANT_DIR="data/v2/ocr_rag/${OCR_ENGINE}"
if [[ -n "${OCR_IMAGES_TAG}" ]]; then
  RAG_VARIANT_DIR="${RAG_VARIANT_DIR}/${OCR_IMAGES_TAG}"
fi

OUTPUT_EMBEDDED="${OUTPUT_EMBEDDED:-${RAG_VARIANT_DIR}/ocr_input_embedded.jsonl}"
INDEX_DIR="${INDEX_DIR:-${RAG_VARIANT_DIR}/chroma_index}"

echo "[RAG STAGE] input=${INPUT_CHUNKS}"
echo "[RAG STAGE] ocr_engine=${OCR_ENGINE}"
echo "[RAG STAGE] images_tag=${OCR_IMAGES_TAG:-<none>}"
echo "[RAG STAGE] output_embedded=${OUTPUT_EMBEDDED}"
echo "[RAG STAGE] index_dir=${INDEX_DIR}"

mkdir -p "$(dirname "${OUTPUT_EMBEDDED}")" "${INDEX_DIR}"

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
