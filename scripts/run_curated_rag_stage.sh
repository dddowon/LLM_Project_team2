#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# OCR 재추론 없이 curated JSON 기반으로 OCR->RAG handoff를 재생성하고,
# STRICT_CURATED 게이트를 통과한 뒤 RAG 인덱스를 재생성한다.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OCR_ENV_NAME="${OCR_ENV_NAME:-ocr_vl15}"
RAG_ENV_NAME="${RAG_ENV_NAME:-llm_team2}"
OCR_CONFIG_PATH="${OCR_CONFIG_PATH:-configs/ocr_default.yaml}"
OCR_EVAL_ROOT="${OCR_EVAL_ROOT:-data/v2/ocr_outputs}"
OCR_ENGINE="${OCR_ENGINE:-paddleocr_vl}"
OCR_OUTPUT_VERSION="${OCR_OUTPUT_VERSION:-}"
DOC_KEY="${DOC_KEY:-}"
CURATED_ROOT="${CURATED_ROOT:-data/v2/ocr_curated}"
CURATED_FILE_NAME="${CURATED_FILE_NAME:-pred_table_layout.curated.json}"
OCR_CURATED_VERSION="${OCR_CURATED_VERSION:-}"
INPUT_VERSION="${INPUT_VERSION:-}"
OCR_ENGINE_VERSION="${OCR_ENGINE_VERSION:-}"
RAG_INDEX_VERSION="${RAG_INDEX_VERSION:-}"
EXCLUDE_REVIEW_REQUIRED="${EXCLUDE_REVIEW_REQUIRED:-0}"
INCLUDE_HTML_CHUNK="${INCLUDE_HTML_CHUNK:-0}"
HTML_CHUNK_MAX_CHARS="${HTML_CHUNK_MAX_CHARS:-1200}"
INPUT_CHUNKS="${INPUT_CHUNKS:-}"
OUTPUT_EMBEDDED="${OUTPUT_EMBEDDED:-}"
INDEX_DIR="${INDEX_DIR:-}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE_REAL="${FORCE_REAL:-0}"
STRICT_CURATED="${STRICT_CURATED:-1}"

if [[ -z "${OCR_OUTPUT_VERSION}" ]]; then
  echo "[ERROR] OCR_OUTPUT_VERSION is required. Example: OCR_OUTPUT_VERSION=v1_extracted_unfiltered"
  exit 1
fi
if [[ -z "${OCR_CURATED_VERSION}" ]]; then
  echo "[ERROR] OCR_CURATED_VERSION is required. Example: OCR_CURATED_VERSION=v4_curated_20260601_1542"
  exit 1
fi
if [[ "${STRICT_CURATED}" != "1" ]]; then
  echo "[ERROR] run_curated_rag_stage.sh requires STRICT_CURATED=1"
  exit 1
fi

RAG_HANDOFF_DIR="data/v2/ocr_rag/${OCR_ENGINE}/${OCR_OUTPUT_VERSION}"
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-${RAG_HANDOFF_DIR}/ocr_input_manifest.jsonl}"
CHUNKS_OUTPUT="${CHUNKS_OUTPUT:-${RAG_HANDOFF_DIR}/ocr_input_chunks.jsonl}"
INPUT_CHUNKS="${INPUT_CHUNKS:-${CHUNKS_OUTPUT}}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found."
  exit 1
fi

echo "[CURATED RELEASE] ocr_output_version=${OCR_OUTPUT_VERSION}"
echo "[CURATED RELEASE] ocr_curated_version=${OCR_CURATED_VERSION}"
echo "[CURATED RELEASE] curated_root=${CURATED_ROOT}"

eval "$(conda shell.bash hook)"
conda activate "${OCR_ENV_NAME}"

EXPORT_ARGS=(
  -m src.cli ocr-export-rag
  --ocr-eval-root "${OCR_EVAL_ROOT}"
  --engine "${OCR_ENGINE}"
  --images-tag "${OCR_OUTPUT_VERSION}"
  --curated-root "${CURATED_ROOT}"
  --curated-file-name "${CURATED_FILE_NAME}"
  --ocr-curated-version "${OCR_CURATED_VERSION}"
  --ocr-output-version "${OCR_OUTPUT_VERSION}"
  --output-manifest "${MANIFEST_OUTPUT}"
  --output-chunks "${CHUNKS_OUTPUT}"
  --allow-inference-only
)
if [[ -n "${INPUT_VERSION}" ]]; then
  EXPORT_ARGS+=(--input-version "${INPUT_VERSION}")
fi
if [[ -n "${OCR_ENGINE_VERSION}" ]]; then
  EXPORT_ARGS+=(--ocr-engine-version "${OCR_ENGINE_VERSION}")
fi
if [[ -n "${RAG_INDEX_VERSION}" ]]; then
  EXPORT_ARGS+=(--rag-index-version "${RAG_INDEX_VERSION}")
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

python - "${MANIFEST_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"[ERROR] manifest not found: {path}")
    raise SystemExit(1)
rows = 0
raw_rows = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows += 1
        obj = json.loads(line)
        if str(obj.get("table_source", "")).strip().lower() == "raw":
            raw_rows += 1
if rows == 0:
    print(f"[ERROR] empty manifest: {path}")
    raise SystemExit(1)
if raw_rows:
    print(f"[ERROR] STRICT_CURATED failed: table_source=raw rows={raw_rows}/{rows}")
    raise SystemExit(1)
print(f"[OK] STRICT_CURATED passed: all rows are curated ({rows}/{rows})")
PY

conda activate "${RAG_ENV_NAME}"
RAG_INDEX_VERSION="${RAG_INDEX_VERSION:-rag_${OCR_CURATED_VERSION}}"

RAG_ENV_NAME="${RAG_ENV_NAME}" \
OCR_CONFIG_PATH="${OCR_CONFIG_PATH}" \
INPUT_CHUNKS="${INPUT_CHUNKS}" \
OUTPUT_EMBEDDED="${OUTPUT_EMBEDDED}" \
INDEX_DIR="${INDEX_DIR}" \
EMBED_MODEL="${EMBED_MODEL}" \
BATCH_SIZE="${BATCH_SIZE}" \
FORCE_REAL="${FORCE_REAL}" \
RAG_INDEX_VERSION="${RAG_INDEX_VERSION}" \
./scripts/run_rag_stage.sh

echo "[CURATED RELEASE DONE]"
echo "manifest: ${MANIFEST_OUTPUT}"
echo "chunks: ${CHUNKS_OUTPUT}"
