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
    "IMAGES_ROOT": cfg.paths.images_root,
    "GT_ROOT": cfg.paths.gt_root,
    "OCR_OUTPUT_ROOT": cfg.paths.output_root,
    "SCORE_THRESHOLD": cfg.ocr.score_threshold,
    "STRUCTURE_THRESHOLD": cfg.ocr.structure_match_threshold,
}
for key, value in pairs.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

OCR_ENV_NAME="${OCR_ENV_NAME:-ocr_vl15}"
OCR_CONFIG_PATH="${OCR_CONFIG_PATH:-configs/ocr_default.yaml}"
DOC_KEY="${DOC_KEY:-}"

RAG_HANDOFF_DIR="${RAG_HANDOFF_DIR:-}"
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-}"
CHUNKS_OUTPUT="${CHUNKS_OUTPUT:-}"
EXCLUDE_REVIEW_REQUIRED="${EXCLUDE_REVIEW_REQUIRED:-0}"
INCLUDE_HTML_CHUNK="${INCLUDE_HTML_CHUNK:-0}"
HTML_CHUNK_MAX_CHARS="${HTML_CHUNK_MAX_CHARS:-1200}"
USE_DOC_UNWARPING="${USE_DOC_UNWARPING:-1}"
TABLE_DUAL_PASS="${TABLE_DUAL_PASS:-0}"
OCR_USE_GT="${OCR_USE_GT:-0}"
CURATED_ROOT="${CURATED_ROOT:-}"
CURATED_FILE_NAME="${CURATED_FILE_NAME:-pred_table_layout.curated.json}"
INPUT_VERSION="${INPUT_VERSION:-}"
OCR_ENGINE_VERSION="${OCR_ENGINE_VERSION:-}"
OCR_OUTPUT_VERSION="${OCR_OUTPUT_VERSION:-}"
OCR_CURATED_VERSION="${OCR_CURATED_VERSION:-}"
RAG_INDEX_VERSION="${RAG_INDEX_VERSION:-}"
STRICT_CURATED="${STRICT_CURATED:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Activate '${OCR_ENV_NAME}' manually and rerun."
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${OCR_ENV_NAME}"

eval "$(load_ocr_runtime_from_config "${OCR_CONFIG_PATH}")"
OCR_IMAGES_TAG="$(infer_images_tag "${IMAGES_ROOT}")"
RAG_HANDOFF_DIR="${RAG_HANDOFF_DIR:-data/v2/ocr_rag/${OCR_ENGINE}}"
if [[ -n "${OCR_IMAGES_TAG}" ]]; then
  RAG_HANDOFF_DIR="${RAG_HANDOFF_DIR}/${OCR_IMAGES_TAG}"
fi
OCR_ENGINE_VERSION="${OCR_ENGINE_VERSION:-${OCR_ENGINE}}"
OCR_OUTPUT_VERSION="${OCR_OUTPUT_VERSION:-${OCR_IMAGES_TAG}}"
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-${RAG_HANDOFF_DIR}/ocr_input_manifest.jsonl}"
CHUNKS_OUTPUT="${CHUNKS_OUTPUT:-${RAG_HANDOFF_DIR}/ocr_input_chunks.jsonl}"

echo "[OCR STAGE] root=${ROOT_DIR}"
echo "[OCR STAGE] env=${OCR_ENV_NAME}"
echo "[OCR STAGE] ocr_config=${OCR_CONFIG_PATH}"
echo "[OCR STAGE] engine=${OCR_ENGINE}"
echo "[OCR STAGE] images_root=${IMAGES_ROOT}"
echo "[OCR STAGE] gt_root=${GT_ROOT}"
echo "[OCR STAGE] output_root=${OCR_OUTPUT_ROOT}"
if [[ -n "${OCR_IMAGES_TAG}" ]]; then
  echo "[OCR STAGE] images_tag=${OCR_IMAGES_TAG}"
fi
if [[ -n "${CURATED_ROOT}" ]]; then
  echo "[OCR STAGE] curated_root=${CURATED_ROOT}"
  echo "[OCR STAGE] curated_file_name=${CURATED_FILE_NAME}"
fi

OCR_BATCH_ARGS=(
  -m src.cli ocr-run-batch
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
if [[ -n "${OCR_IMAGES_TAG}" ]]; then
  EXPORT_ARGS+=(--images-tag "${OCR_IMAGES_TAG}")
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
if [[ -n "${CURATED_ROOT}" ]]; then
  EXPORT_ARGS+=(--curated-root "${CURATED_ROOT}" --curated-file-name "${CURATED_FILE_NAME}")
fi
if [[ -n "${INPUT_VERSION}" ]]; then
  EXPORT_ARGS+=(--input-version "${INPUT_VERSION}")
fi
if [[ -n "${OCR_ENGINE_VERSION}" ]]; then
  EXPORT_ARGS+=(--ocr-engine-version "${OCR_ENGINE_VERSION}")
fi
if [[ -n "${OCR_OUTPUT_VERSION}" ]]; then
  EXPORT_ARGS+=(--ocr-output-version "${OCR_OUTPUT_VERSION}")
fi
if [[ -n "${OCR_CURATED_VERSION}" ]]; then
  EXPORT_ARGS+=(--ocr-curated-version "${OCR_CURATED_VERSION}")
fi
if [[ -n "${RAG_INDEX_VERSION}" ]]; then
  EXPORT_ARGS+=(--rag-index-version "${RAG_INDEX_VERSION}")
fi

python "${EXPORT_ARGS[@]}"

if [[ "${STRICT_CURATED}" == "1" ]]; then
  python - "${MANIFEST_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.exists():
    print(f"[ERROR] STRICT_CURATED=1 but manifest not found: {manifest_path}")
    raise SystemExit(1)

raw_count = 0
rows = 0
with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows += 1
        obj = json.loads(line)
        if str(obj.get("table_source", "")).strip().lower() == "raw":
            raw_count += 1

if rows == 0:
    print(f"[ERROR] STRICT_CURATED=1 but manifest is empty: {manifest_path}")
    raise SystemExit(1)
if raw_count > 0:
    print(f"[ERROR] STRICT_CURATED=1 failed: table_source=raw rows={raw_count}/{rows}")
    raise SystemExit(1)
print(f"[OK] STRICT_CURATED=1 passed: all rows use curated source ({rows}/{rows})")
PY
fi

STAGE_END_TS="$(date +%s)"
STAGE_ELAPSED_SEC="$((STAGE_END_TS - STAGE_START_TS))"
STAGE_TOTAL_LATENCY_MS="$((STAGE_ELAPSED_SEC * 1000))"
STAGE_TOTAL_LATENCY_HMS="$(format_elapsed "${STAGE_ELAPSED_SEC}")"

ENGINE_SUMMARY_DIR="${OCR_OUTPUT_ROOT}/${OCR_ENGINE}"
if [[ -n "${OCR_IMAGES_TAG}" ]]; then
  ENGINE_SUMMARY_DIR="${ENGINE_SUMMARY_DIR}/${OCR_IMAGES_TAG}"
fi
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
8. images_root: ${IMAGES_ROOT}
9. images_tag: ${OCR_IMAGES_TAG:-<none>}
10. manifest: ${MANIFEST_OUTPUT}
11. chunks: ${CHUNKS_OUTPUT}
12. input_version: ${INPUT_VERSION:-<none>}
13. ocr_engine_version: ${OCR_ENGINE_VERSION:-<none>}
14. ocr_output_version: ${OCR_OUTPUT_VERSION:-<none>}
15. ocr_curated_version: ${OCR_CURATED_VERSION:-<none>}
16. rag_index_version: ${RAG_INDEX_VERSION:-<none>}
17. total_latency_ms: ${STAGE_TOTAL_LATENCY_MS}
18. total_latency_hms: ${STAGE_TOTAL_LATENCY_HMS}
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
8. images_root: ${IMAGES_ROOT}
9. images_tag: ${OCR_IMAGES_TAG:-<none>}
10. manifest: ${MANIFEST_OUTPUT}
11. chunks: ${CHUNKS_OUTPUT}
12. input_version: ${INPUT_VERSION:-<none>}
13. ocr_engine_version: ${OCR_ENGINE_VERSION:-<none>}
14. ocr_output_version: ${OCR_OUTPUT_VERSION:-<none>}
15. ocr_curated_version: ${OCR_CURATED_VERSION:-<none>}
16. rag_index_version: ${RAG_INDEX_VERSION:-<none>}
17. total_latency_ms: ${STAGE_TOTAL_LATENCY_MS}
18. total_latency_hms: ${STAGE_TOTAL_LATENCY_HMS}
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
  "images_root": "${IMAGES_ROOT}",
  "images_tag": "${OCR_IMAGES_TAG}",
  "manifest": "${MANIFEST_OUTPUT}",
  "chunks": "${CHUNKS_OUTPUT}",
  "input_version": "${INPUT_VERSION}",
  "ocr_engine_version": "${OCR_ENGINE_VERSION}",
  "ocr_output_version": "${OCR_OUTPUT_VERSION}",
  "ocr_curated_version": "${OCR_CURATED_VERSION}",
  "rag_index_version": "${RAG_INDEX_VERSION}",
  "total_latency_ms": ${STAGE_TOTAL_LATENCY_MS},
  "total_latency_hms": "${STAGE_TOTAL_LATENCY_HMS}"
}
EOF

echo "12. output"
echo "12-1. saved_ocr_stage_summary_json: ${STAGE_SUMMARY_JSON}"
echo "12-2. saved_ocr_stage_summary_txt: ${STAGE_SUMMARY_TXT}"
