# OCR 파이프라인 상세

## OCR 환경 설치

```bash
conda create -n ocr_vl15 python=3.10.20 -y
conda activate ocr_vl15
pip install -r requirements_ocr_vl15.txt
python -m src.cli check-ocr3-setup
```

`check-ocr3-setup`은 기본 import/버전을 확인합니다. 최종 검증은 실제 OCR 추론 스모크 테스트까지 수행하세요.

## 입력/출력 규칙

- 이미지 입력: `data/v2/ocr_images/<images_tag>/<doc_key>/img_001.jpg`
- GT 입력: `data/v2/ocr_outputs/incoming_gt/<images_tag>/<doc_key>.jsonl` (없으면 `.json` fallback)
- 주요 출력:
  - `data/v2/ocr_outputs/<engine>/<images_tag>/<doc_key>/<image_stem>/inference/*`
  - `data/v2/ocr_outputs/<engine>/<images_tag>/<doc_key>/<image_stem>/eval/*` (GT 모드)
- 현재 기본 `<images_tag>`는 `configs/ocr_default.yaml`의 `paths.images_root` 기준 `v4_table_filtered_260531`
- `--doc-key`는 파일명이 아니라 `ocr_images` 하위 폴더명 기준

## 주요 실행 명령

### OCR 입력 이미지 추출

```bash
python -m src.cli extract-ocr-images \
  --input-dir "data/raw" \
  --output-dir "data/v2/ocr_images" \
  --source-type all
```

### 단일 이미지 OCR

```bash
python -m src.cli ocr-run-image \
  --doc-key "문서_키" \
  --image-name "img_001.jpg" \
  --ocr-config "configs/ocr_default.yaml" \
  --score-threshold 0.0 \
  --structure-threshold 0.65
```

GT 없이 추론만:

```bash
python -m src.cli ocr-run-image \
  --doc-key "문서_키" \
  --image-name "img_001.jpg" \
  --ocr-config "configs/ocr_default.yaml" \
  --no-gt
```

### OCR bbox / 전처리 입력 이미지 저장

단일 이미지 디버깅 시 bbox 시각화와 실제 OCR 입력 이미지를 저장할 수 있습니다.

```bash
python -m src.cli ocr-run-image \
  --doc-key "문서_키" \
  --image-name "img_001.jpg" \
  --ocr-config "configs/ocr_default.yaml" \
  --image-preprocess clahe \
  --clahe-clip-limit 1.5 \
  --clahe-tile-grid-size 8 \
  --save-bbox-image \
  --save-preprocessed-image \
  --no-gt
```

- `pred_bbox_<image_stem>.jpg`: OCR bbox 시각화 이미지
- `input_preprocessed.png`: resize/전처리 후 실제 OCR에 입력된 이미지

### OCR 전처리 sweep

GT가 없는 전처리 실험에서는 CER/WER 대신 OCR 검출 결과 기반 proxy metric을 사용합니다.

| 지표 | 목적 |
| --- | --- |
| `ocr_line_count` | 텍스트 누락/과검출 확인 |
| `raw_text_length` | 전체 추출 텍스트량 확인 |
| `table_row_count` | 표 행 구조 유지 여부 확인 |
| `layout_text_length` | 구조화 결과 텍스트량 확인 |
| `empty_cell_ratio` | 구조화 JSON의 빈 셀 증가 여부 확인 |
| `keyword_hit_count` | 핵심 필드 유지 여부 확인 |
| `latency_ms` | 전처리 비용 확인 |

Grayscale 단일 이미지 sweep:

```bash
python -m src.cli ocr-sweep-preprocess-image \
  --doc-key "문서_키" \
  --image-name "img_001.jpg" \
  --preprocess grayscale \
  --strengths 0.0 0.4 \
  --save-preprocessed-image
```

CLAHE 단일 이미지 sweep:

```bash
python -m src.cli ocr-sweep-preprocess-image \
  --doc-key "문서_키" \
  --image-name "img_001.jpg" \
  --preprocess clahe \
  --clahe-clip-limits 0.0 1.5 \
  --clahe-tile-grid-size 8 \
  --save-preprocessed-image
```

문서 batch sweep:

```bash
python -m src.cli ocr-sweep-preprocess-batch \
  --preprocess clahe \
  --clahe-clip-limits 0.0 1.5 \
  --clahe-tile-grid-size 8 \
  --limit-docs 10 \
  --save-preprocessed-image
```

주요 출력:

- 단일 이미지: `<image_stem>_<preprocess>_sweep/inference/<preprocess>_sweep_summary.csv`
- 전체 요약: `data/v2/ocr_outputs/<engine>/<images_tag>/<preprocess>_sweep_summary_all.csv`
- variant별 OCR 결과: `<image_stem>_grayscale_0p4/inference/*`, `<image_stem>_clahe_c1p5_t8/inference/*`
- variant별 입력 이미지: `<variant>/inference/input_preprocessed.png`

### 문서/전체 배치 OCR

```bash
python -m src.cli ocr-run-batch --doc-key "문서_키" --ocr-config "configs/ocr_default.yaml"
python -m src.cli ocr-run-batch --ocr-config "configs/ocr_default.yaml" --all-engines
```

## OCR -> RAG Stage 스크립트

권장 엔트리포인트:

```bash
# OCR만
./scripts/run_ocr_stage.sh

# RAG 임베딩/인덱싱만
./scripts/run_rag_stage.sh

# OCR -> RAG 연속 실행
./scripts/run_all_ocr_rag_pipeline.sh
```

기본 동작:

- `run_ocr_stage.sh` 기본값은 `OCR_USE_GT=0` (추론 전용)
- 경로는 `configs/ocr_default.yaml`의 `paths`만 사용
- `EXCLUDE_REVIEW_REQUIRED=1`이면 품질 게이트 통과건만 handoff 포함

예시:

```bash
DOC_KEY="인천광역시_도시계획위원회 통합관리시스템 구축용역" ./scripts/run_ocr_stage.sh
OCR_USE_GT=0 RUN_RAG_STAGE=1 DOC_KEY="문서_키" ./scripts/run_all_ocr_rag_pipeline.sh
```

## Curated 반영 릴리즈

```bash
OCR_OUTPUT_VERSION=v4_table_filtered_260531 \
OCR_CURATED_VERSION=v4_curated_20260601_1542 \
./scripts/run_curated_rag_stage.sh
```

`STRICT_CURATED=1` 게이트로 raw fallback 유입을 차단합니다.

## OCR->RAG handoff를 CLI로 직접 생성

```bash
python -m src.cli ocr-export-rag \
  --ocr-eval-root "data/v2/ocr_outputs" \
  --engine "paddleocr_vl" \
  --images-tag "v4_table_filtered_260531" \
  --output-manifest "data/v2/ocr_rag/paddleocr_vl/v4_table_filtered_260531/ocr_input_manifest.jsonl" \
  --output-chunks "data/v2/ocr_rag/paddleocr_vl/v4_table_filtered_260531/ocr_input_chunks.jsonl"
```

병합 유닛 반영이 필요하면 `--use-merge-manifest`를 curated 릴리즈에서만 사용하세요.
