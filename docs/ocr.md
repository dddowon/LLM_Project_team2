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

- 이미지 입력: `data/v2/ocr_images/<doc_key>/img_001.jpg`
- GT 입력: `data/v2/ocr_outputs/incoming_gt/<doc_key>.jsonl` (없으면 `.json` fallback)
- 주요 출력:
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/inference/*`
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/eval/*` (GT 모드)
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
