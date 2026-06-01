# 입찰메이트 RAG 시스템

공공입찰 RFP 문서를 대상으로 질의응답을 수행하는 시나리오 B(LLM API 기반) RAG baseline입니다.

## 목표

- PDF/HWP RFP 문서와 `data_list.csv` 메타데이터를 불러옵니다.
- 문서를 청킹하고 `text-embedding-3-small`로 임베딩합니다.
- Chroma vector DB를 생성합니다.
- 검색된 문서 근거를 바탕으로 `gpt-5-mini`가 답변합니다.
- 평가 harness의 LLM judge·질문 자동생성(`--call-openai`)도 기본 `gpt-5-mini`입니다.
- 평가 질문셋으로 검색/생성 결과를 반복 비교합니다.

## 기술 스택

- LLM API: OpenAI `gpt-5-mini`
- Embedding: OpenAI `text-embedding-3-small`
- Vector DB: Chroma `hnsw`
- Similarity: Chroma cosine distance 기반 검색

## 프로젝트 구조

```text
.
├── configs/                 # 실험 설정
│   └── default.yaml
├── data/
│   ├── raw/                 # 원본 데이터(PDF/HWP + data_list.csv), git 업로드 금지
│   ├── v1/                  # Hugging Face 시나리오 데이터/산출물
│   └── v2/                  # OpenAI 시나리오 데이터/산출물
├── checkpoints/             # Chroma 인덱스, git 업로드 금지
├── outputs/                 # 평가 결과, git 업로드 금지
├── src/
│   ├── dataset/             # 문서/메타데이터 로더
│   ├── preprocessing/       # 텍스트 정규화 및 청킹
│   ├── models/              # OpenAI API 래퍼
│   ├── engine/              # vector store, prompt, RAG engine
│   └── cli.py               # 실행 진입점
└── tests/
```

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
```

`.env`에 팀 OpenAI API key를 입력합니다.

```bash
OPENAI_API_KEY=...
```

원본 파일은 외부 공유 금지 대상이므로 git에 올리지 않습니다.

```text
data/raw/문서파일.pdf
data/raw/문서파일.hwp
data/raw/data_list.csv
```

현재 세팅이 잘 잡혔는지 먼저 확인할 수 있습니다.

```bash
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli check-setup
```

API key와 네트워크 연결까지 실제로 확인하려면 아래처럼 실행합니다.

```bash
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli check-setup --check-openai
```

## 실행

인덱스 생성:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli ingest
```

질의:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli query "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘."
```

평가:

```bash
# 기본 설정은 data/v2/eval_questions.example.jsonl 을 사용합니다.
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli evaluate
```

LangSmith 하네스 평가(검색·LLM judge·트레이스):

```bash
# .env: OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT=bidmate-rag-eval
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli evaluate-harness --config configs/default.yaml
```

결과 JSONL은 `outputs/eval_harness_results.jsonl`, 트레이스는 [https://smith.langchain.com](https://smith.langchain.com) 프로젝트에서 확인합니다.

HWP 파일을 처리해야 한다면 VM에서 선택 의존성을 추가로 설치합니다.

```bash
pip install -e ".[hwp]"
```

## 운영 파이프라인 (HWP + OCR 전체)

**RFP 100건**(`data/raw/RFP_file_100`) 기준으로 **HWP/PDF 본문 + 문서 내 이미지(OCR)** 를 RAG·평가에 넣는 표준 순서입니다.

- RAG/평가: `~/llm_team2` (또는 팀 venv)
- OCR 추론: `conda activate ocr_vl15` ([OCR 파이프라인](#ocr-파이프라인) 절, `requirements_ocr_vl15.txt`)

**새 터미널마다** 프로젝트 루트에서:

```bash
cd ~/LLM_Project_team2
source ~/llm_team2/bin/activate
export PYTHONPYCACHEPREFIX=.cache/pycache
```

### 0) 환경·폴더·원본

```bash
cd ~/LLM_Project_team2
pip install -e ".[dev,hwp]"   # 최초 1회
cp .env.example .env         # OPENAI_API_KEY 등

mkdir -p outputs logs checkpoints data/raw data/v2 \
  data/v2/ocr_images data/v2/ocr_outputs data/v2/ocr_rag \
  .cache/pycache
```

원본 zip (`unzip` 없으면 Python):

```bash
python - <<'PY'
import zipfile
from pathlib import Path
z = Path("data/raw/RFP_file_100.zip")
d = Path("data/raw/RFP_file_100")
d.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(z) as f:
    f.extractall(d)
print("done:", d)
PY
```

이미지 입력 규칙: `data/v2/ocr_images/<doc_key>/img_001.jpg` …  
`<doc_key>`는 HWP 산출 폴더명(`data/v2/<sanitize된_파일명>/`)과 **동일**하게 맞춥니다.

### 1) 텍스트 파이프라인 (`llm_team2`)

```bash
source ~/llm_team2/bin/activate

python -m src.cli run-pipeline \
  --input-dir "data/raw/RFP_file_100" \
  --output-dir "data/v2" \
  --recursive \
  --force-real
```

### 2) OCR — GT 없이 추론 + chunk export (`ocr_vl15`)

```bash
conda activate ocr_vl15
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}   # WSL + NVIDIA GPU일 때만

OCR_USE_GT=0 ./scripts/run_ocr_stage.sh

wc -l data/v2/ocr_rag/ocr_input_chunks.jsonl
```

- `OCR_USE_GT=0` → GT 없이 `inference/*` 생성 + `ocr-export-rag --allow-inference-only`
- 산출: `data/v2/ocr_outputs/.../inference/*`, `data/v2/ocr_rag/ocr_input_chunks.jsonl`

### 2.5) OCR 임베딩 + OCR 전용 Chroma (`llm_team2`)

```bash
source ~/llm_team2/bin/activate

OCR_BASE="data/v2/ocr_rag/paddleocr_vl/v4_table_filtered_260531"

python -m src.cli embed-jsonl \
  --input "${OCR_BASE}/ocr_input_chunks.jsonl" \
  --output "${OCR_BASE}/ocr_input_embedded.jsonl" \
  --force-real
python -m src.cli build-chroma \
  --input "${OCR_BASE}/ocr_input_embedded.jsonl" \
  --index-dir "${OCR_BASE}/chroma_index"
```

### 3) 통합 Chroma (`llm_team2`) — query·eval 공용

```bash
python -m src.cli --config configs/default.yaml merge-embedded \
  --input-dir data/v2 \
  --pattern "*embedded*.jsonl" \
  --index-dir checkpoints/chroma_openai
```

HWP `*_embedded.jsonl`과 `ocr_input_embedded.jsonl`이 함께 병합됩니다.  
eval/query는 `**checkpoints/chroma_openai`만** 사용합니다 (`ocr_rag/chroma_index` 아님).

### 4) 질의 테스트

```bash
python -m src.cli --config configs/default.yaml query \
  "BIFF 온라인서비스 재개발 사업의 주요 과업 범위는 뭐야?"
```

### 6) 평가 질문 입력 (`llm_team2`)

`sampling` 없이 `data/v2` 하위 `*_chunks.jsonl`과 `ocr_input_chunks.jsonl`을 함께 읽습니다.

```bash
python -m src.evaluation.generate_eval_questions \
  --input-dir data/v2 \
  --pattern "mixed_chunks_slim.jsonl" \
  --extra-chunk-file "data/v2/ocr_rag/paddleocr_vl/v4_table_filtered_260531/ocr_input_chunks.jsonl" \
  --max-docs 100 \
  --max-chunks-per-doc 12 \
  --questions-per-doc 5 \
  --output data/v2/eval_question_generation_inputs.jsonl \
  --overwrite
```

### 7) OpenAI 질문셋

```bash
python -m src.evaluation.generate_eval_questions \
  --call-openai \
  --generation-input data/v2/eval_question_generation_inputs.jsonl \
  --eval-output data/v2/eval_questions.jsonl \
  --model gpt-5-mini \
  --overwrite
```

### 8) harness

```bash
python -m src.cli --config configs/default.yaml evaluate-harness \
  --output outputs/eval_harness_results.jsonl \
  --judge-model gpt-5-mini \
  --no-langsmith-feedback
```

`--no-langsmith-feedback`: 로컬 jsonl·리포트에는 영향 없음. LangSmith feedback API에 `total_latency_ms` 등을 보낼 때 score 범위 초과(422)를 피하기 위함. trace는 별도 설정 시 남을 수 있음.

### 9) 리포트 (점수·오답노트)

```bash
python -m src.evaluation.build_eval_report \
  --input outputs/eval_harness_results.jsonl \
  --html-output outputs/eval_report.html \
  --failures-output outputs/eval_failures.csv \
  --successes-output outputs/eval_successes.csv \
  --top-n 500
```

리포트 HTML에 `eval_focus`(본문 vs OCR 이미지) 구간이 포함됩니다. **점수·오답 분석은 이 HTML/CSV가 본체**입니다.

긴 단계는 `nohup ... > logs/단계명.log 2>&1 &` 로 백그라운드 실행을 권장합니다. 디스크 여유 **10GB+** 권장 (`df -h`).

## HWP 파싱/청킹/임베딩 파이프라인

### 원클릭 실행(권장)

```bash
python -m src.cli run-pipeline \
  --input-dir "data/raw/폴더 이름" \
  --output-dir "data/v2" \
  --index-dir "checkpoints/chroma_openai" \
  --force-real
```

메타데이터 샘플 저장이 필요한 경우:

```bash
python -m src.cli run-pipeline \
  --input "data/raw/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시.hwp" \
  --output-dir "data/v2" \
  --dump-metadata-sample \
  --dump-limit 20
```

기본 동작:

- 출력은 `data/v2/<sanitize된_파일명>/` 하위로 문서별 분리 저장됩니다.
- `--doc-id`는 선택 옵션이며, 폴더명에는 영향이 없고 Chroma 메타데이터의 `doc_id`에만 반영됩니다.
- 기본값으로 Chroma 메타데이터 샘플 JSON은 저장하지 않습니다(운영 권장).
- 필요할 때만 `--dump-metadata-sample` 옵션으로 저장합니다.
  - 파일명: `<원본파일명>_chroma_metadata_sample.json`
  - 개수: 기본 20개 (`--dump-limit`으로 변경)

실행 단계:

1. `parse-hwp`
2. `chunk-jsonl`
3. `embed-jsonl`
4. `build-chroma`

생성 산출물:

- `<doc_dir>/<원본파일명>_prechunk.jsonl`
- `<doc_dir>/<원본파일명>_heading_debug.jsonl`
- `<doc_dir>/<원본파일명>_chunks.jsonl`
- `<doc_dir>/<원본파일명>_chunks_summary.csv`
- `<doc_dir>/<원본파일명>_chunks_sample.jsonl`
- `<doc_dir>/<원본파일명>_embedded.jsonl`
- `<doc_dir>/<원본파일명>_chroma_metadata_sample.json`
- `<doc_dir>/chroma_index/chroma.sqlite3`
- `<doc_dir>/chroma_index/chunks.json`

여기서 `<doc_dir>`는 `data/v2/<sanitize된_파일명>/`입니다.

### 단계별 실행

디버깅 또는 중간 산출물(prechunk/chunks/embedded) 확인이 필요할 때만 사용합니다. 일반 실행은 원클릭 `run-pipeline`을 권장합니다.

- 원본 HWP: `data/raw/`
- 산출 경로: `run-pipeline`과 동일 (`data/v2/<sanitize된_파일명>/` 아래, 파일명은 원본 stem 유지)

```bash
python -m src.cli parse-hwp \
  --input "data/raw/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시.hwp" \
  --output "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_prechunk.jsonl" \
  --debug-headings "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_heading_debug.jsonl"

python -m src.cli chunk-jsonl \
  --input "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_prechunk.jsonl" \
  --output "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_chunks.jsonl"

python -m src.cli embed-jsonl \
  --input "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_chunks.jsonl" \
  --output "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_embedded.jsonl" \
  --model "text-embedding-3-small" \
  --force-real

python -m src.cli build-chroma \
  --input "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_embedded.jsonl" \
  --index-dir "checkpoints/chroma_openai"
```

`build-chroma`의 `--index-dir`은 원클릭 실행과 같이 팀 공용 인덱스(`configs/default.yaml`의 `paths.index_dir`)를 쓰는 것을 권장합니다. 문서 폴더 안에만 두려면 `data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/chroma_index`처럼 지정할 수 있습니다.

### 통합 인덱스(여러 문서 embedded 병합)

`run-pipeline --input-dir`로 문서별 `*_embedded.jsonl`을 만든 뒤, **eval/query가 쓰는 통합 Chroma**는 아래 명령으로 한 번에 만듭니다.
(`--index-dir`로 파일마다 build-chroma를 반복하면 마지막 문서만 남습니다.)

```bash
python -m src.cli merge-embedded \
  --config configs/default.yaml \
  --input-dir data/v2 \
  --index-dir checkpoints/chroma_openai
```

- 병합 JSONL 기본 경로: `checkpoints/all_embedded.jsonl`
- OCR 포함 전체 순서는 위 [운영 파이프라인 (HWP + OCR 전체)](#운영-파이프라인-hwp--ocr-전체) 참고.
- `ocr_input_embedded.jsonl`은 `data/v2` 재귀 `*_embedded.jsonl` 검색에 포함됩니다.
- `ocr_rag/chroma_index/`는 중간 산출물이며 eval/query 인덱스가 **아닙니다**.

### `embed-jsonl` 입력 포맷(고정)

`embed-jsonl`은 아래 청킹 포맷만 입력으로 받습니다.

- `chunk_id`
- `chunk_type`
- `chunk_text`
- `metadata` (object)

출력은 입력 row를 유지하고 `embedding`, `metadata.embedding_source`를 추가합니다.
`metadata`는 필수이며 `dict` 타입이어야 합니다. 누락되거나 타입이 다르면 `ValueError`를 발생시킵니다.

`OPENAI_API_KEY`가 없으면 `mock` 임베딩으로 동작하고, 실 API 강제 검증은 `--force-real` 옵션을 사용합니다.

## 실험 포인트

- 청킹: `chunk_size`, `chunk_overlap`, 목차/장절 기반 의미 청킹 비교
- 검색: `top_k`, 메타데이터 필터링, MMR, hybrid search, re-ranking 비교
- 생성: 프롬프트, 답변 포맷, 답변 길이, 대화 히스토리 반영 방식 비교
- 평가: 단일 문서 정확도, 다중 문서 종합, 후속 질문 맥락 유지, 모르는 내용 거절 여부

## 참고한 공식 문서

- OpenAI Embeddings guide: [https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
- OpenAI Responses API: [https://platform.openai.com/docs/api-reference/responses/create](https://platform.openai.com/docs/api-reference/responses/create)

## 파싱/청킹/샘플링 단독 실행

임베딩·Chroma 인덱싱 없이 **파싱 → 청킹 → (선택) 평가용 샘플링**만 실행합니다. 경로 규칙은 위 단계별 실행과 동일합니다.

파싱:

```bash
python -m src.cli parse-hwp \
  --input "data/raw/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시.hwp" \
  --output "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_prechunk.jsonl" \
  --debug-headings "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_heading_debug.jsonl"
```

청킹 (`chunks_summary.csv`, `chunks_sample.jsonl`은 옵션으로 함께 저장):

```bash
python -m src.cli chunk-jsonl \
  --input "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_prechunk.jsonl" \
  --output "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_chunks.jsonl"
```

`chunks_sample.jsonl`은 문서당 청크 미리보기(기본 20개)이며, 아래 `eval_sample_chunks.jsonl`과는 다릅니다.

평가용 샘플링 — 문서 1개:

```bash
python -m src.cli sampling \
  --input "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_chunks.jsonl" \
  --output "data/v2/eval_sample_chunks.jsonl"
```

`run-pipeline`으로 여러 문서를 처리한 뒤 일괄 샘플링:

```bash
python -m src.cli sampling \
  --input-dir "data/v2" \
  --pattern "*_chunks.jsonl" \
  --output "data/v2/eval_sample_chunks.jsonl"
```

기준 청킹 결과와 overlap을 맞춰야 하는 경우 `chunk-jsonl`에 `--text-overlap 150`을 추가합니다.

## OCR 파이프라인

### OCR 환경 설치

```bash
conda create -n ocr_vl15 python=3.10.20 -y
conda activate ocr_vl15
pip install -r requirements_ocr_vl15.txt
python -m src.cli check-ocr3-setup
```

`check-ocr3-setup`은 패키지 설치/버전과 기본 import 상태를 검증합니다.
PPStructureV3, PaddleOCRVL, table_recognition_v2는 런타임 모델 다운로드가 필요하므로
최종 검증은 실제 OCR 추론 스모크 테스트까지 통과해야 완료입니다.

재현 가능한 환경 고정을 위해 설치 직후 아래 파일을 커밋합니다.

```bash
conda env export -n ocr_vl15 > envs/ocr_vl15.lock.yml
pip freeze > envs/ocr_vl15.freeze.txt
```

현재 `requirements_ocr_vl15.txt`에는 `paddlepaddle-gpu`가 포함되어 있습니다.

### 개요

현재 OCR 실행은 CLI에서 아래 명령으로 통합 운영합니다.

```bash
# WSL + NVIDIA GPU 환경에서만 필요
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
```

입력/출력 규칙:

- 이미지 입력: `data/v2/ocr_images/<doc_key>/img_001.jpg`
- GT 입력: `data/v2/ocr_outputs/incoming_gt/<doc_key>.jsonl` (없으면 `.json` fallback)
- 결과 출력:
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/inference/pred_raw.json`
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/inference/pred_table_raw.html`
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/inference/pred_table_layout.json`
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/inference/pred_table_layout.html`
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/eval/gt_pred_structured.json`
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/eval/gt_eval_summary.json`
  - `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/eval/gt_eval_debug.json` (review_required=true일 때만 생성)
- `<engine>`: `pp_ocrv5`, `pp_ocrv5_transformers`, `pp_structurev3`, `table_recognition_v2`, `paddleocr_vl`
- `--doc-key`는 파일명이 아니라 `ocr_images` 하위 폴더명입니다.

GT 의존성:

- 기본 모드에서는 GT가 있어야 `eval/*`까지 생성됩니다.
- `--no-gt`(또는 `OCR_NO_GT=1`)를 쓰면 GT 없이 `inference/*`만 생성하는 추론 전용 모드로 동작합니다.

Threshold 의미:

- `--score-threshold`:
  - OCR 예측 생성 단계의 confidence 하한값입니다.
  - 이 값보다 낮은 인식 결과는 `eval/gt_pred_structured.json` 구성에서 제외됩니다.
  - 즉, **예측값 자체를 필터링**하는 파라미터입니다.
- `--structure-threshold`:
  - 평가 단계에서 GT 값과 pred 값을 매칭할 때 쓰는 유사도 임계값입니다.
  - 이 값은 `eval/gt_eval_summary.json`의 구조 지표(`structure_micro_recall`, `structure_macro_f1`)에 영향을 줍니다.
  - 즉, **점수 계산 기준을 조정**하는 파라미터입니다.
- 둘은 역할이 다르므로 같은 값으로 고정할 필요는 없습니다.

### OCR 입력 이미지 추출 (`extract-ocr-images`)

HWP/PDF를 한 번에 처리해서 `data/v2/ocr_images/<doc_key>/img_XXX.<ext>` 구조로 저장합니다.

```bash
python -m src.cli extract-ocr-images \
  --input-dir "data/raw" \
  --output-dir "data/v2/ocr_images" \
  --source-type all
```

PDF에서 불필요한 작은 이미지를 줄이려면 필터 임계값을 조정합니다.

```bash
python -m src.cli extract-ocr-images \
  --input-dir "data/raw" \
  --source-type pdf \
  --pdf-min-width 100 \
  --pdf-min-height 40 \
  --pdf-min-area 10000 \
  --pdf-min-bytes 1000
```

### 이미지 1개 실행 (`ocr-run-image`)

```bash
python -m src.cli ocr-run-image \
  --doc-key "한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" \
  --image-name "img_001.jpg" \
  --ocr-config "configs/ocr_default.yaml" \
  --score-threshold 0.0 \
  --structure-threshold 0.65
```

경로는 `--ocr-config`의 `paths` 섹션에서만 관리합니다(스크립트/CLI 경로 오버라이드 미사용).

예시:

```bash
cd /home/imella0707/personal/LLM_Project_team2
conda activate ocr_vl15
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}   # WSL + NVIDIA GPU일 때만

sed -n '1,80p' /home/imella0707/personal/LLM_Project_team2/configs/ocr_default.yaml
```

GT JSON에 `id`가 여러 개면 `--id`를 함께 지정하세요.

### GT 없이 추론만 실행 (`--no-gt`)

```bash
python -m src.cli ocr-run-image \
  --doc-key "한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" \
  --image-name "img_001.jpg" \
  --ocr-config "configs/ocr_default.yaml" \
  --no-gt
```

### 문서 폴더 1개 실행 (`ocr-run-batch --doc-key`)

```bash
python -m src.cli ocr-run-batch \
  --doc-key "한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" \
  --ocr-config "configs/ocr_default.yaml" \
  --score-threshold 0.0 \
  --structure-threshold 0.65
```

GT 없이 문서 폴더 1개 추론만:

```bash
python -m src.cli ocr-run-batch \
  --doc-key "한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" \
  --ocr-config "configs/ocr_default.yaml" \
  --no-gt
```

### 전체 문서 배치 실행 (`ocr-run-batch`)

```bash
python -m src.cli ocr-run-batch \
  --ocr-config "configs/ocr_default.yaml" \
  --score-threshold 0.0 \
  --structure-threshold 0.65
```

5개 ocr엔진을 한 번에 자동 실행하려면:

```bash
python -m src.cli ocr-run-batch \
  --ocr-config "configs/ocr_default.yaml" \
  --score-threshold 0.0 \
  --structure-threshold 0.65 \
  --all-engines
```

경로 변경은 `configs/ocr_default.yaml`의 `paths`만 수정하세요:

```bash
cd /home/imella0707/personal/LLM_Project_team2
conda activate ocr_vl15
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}   # WSL + NVIDIA GPU일 때만

python -m src.cli ocr-run-batch \
  --ocr-config "/home/imella0707/personal/LLM_Project_team2/configs/ocr_default.yaml" \
  --score-threshold 0.0 \
  --structure-threshold 0.65
```

`configs/ocr_default.yaml` 예시:

```yaml
ocr:
  engine: paddleocr_vl
  device: gpu:0
  lang: korean
  score_threshold: 0.0
  structure_match_threshold: 0.65
  batch_size: 1

paths:
  images_root: data/v2/ocr_images/v4_table_filtered_260531
  gt_root: data/v2/ocr_outputs/incoming_gt
  output_root: data/v2/ocr_outputs
```

### OCR → RAG Stage 스크립트 (권장)

용어 정리 (`OCR→RAG handoff`):

- `ocr_input_manifest.jsonl`: OCR 이미지 단위 메타데이터 목록(문서키, 이미지 스템, 타입, 경로, 리뷰 플래그 등)
- `ocr_input_chunks.jsonl`: RAG 임베딩 대상으로 넘기는 실제 텍스트 청크 목록

개발/운영은 아래 3개 엔트리포인트로 분리합니다.
OCR과 RAG는 의존성 충돌 방지를 위해 가상환경을 분리해서 실행합니다.

```bash
# OCR만 실행 (OCR env: 기본 ocr_vl15)
./scripts/run_ocr_stage.sh

# RAG만 실행 (RAG env: 기본 llm_team2)
./scripts/run_rag_stage.sh

# OCR -> RAG 연속 실행
./scripts/run_all_ocr_rag_pipeline.sh
```

기본 동작(중요):

- `run_ocr_stage.sh` 기본값은 GT 없이 추론 전용 모드입니다.
  - 기본값: `OCR_USE_GT=0`
- OCR 데이터 경로(`images_root`, `gt_root`, `output_root`)는 `configs/ocr_default.yaml`의 `paths`에서만 읽습니다.
- GT 없이 추론 전용으로 실행하려면 `OCR_USE_GT=0`으로 실행하세요.
  - 이 모드에서도 `ocr-export-rag`는 `--allow-inference-only`로 실행되어 handoff 청크를 생성합니다.
  - 기본값: `EXCLUDE_REVIEW_REQUIRED=0`
  - `EXCLUDE_REVIEW_REQUIRED=1`일 때만 `review_required=true` 항목을 제외합니다.
- `USE_DOC_UNWARPING=0`이 기본값입니다.
- `INCLUDE_HTML_CHUNK=0`이 기본값입니다. 즉 HTML 스니펫은 RAG 청크에 기본 포함되지 않습니다.

선택한 일부 폴더만 OCR 수행 후 RAG handoff 파일로 내보내기:

```bash
# 예시 1) 특정 doc_key 1개만 처리
DOC_KEY="인천광역시_도시계획위원회 통합관리시스템 구축용역" \
./scripts/run_ocr_stage.sh

# 예시 2) 다른 doc_key로 재실행
DOC_KEY="한국생산기술연구원_EIP3.0 고압가스 안전관리 시스템 구축 용역" \
./scripts/run_ocr_stage.sh
```

여러 폴더를 연속 처리하려면 `DOC_KEY`를 바꿔 반복 실행하거나, 전체 처리(`DOC_KEY` 미지정)를 사용합니다.

문서 왜곡 보정을 끄고 비교 실행하려면:

```bash
USE_DOC_UNWARPING=0 ./scripts/run_ocr_stage.sh
```

품질 게이트 통과건만 RAG handoff에 포함하려면:

```bash
EXCLUDE_REVIEW_REQUIRED=1 ./scripts/run_ocr_stage.sh
```

HTML 스니펫까지 RAG 청크에 포함하려면(기본 비권장):

```bash
INCLUDE_HTML_CHUNK=1 HTML_CHUNK_MAX_CHARS=1200 ./scripts/run_ocr_stage.sh
```

`run_ocr_stage.sh` 산출물:

- 공통 (`OCR_USE_GT=0` / `OCR_USE_GT=1` 모두 생성)
  - `data/v2/ocr_rag/<engine>/<images_tag>/ocr_input_manifest.jsonl`
  - `data/v2/ocr_rag/<engine>/<images_tag>/ocr_input_chunks.jsonl`
  - `paths.images_root`가 `.../ocr_images` 루트면 `images_tag` 없이 `data/v2/ocr_rag/<engine>/...`로 생성
- `OCR_USE_GT=0` 추론 산출 경로
  - `paths.images_root`가 `.../ocr_images/<images_tag>`일 때:
    `data/v2/ocr_outputs/<engine>/<images_tag>/<doc_key>/<image_stem>/inference/*`
  - `paths.images_root`가 `.../ocr_images` 루트일 때(태그 없음):
    `data/v2/ocr_outputs/<engine>/<doc_key>/<image_stem>/inference/*`

- `OCR_USE_GT=1` GT 평가 산출 경로
  - `OCR_USE_GT=0` 경로에 더해 `eval/*`(예: `gt_eval_summary.json`, `gt_pred_structured.json`) 생성

GT 모드 + RAG handoff까지 포함하려면:

```bash
OCR_USE_GT=1 ./scripts/run_ocr_stage.sh
```

GT 없이 Stage-1 추론만 실행하려면:

```bash
OCR_USE_GT=0 DOC_KEY="한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" ./scripts/run_ocr_stage.sh
```

- `OCR_USE_GT=0`일 때는 GT 기반 `eval/*` 없이 추론/청크 생성을 수행합니다.

`run_all_ocr_rag_pipeline.sh`에서도 동일:

```bash
OCR_USE_GT=0 RUN_RAG_STAGE=1 DOC_KEY="한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" ./scripts/run_all_ocr_rag_pipeline.sh
```

- `OCR_USE_GT=0`이어도 `RUN_RAG_STAGE=1`이면 RAG stage를 실행합니다.

#### 경희대학교 문서 1건 테스트 예시

```bash
# GT 없이 OCR만 실행
DOC_KEY="경희대학교_[입찰공고] 산학협력단 정보시스템 운영 용역업체 선정" \
OCR_USE_GT=0 \
./scripts/run_ocr_stage.sh

# RAG만 실행 (run_rag_stage.sh는 DOC_KEY를 직접 사용하지 않음)
./scripts/run_rag_stage.sh

# OCR + RAG 전체 실행 (GT 없이)
DOC_KEY="경희대학교_[입찰공고] 산학협력단 정보시스템 운영 용역업체 선정" \
OCR_USE_GT=0 \
./scripts/run_all_ocr_rag_pipeline.sh

# GT 기반 OCR + RAG 전체 실행
DOC_KEY="경희대학교_[입찰공고] 산학협력단 정보시스템 운영 용역업체 선정" \
OCR_USE_GT=1 \
./scripts/run_all_ocr_rag_pipeline.sh
```

`run_rag_stage.sh` 기본 산출물:

- `ocr_default.yaml`의 `ocr.engine`, `paths.images_root` 기준:
  - `data/v2/ocr_rag/<engine>/<images_tag>/ocr_input_embedded.jsonl`
  - `data/v2/ocr_rag/<engine>/<images_tag>/chroma_index/`
- `paths.images_root`가 `.../ocr_images` 루트면 `images_tag` 없이 `data/v2/ocr_rag/<engine>/...`로 생성

eval/query에서 OCR을 검색하려면 위 embedded를 **통합 인덱스** 절의 `merge-embedded`로 `checkpoints/chroma_openai`에 병합하세요.

운영 원칙:

- 팀 간 OCR→RAG 인터페이스 파일은 `data/v2/ocr_rag/ocr_input_chunks.jsonl` 단일 파일로 고정합니다.
- `chroma_index/`는 RAG 임베딩/인덱싱 이후의 런타임 산출물이며 전달 표준 포맷이 아닙니다.
- `pred_table_layout.html`은 사람 검수용 참고 파일이며, 기본 RAG 입력 계약 포맷이 아닙니다.

### OCR→RAG handoff를 CLI로 직접 생성 (`ocr-export-rag`)

```bash
python -m src.cli ocr-export-rag \
  --ocr-eval-root "data/v2/ocr_outputs" \
  --engine "paddleocr_vl" \
  --images-tag "v4_table_filtered_260531" \
  --output-manifest "data/v2/ocr_rag/ocr_input_manifest.jsonl" \
  --output-chunks "data/v2/ocr_rag/ocr_input_chunks.jsonl"
```

- 같은 `doc_key`가 여러 버전 태그(`v1_*`, `v4_*`)에 공존하면 `--images-tag`를 지정해서 현재 실험 태그만 export하세요.
- `scripts/run_ocr_stage.sh`는 `ocr_default.yaml`의 `paths.images_root`에서 추출한 `images_tag`를 자동으로 `ocr-export-rag`에 전달합니다.

### 결과 확인 위치

- OCR 이미지 추출 결과: `data/v2/ocr_images/`
- OCR 산출물(`inference/*`) + OCR 평가 산출물(`eval/*`): `data/v2/ocr_outputs/`
- 엔진 단위 요약:
  - `data/v2/ocr_outputs/<engine>/ocr_eval_summary.{csv,json,txt}`
  - `data/v2/ocr_outputs/<engine>/review_queue.jsonl` (규칙 기반 실패 큐)
- `eval/gt_eval_summary.json` 주요 지표:
  - `schema_version`, `gt_path`, `pred_structured_path`
  - `review_required`, `review_reasons`
  - `type`, `status`, `latency_ms`
  - `text.char_similarity_pct`, `text.cer`, `text.wer`, `text.exact_match`
  - `structure_micro_recall`, `structure_macro_f1`
  - `structure.aggregate` (`matched`, `gt_total`, `pred_total`, `micro_precision`, `micro_recall`, `micro_f1`, `macro_f1`)
  - `table_html.exists`, `table_rows.exists`
