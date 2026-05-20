# 입찰메이트 RAG 시스템

공공입찰 RFP 문서를 대상으로 질의응답을 수행하는 시나리오 B(LLM API 기반) RAG baseline입니다.

## 목표

- PDF/HWP RFP 문서와 `data_list.csv` 메타데이터를 불러옵니다.
- 문서를 청킹하고 `text-embedding-3-small`로 임베딩합니다.
- Chroma vector DB를 생성합니다.
- 검색된 문서 근거를 바탕으로 `gpt-5-mini`가 답변합니다.
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

결과 JSONL은 `outputs/eval_harness_results.jsonl`, 트레이스는 https://smith.langchain.com 프로젝트에서 확인합니다.
MLflow 서버 없이 동작합니다. GCP/MLflow는 `pip install -e ".[mlflow]"` 후 `evaluate-mlflow`로 나중에 사용할 수 있습니다.

HWP 파일을 처리해야 한다면 VM에서 선택 의존성을 추가로 설치합니다.

```bash
pip install -e ".[hwp]"
```

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
conda create -n ocr_eval_vlm python=3.10.20 -y
conda activate ocr_eval_vlm
pip install -r requirements_ocr_eval_vlm.txt
```

### 개요

현재 OCR 실행은 CLI에서 아래 명령으로 통합 운영합니다.

```bash
# WSL + NVIDIA GPU 환경에서만 필요
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
```

입력/출력 규칙:
- 이미지 입력: `data/v2/ocr_images/<doc_key>/img_001.jpg`
- GT 입력: `data/v2/ocr_eval/incoming_gt/<doc_key>.json`
- 결과 출력: `data/v2/ocr_eval/<doc_key>/{pred_raw,pred_structured,report}.json`
- `--doc-key`는 파일명이 아니라 `ocr_images` 하위 폴더명입니다.

### 문서 1개 실행

```bash
python -m src.cli ocr-run-doc \
  --doc-key "한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" \
  --ocr-config "configs/ocr_default.yaml"
```

실제 경로를 명시하려면:

```bash
cd /home/imella0707/personal/LLM_Project_team2
conda activate ocr_eval_vlm
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}   # WSL + NVIDIA GPU일 때만

python -m src.cli ocr-run-doc \
  --doc-key "한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보" \
  --images-root "/home/imella0707/personal/LLM_Project_team2/data/v2/ocr_images" \
  --gt-root "/home/imella0707/personal/LLM_Project_team2/data/v2/ocr_eval/incoming_gt" \
  --output-root "/home/imella0707/personal/LLM_Project_team2/data/v2/ocr_eval" \
  --ocr-config "/home/imella0707/personal/LLM_Project_team2/configs/ocr_default.yaml"
```

GT JSON에 `id`가 여러 개면 `--id`를 함께 지정하세요.

### 전체 문서 배치 실행

```bash
python -m src.cli ocr-run-batch \
  --ocr-config "configs/ocr_default.yaml"
```

실제 경로를 명시하려면:

```bash
cd /home/imella0707/personal/LLM_Project_team2
conda activate ocr_eval_vlm
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}   # WSL + NVIDIA GPU일 때만

python -m src.cli ocr-run-batch \
  --images-root "/home/imella0707/personal/LLM_Project_team2/data/v2/ocr_images" \
  --gt-root "/home/imella0707/personal/LLM_Project_team2/data/v2/ocr_eval/incoming_gt" \
  --output-root "/home/imella0707/personal/LLM_Project_team2/data/v2/ocr_eval" \
  --ocr-config "/home/imella0707/personal/LLM_Project_team2/configs/ocr_default.yaml"
```

### 결과 확인 위치

- OCR 이미지 추출 결과: `data/v2/ocr_images/`
- OCR 평가 산출물(`pred_raw`, `pred_structured`, `report`): `data/v2/ocr_eval/`
- `report.json` 주요 지표:
  - `type`, `status`, `latency_ms`
  - `text.char_similarity_pct`, `text.cer`, `text.wer`
  - `field_match.rate_pct` (매칭 개수/전체 개수)
  - `structure.<필드>.precision/recall/f1`