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

HWP 파일을 처리해야 한다면 VM에서 선택 의존성을 추가로 설치합니다.

```bash
pip install -e ".[hwp]"
```

## HWP 파싱/청킹/임베딩 파이프라인

### 원클릭 실행(권장)

```bash
python -m src.cli run-pipeline \
  --input "data/raw/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시.hwp" \
  --output-dir "data/v2" \
  --dump-limit 20
```

기본 동작:
- 출력은 `data/v2/<sanitize된_파일명>/` 하위로 문서별 분리 저장됩니다.
- `--doc-id`는 선택 옵션이며, 폴더명에는 영향이 없고 Chroma 메타데이터의 `doc_id`에만 반영됩니다.
- Chroma 메타데이터 샘플 JSON이 자동 저장됩니다.
  - 파일명: `<원본파일명>_chroma_metadata_sample.json`
  - 개수: 기본 20개 (`--dump-limit`으로 변경)
  - 비활성화: `--no-dump-metadata-sample`

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

단계별 실행은 디버깅 또는 중간 산출물(prechunk/chunks/embedded) 확인이 필요한 경우에만 사용하고, 일반 실행은 원클릭 `run-pipeline`을 권장합니다.

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
  --model "text-embedding-3-small"

python -m src.cli build-chroma \
  --input "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/(사)부산국제영화제_2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원시_embedded.jsonl" \
  --index-dir "data/v2/(사)부산국제영화제_2024년_BIFF_ACFM_온라인서비스_재개발_및_행사지원시/chroma_index"
```

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

- OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses/create
