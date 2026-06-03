# 입찰메이트 RAG 시스템

공공입찰 RFP 문서를 대상으로 질의응답을 수행하는 시나리오 B(LLM API 기반) RAG baseline입니다.

## 프로젝트 설명

- 입력: PDF/HWP RFP 문서 + `data_list.csv`
- 처리: 파싱/청킹 -> 임베딩(`text-embedding-3-small`) -> Chroma 인덱싱
- 출력: 검색 근거 기반 답변 생성(`gpt-5-mini`) 및 평가 리포트
- CLI 구분:
  - `src/cli.py`: OpenAI API 기반 표준 실행 CLI
  - `src/cli_hug.py`: Hugging Face 모델 기반 로컬 실행 CLI

## 프로젝트 구조

```text
.
├── configs/                 # 실행/실험 설정
│   └── default.yaml
├── data/
│   ├── raw/                 # 원본 데이터(PDF/HWP + data_list.csv), git 업로드 금지
│   ├── v1/                  # Hugging Face 시나리오 데이터/산출물
│   └── v2/                  # OpenAI 시나리오 데이터/산출물
├── checkpoints/             # Chroma 인덱스, git 업로드 금지
├── outputs/                 # 평가 결과, git 업로드 금지
├── scripts/                 # 운영 보조 스크립트
├── src/
│   ├── cli.py               # OpenAI 전용 표준 CLI
│   ├── cli_hug.py           # Hugging Face 전용 CLI
│   ├── dataset/             # 문서/메타데이터 로더
│   ├── preprocessing/       # 텍스트 전처리/청킹
│   ├── models/              # OpenAI 모델 클라이언트
│   ├── engine/              # 검색/생성 엔진
│   └── evaluation/          # 평가 하네스/리포트/질문셋
└── docs/
    ├── operations.md        # 운영 파이프라인 상세
    ├── evaluation.md        # 평가/질문셋/리포트 상세
    └── ocr.md               # OCR 실행 상세
```

## 표준 실행 경로

### 시작 전 확인

- 권장 환경: Python 3.10+, Linux/WSL(Windows는 WSL 권장)
- 기본 문서는 OpenAI 경로(`src/cli.py`) 기준입니다.
- Hugging Face 로컬 실행이 필요하면 `src/cli_hug.py`를 사용합니다.

### 1) 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
```

`.env`:

```bash
OPENAI_API_KEY=...
```

원본 데이터:

```text
data/raw/문서파일.pdf
data/raw/문서파일.hwp
data/raw/data_list.csv
```

### 2) 사전 점검

```bash
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli check-setup
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli check-setup --check-openai
```

### 3) 인덱스 생성 -> 질의 -> 평가

```bash
# 인덱스 생성
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli ingest

# 질의
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli query "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘."

# 평가 (기본: data/v2/eval_questions.example.jsonl)
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli evaluate
```

성공 기준(최소):

- `ingest` 완료 후 `checkpoints/chroma_openai`에 인덱스 파일 생성
- `query` 실행 시 답변과 `Sources`가 함께 출력
- `evaluate` 실행 시 `outputs/` 아래 평가 결과 파일 생성

LangSmith 하네스 평가:

```bash
# .env: OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT=bidmate-rag-eval
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli evaluate-harness --config configs/default.yaml
```

## 상세 문서

- 운영 파이프라인(HWP + OCR 전체): [docs/operations.md](docs/operations.md)
- 평가/질문셋/리포트: [docs/evaluation.md](docs/evaluation.md)
- OCR 파이프라인 상세: [docs/ocr.md](docs/ocr.md)
- 공식 문서 참고:
  - [OpenAI Embeddings guide](https://platform.openai.com/docs/guides/embeddings)
  - [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses/create)

## 팀 협업 일지 (Notion)

- 박도원: [협업 일지 링크](https://www.notion.so/)
- 정신우: [협업 일지 링크](https://www.notion.so/)
- 김태민: [협업 일지 링크](https://www.notion.so/)
- 안수진: [협업 일지 링크](https://www.notion.so/)
- 김범수: [협업 일지 링크](https://www.notion.so/)


