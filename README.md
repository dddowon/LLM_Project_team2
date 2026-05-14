# 입찰메이트 RAG 시스템

공공입찰 RFP 문서를 대상으로 질의응답을 수행하는 시나리오 B(LLM API 기반) RAG baseline입니다.

## 목표

- PDF/HWP RFP 문서와 `data_list.csv` 메타데이터를 불러옵니다.
- 문서를 청킹하고 `text-embedding-3-small`로 임베딩합니다.
- FAISS vector DB를 생성합니다.
- 검색된 문서 근거를 바탕으로 `gpt-5-mini`가 답변합니다.
- 평가 질문셋으로 검색/생성 결과를 반복 비교합니다.

## 프로젝트 구조

```text
.
├── configs/                 # 실험 설정
│   └── default.yaml
├── data/
│   ├── v1/                  # 원본 데이터 위치, git 업로드 금지
│   │   ├── raw/             # PDF/HWP 파일 배치
│   │   └── data_list.csv    # 제공 메타데이터
│   └── v2/                  # 평가 질문셋 등 2차 가공 데이터
├── checkpoints/             # FAISS 인덱스, git 업로드 금지
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
data/v1/raw/문서파일.pdf
data/v1/raw/문서파일.hwp
data/v1/data_list.csv
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
cp data/v2/eval_questions.example.jsonl data/v2/eval_questions.jsonl
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli evaluate
```

HWP 파일을 처리해야 한다면 VM에서 선택 의존성을 추가로 설치합니다.

```bash
pip install -e ".[hwp]"
```

## 실험 포인트

- 청킹: `chunk_size`, `chunk_overlap`, 목차/장절 기반 의미 청킹 비교
- 검색: `top_k`, 메타데이터 필터링, MMR, hybrid search, re-ranking 비교
- 생성: 프롬프트, 답변 포맷, 답변 길이, 대화 히스토리 반영 방식 비교
- 평가: 단일 문서 정확도, 다중 문서 종합, 후속 질문 맥락 유지, 모르는 내용 거절 여부

## 참고한 공식 문서

- OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses/create
