# 평가/질문셋/리포트

## 기본 평가

```bash
# 기본 설정은 data/v2/eval_questions.example.jsonl 사용
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli evaluate
```

## LangSmith 하네스 평가

```bash
# .env: OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT=bidmate-rag-eval
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m src.cli evaluate-harness --config configs/default.yaml
```

- 결과 JSONL: `outputs/eval_harness_results.jsonl`
- 트레이스: [https://smith.langchain.com](https://smith.langchain.com)

## 운영 파이프라인 기준 평가 절차

### 1) 평가 질문 입력 생성

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

### 2) OpenAI 질문셋 생성

```bash
python -m src.evaluation.generate_eval_questions \
  --call-openai \
  --generation-input data/v2/eval_question_generation_inputs.jsonl \
  --eval-output data/v2/eval_questions.jsonl \
  --model gpt-5-mini \
  --overwrite
```

### 3) 하네스 실행

```bash
python -m src.cli --config configs/default.yaml evaluate-harness \
  --output outputs/eval_harness_results.jsonl \
  --judge-model gpt-5-mini \
  --no-langsmith-feedback
```

`--no-langsmith-feedback`는 로컬 결과에는 영향 없고, LangSmith feedback 업로드 시 score 범위 이슈를 피할 때 사용합니다.

### 4) 리포트 생성

```bash
python -m src.evaluation.build_eval_report \
  --input outputs/eval_harness_results.jsonl \
  --html-output outputs/eval_report.html \
  --failures-output outputs/eval_failures.csv \
  --successes-output outputs/eval_successes.csv \
  --top-n 500
```

- `outputs/eval_report.html` 중심으로 점수/오답 분석
- 실패/성공 샘플은 CSV로 후속 분석

## 실험 포인트

- 청킹: `chunk_size`, `chunk_overlap`, 목차/장절 기반 의미 청킹
- 검색: `top_k`, 메타데이터 필터링, MMR, hybrid search, re-ranking
- 생성: 프롬프트, 답변 포맷, 답변 길이, 대화 히스토리 반영
- 평가: 단일 문서 정확도, 다중 문서 종합, 후속 질문 맥락 유지, 모르는 내용 거절
