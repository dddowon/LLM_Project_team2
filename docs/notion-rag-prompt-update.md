# RAG 답변 프롬프트 개선 (eval 대응)

**작업일:** 2026-05-26  
**목적:** `wrong_refusal`·`partial_limitation`·`comparison` 저성능 완화를 위한 생성 단계 프롬프트·eval 연동

---

## 요약

| 구분 | 내용 |
|------|------|
| 변경 파일 | `src/engine/prompts.py`, `src/engine/rag.py`, `src/evaluation/langsmith_harness.py` |
| 검색/인덱스 | 변경 없음 (재인덱싱 불필요) |
| eval 재실행 | **답변 생성부터** 다시 돌려야 반영됨 |

---

## 1. 공통 정책 (`SYSTEM_POLICY`) — `[근거]` 추가

**추가 문구**

> 컨텍스트에 질문과 관련된 내용이 있으면, 답 전체를 "확인 불가"로 거절하지 말고 확인된 범위만 답하세요.

**왜**

- 검색은 됐는데 답이 전부 "문서에서 확인되지 않습니다"만 나오는 **전면 거절(`wrong_refusal`)** 완화
- eval은 본문 실질 내용이 80자 미만이면 `full` refusal로 분류함 (`answer.py`)

**기대 지표**

- `wrong_refusal` ↓
- `task_success` ↑ (전면 거절이면 correctness와 무관하게 실패할 수 있음)

**이미 있던 규칙 (유지)**

- 질문에 없는 항목 나열 금지
- "문서에서 확인되지 않습니다"는 **사용자가 명시적으로 물은 항목**에만

---

## 2. 질문 유형별 출력 힌트 (`QUESTION_TYPE_HINTS`)

eval JSONL의 `question_type`에 따라 프롬프트에 짧은 블록을 추가함.

### `comparison` (비교)

- 출력 순서 고정: **(1) A 요약 (2) B 요약 (3) 공통점 (4) 차이점**
- 질문에 없는 예산·일정·제출방식 등 추가 금지
- A/B 중 없는 쪽만, 질문에 나온 대상에 한해 미확인 표기

**기대:** `comparison` `task_success` ↑, LLM judge `s_score` ↑

### `requirement_detail` (요구사항 상세)

- 해당 요구사항의 **내용·조건·수치만** 답변
- 예산·일정·제출방식·사업개요 등 질문에 없는 필드 자동 확장 금지

**기대:** `partial_limitation` ↓, `wrong_answer` ↓

### 그 외 유형 (`fact`, `summary` 등)

- 추가 블록 없음 → 기존 `SYSTEM_POLICY`만 적용

---

## 3. 코드 연동 (데이터 흐름)

```
eval_questions.jsonl (question_type)
    → langsmith_harness._eval_one_row
    → RagEngine.answer(..., question_type=...)
    → build_rag_prompt(..., question_type=...)
    → LLM generate
```

| 파일 | 변경 |
|------|------|
| `prompts.py` | `QUESTION_TYPE_HINTS`, `question_type_addon()`, `build_rag_prompt(question_type=...)` |
| `rag.py` | `answer(..., question_type=None)` 선택 인자 |
| `langsmith_harness.py` | row의 `question_type`을 `engine.answer`에 전달 |

**CLI / 일반 채팅**

- `question_type` 미전달 시 동작은 **이전과 동일** (유형 힌트 없음)

---

## 4. 이번에 하지 않은 것

- 리랭킹, BM25 하이브리드
- 청킹·재임베딩
- `score_threshold` / `top_k` 튜닝 (`configs/default.yaml`는 별도 실험)
- `ground_truth_keywords` / `expected_answer` 정제

---

## 5. 재평가 시 체크리스트

1. eval harness로 **답변 재생성** (기존 `eval_results.jsonl`만 재채점하면 프롬프트 미반영)
2. 리포트에서 확인:
   - 전체 `mean_wrong_refusal`
   - `question_type = comparison` 구간 `task_success`
   - `partial_limitation` 건수
   - `s_score` (comparison 필터 권장)
3. (선택) 검색 튜닝과 A/B: `score_threshold` 0.25, `top_k` 8 등

---

## 6. 변경 파일 diff 요약

- **`src/engine/prompts.py`**: 전면 거절 방지 1줄 + 유형별 힌트 dict + `question_type` 파라미터
- **`src/engine/rag.py`**: `question_type` 전달
- **`src/evaluation/langsmith_harness.py`**: eval row → `question_type` 전달
