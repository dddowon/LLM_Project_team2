"""답변 평가 지표 계산 (correctness, refusal, failure_reason)."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from openai import OpenAI

from src.models.openai_client import supports_chat_temperature

UNANSWERABLE_TYPES = frozenset({"unanswerable"})
RefusalKind = Literal["none", "full", "partial"]
MIN_SUBSTANTIVE_CHARS = 80
MIN_SUBSTANTIVE_LINE_CHARS = 12
_REFUSAL_CONTEXT_HINT = re.compile(
    r"관련\s*문서|기재된|명시|다음과\s*같|아래와\s*같|:\s*[^\s]",
    re.I,
)


def _has_refusal_phrase(text: str) -> bool:
    if "확인되지 않" in text or "확인할 수 없" in text or "존재하지 않" in text:
        return True
    if "제공된 문서" in text and any(token in text for token in ("없", "포함", "확인")):
        return True
    return False


def _line_is_refusal_boilerplate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if "확인되지 않" in stripped or "확인할 수 없" in stripped or "존재하지 않" in stripped:
        return True
    if "제공된 문서" in stripped and any(token in stripped for token in ("없", "포함", "확인")):
        return True
    if stripped.startswith("문서에서 확인") and len(stripped) < 72:
        return True
    if "추측하지 않" in stripped and len(stripped) < 96:
        return True
    if stripped.startswith("결론") and "확인되지" in stripped and len(stripped) < 120:
        return True
    return False


def normalize_for_answer_match(text: str) -> str:
    """공백·구두점 차이를 무시하고 기대 답안 포함 여부를 비교."""
    lowered = str(text or "").casefold()
    return re.sub(r"[\s\W_]+", "", lowered, flags=re.UNICODE)


def answer_contains_expected(expected: str, answer: str) -> bool:
    expected_norm = normalize_for_answer_match(expected)
    answer_norm = normalize_for_answer_match(answer)
    if not expected_norm or not answer_norm:
        return False
    if expected_norm in answer_norm:
        return True
    if re.fullmatch(r"\d+", expected_norm):
        return bool(re.search(rf"(?<!\d){re.escape(expected_norm)}(?!\d)", answer_norm))
    return False


def heuristic_correctness_score(expected: str, answer: str) -> int | None:
    """LLM judge 전에 명확히 맞는 답이면 5점. 애매하면 None → judge 호출."""
    expected = str(expected or "").strip()
    answer_text = str(answer or "").strip()
    if not expected or not answer_text:
        return None
    if answer_contains_expected(expected, answer_text):
        return 5
    exp_norm = normalize_for_answer_match(expected)
    if len(exp_norm) >= 8:
        # 긴 기대 답안: 핵심 토큰 대부분이 답변에 있으면 4점
        tokens = [t for t in re.findall(r"[가-힣a-z0-9]{2,}", expected, re.I) if len(t) >= 2]
        if len(tokens) >= 2:
            hits = sum(1 for t in tokens if normalize_for_answer_match(t) in normalize_for_answer_match(answer_text))
            if hits >= max(2, int(len(tokens) * 0.7)):
                return 4
    return None


def classify_answer_refusal(answer: str) -> RefusalKind:
    """전면 거절(full) vs 본문 답변 + 일부 부재 표기(partial) vs 없음(none)."""
    text = str(answer or "").strip()
    if not text:
        return "full"
    if not _has_refusal_phrase(text):
        return "none"

    substantive_chars = 0
    has_answer_line = False
    for line in text.splitlines():
        line = line.strip()
        if not line or _line_is_refusal_boilerplate(line):
            continue
        substantive_chars += len(line)
        if len(line) >= MIN_SUBSTANTIVE_LINE_CHARS and (
            _REFUSAL_CONTEXT_HINT.search(line) or not _has_refusal_phrase(line)
        ):
            has_answer_line = True

    if has_answer_line or substantive_chars >= MIN_SUBSTANTIVE_CHARS:
        return "partial"
    return "full"


def is_full_refusal_answer(answer: str) -> bool:
    return classify_answer_refusal(answer) == "full"


def is_refusal_answer(answer: str) -> bool:
    """전면 거절 여부. 부분 범위 제한(일부만 '문서에 없음' 표기)은 False."""
    return is_full_refusal_answer(answer)


def is_answerable_question_type(question_type: str | None) -> bool:
    return str(question_type or "").strip().lower() not in UNANSWERABLE_TYPES


def score_pass(score: int | None, *, threshold: int = 4) -> bool | None:
    if score is None:
        return None
    return int(score) >= threshold


def judge_answer_correctness(
    question: str,
    expected_answer: str,
    answer: str,
    *,
    model: str = "gpt-5-mini",
) -> dict[str, int | str]:
    expected = str(expected_answer or "").strip()
    if not expected:
        return {"correctness_score": 0, "judge_error": "missing expected_answer"}

    client = OpenAI()
    prompt = f"""당신은 RAG 평가자입니다. 모델 답변이 기대 답안의 핵심 요지를 맞췄는지 0~5로 채점하세요.
    - 5: 핵심 요지가 기대 답안과 일치
    - 4: 대체로 맞으나 사소한 누락
    - 3: 부분적으로만 맞음
    - 2: 관련은 있으나 핵심이 다름
    - 1: 거의 틀림
    - 0: 완전히 틀림 또는 환각

    "문서에서 확인되지 않습니다"만 하고 기대 답안 내용이 전혀 없으면 0~2점.
    기대 답안이 "문서에서 확인되지 않음" 이고 거절이 적절하면 5점.
    거절 문구 뒤·다른 줄에 기대 답안과 같은 사실(사업명, 숫자 등)이 있으면 4~5점.
    공백·띄어쓰기·'기반'/'기반' 붙여쓰기 등 사소한 표기 차이는 감점하지 마세요.

    [질문]
    {question}

    [기대 답안]
    {expected}

    [모델 답변]
    {answer}

    JSON만 출력: {{"correctness_score": 0~5 정수}}
    """
    try:
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if supports_chat_temperature(model):
            request["temperature"] = 0
        response = client.chat.completions.create(**request)
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        score = int(data.get("correctness_score", 0))
        return {"correctness_score": max(0, min(5, score))}
    except Exception as exc:
        return {"correctness_score": 0, "judge_error": str(exc)}


def classify_failure_reason(
    *,
    question_type: str | None,
    doc_hit: float | None,
    keyword_hit: float | None,
    is_refusal: bool,
    correctness_pass: bool | None,
    has_doc_id: bool,
) -> str | None:
    answerable = is_answerable_question_type(question_type)
    if not answerable:
        if is_refusal:
            return None
        return "should_refuse"
    if has_doc_id and doc_hit == 0.0 and correctness_pass is not True:
        return "wrong_doc"
    if is_refusal:
        return "wrong_refusal"
    if not is_refusal and correctness_pass is False:
        return "wrong_answer"
    if keyword_hit == 0.0:
        return "low_retrieval"
    return None


def evaluate_answer_metrics(
    row: dict[str, Any],
    answer: str,
    *,
    judge_model: str,
    run_correctness_judge: bool,
    doc_hit: float | None,
    keyword_hit: float | None,
) -> dict[str, Any]:
    question_type = str(row.get("question_type") or "").strip()
    refusal_kind = classify_answer_refusal(answer)
    full_refusal = refusal_kind == "full"
    partial_limitation = refusal_kind == "partial"

    correctness_score: int | None = None
    correctness_pass: bool | None = None
    correctness_error: str | None = None

    if run_correctness_judge:
        expected = str(row.get("expected_answer") or "").strip()
        if expected:
            heuristic = heuristic_correctness_score(expected, answer)
            if heuristic is not None:
                correctness_score = heuristic
                correctness_pass = score_pass(heuristic)
            else:
                payload = judge_answer_correctness(
                    str(row.get("question") or ""),
                    expected,
                    answer,
                    model=judge_model,
                )
                if isinstance(payload.get("correctness_score"), int):
                    correctness_score = int(payload["correctness_score"])
                    correctness_pass = score_pass(correctness_score)
                if payload.get("judge_error"):
                    correctness_error = str(payload["judge_error"])

    answerable = is_answerable_question_type(question_type)
    if answerable:
        if correctness_pass is None:
            task_success = not full_refusal
        else:
            task_success = (not full_refusal) and correctness_pass
    else:
        task_success = full_refusal

    failure_reason = classify_failure_reason(
        question_type=question_type,
        doc_hit=doc_hit,
        keyword_hit=keyword_hit,
        is_refusal=full_refusal,
        correctness_pass=correctness_pass,
        has_doc_id=bool(str(row.get("doc_id") or "").strip()),
    )

    return {
        "refusal_kind": refusal_kind,
        "partial_limitation": partial_limitation,
        "is_refusal": full_refusal,
        "correctness_score": correctness_score,
        "correctness_pass": correctness_pass,
        "task_success": task_success,
        "wrong_refusal": answerable and full_refusal,
        "appropriate_refusal": (not answerable) and full_refusal,
        "failure_reason": failure_reason,
        **({"correctness_judge_error": correctness_error} if correctness_error else {}),
    }
