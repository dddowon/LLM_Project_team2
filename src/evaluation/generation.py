"""생성 성능 지표 계산 (faithfulness, relevance, synthesis)."""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.evaluation.answer import is_refusal_answer
from src.models.openai_client import supports_chat_temperature


def _contexts_to_plain_text(contexts: list[Any], max_chars: int = 24_000) -> str:
    parts: list[str] = []
    for context in contexts:
        if isinstance(context, dict):
            parts.append(str(context.get("text", context.get("content", ""))))
        else:
            parts.append(str(context))
    return " ".join(parts)[:max_chars]


def judge_faithfulness_relevance(
    query: str,
    contexts: list[Any],
    answer: str,
    *,
    model: str = "gpt-5-mini",
) -> dict[str, int | str]:
    client = OpenAI()
    context_text = _contexts_to_plain_text(contexts)

    prompt = f"""당신은 RAG 시스템 평가 전문가입니다. 아래 내용을 바탕으로 점수를 매기세요.
    f_score (Faithfulness): 답변이 근거 문장에 기반하여 정직하게 작성되었는가?
    r_score (Relevance): 답변이 사용자의 질문에 얼마나 적절한가?
    s_score (Synthesis): 여러 근거를 종합해 질문 의도에 맞게 구조화된 답변을 만들었는가?

    [질문]: {query}
    [근거]: {context_text}
    [답변]: {answer}

    결과는 반드시 아래 JSON 형식으로만 출력하세요:
    {{"f_score": 점수(0~5 정수), "r_score": 점수(0~5 정수), "s_score": 점수(0~5 정수)}}
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
        return {
            "f_score": max(0, min(5, int(data.get("f_score", 0)))),
            "r_score": max(0, min(5, int(data.get("r_score", 0)))),
            "s_score": max(0, min(5, int(data.get("s_score", 0)))),
        }
    except Exception as exc:
        return {"f_score": 0, "r_score": 0, "s_score": 0, "judge_error": str(exc)}


def parse_judge_scores(result: dict[str, Any]) -> tuple[int, int, int, str | None]:
    err = result.get("judge_error")
    scores = (
        int(result.get("f_score", 0)),
        int(result.get("r_score", 0)),
        int(result.get("s_score", 0)),
    )
    if isinstance(err, str):
        return (*scores, err)
    return (*scores, None)


def evaluate_generation_metrics(
    question: str,
    contexts: list[Any],
    answer: str,
    *,
    judge_model: str,
    run_llm_judge: bool,
) -> dict[str, Any]:
    if not run_llm_judge:
        return {
            "f_score": None,
            "r_score": None,
            "s_score": None,
            "faithfulness_given_answer": None,
        }

    judge_payload = judge_faithfulness_relevance(
        question,
        contexts,
        answer,
        model=judge_model,
    )
    f_score, r_score, s_score, judge_err = parse_judge_scores(judge_payload)

    faithfulness_given_answer: int | None = None
    if not is_refusal_answer(answer):
        faithfulness_given_answer = f_score

    result: dict[str, Any] = {
        "f_score": f_score,
        "r_score": r_score,
        "s_score": s_score,
        "faithfulness_given_answer": faithfulness_given_answer,
    }
    if judge_err:
        result["judge_error"] = judge_err
    elif judge_payload.get("judge_error"):
        result["judge_error"] = judge_payload["judge_error"]
    return result
