from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


def _contexts_to_plain_text(contexts: list[Any], max_chars: int = 24_000) -> str:
    parts: list[str] = []
    for c in contexts:
        if isinstance(c, dict):
            parts.append(str(c.get("text", c.get("content", ""))))
        else:
            parts.append(str(c))
    joined = " ".join(parts)
    return joined[:max_chars]


def judge_faithfulness_relevance(
    query: str,
    contexts: list[Any],
    answer: str,
    *,
    model: str = "gpt-4o",
) -> dict[str, int | str]:
    client = OpenAI()
    context_text = _contexts_to_plain_text(contexts)

    prompt = f"""당신은 RAG 시스템 평가 전문가입니다. 아래 내용을 바탕으로 점수를 매기세요.
    f_score (Faithfulness): 답변이 근거 문장에 기반하여 정직하게 작성되었는가?
    r_score (Relevance): 답변이 사용자의 질문에 얼마나 적절한가?

    [질문]: {query}
    [근거]: {context_text}
    [답변]: {answer}

    결과는 반드시 아래 JSON 형식으로만 출력하세요:
    {{"f_score": 점수(0~5 정수), "r_score": 점수(0~5 정수)}}
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        f_score = int(data.get("f_score", 0))
        r_score = int(data.get("r_score", 0))
        return {
            "f_score": max(0, min(5, f_score)),
            "r_score": max(0, min(5, r_score)),
        }
    except Exception as exc:
        return {"f_score": 0, "r_score": 0, "judge_error": str(exc)}


def parse_judge_scores(result: dict[str, Any]) -> tuple[int, int, str | None]:
    err = result.get("judge_error")
    if isinstance(err, str):
        return int(result.get("f_score", 0)), int(result.get("r_score", 0)), err
    return int(result.get("f_score", 0)), int(result.get("r_score", 0)), None
