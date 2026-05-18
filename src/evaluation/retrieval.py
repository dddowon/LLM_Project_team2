from __future__ import annotations

from typing import Any


def context_text(context: Any) -> str:
    if isinstance(context, dict):
        return str(context.get("text", context.get("content", "")))
    return str(context)


def contains_keyword(text: str, ground_truth_keywords: list[str]) -> bool:
    return any(keyword in text for keyword in ground_truth_keywords)


def evaluate_retrieval(contexts: list[Any], ground_truth_keywords: list[str]) -> float:
    if not ground_truth_keywords:
        return 0.0
    all_text = " ".join(context_text(context) for context in contexts)
    is_hit = contains_keyword(all_text, ground_truth_keywords)
    return 1.0 if is_hit else 0.0


def evaluate_context_precision(contexts: list[Any], ground_truth_keywords: list[str]) -> float:
    if not contexts or not ground_truth_keywords:
        return 0.0
    relevant_count = sum(
        1 for context in contexts if contains_keyword(context_text(context), ground_truth_keywords)
    )
    return relevant_count / len(contexts)
