from __future__ import annotations

from typing import Any

from src.evaluation.answer import evaluate_answer_metrics
from src.evaluation.generation import evaluate_generation_metrics
from src.evaluation.retrieval import evaluate_retrieval_metrics


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [1.0 if row.get(key) else 0.0 for row in rows if row.get(key) is not None]
    return (sum(values) / len(values)) if values else None


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return (sum(values) / len(values)) if values else None


def _mean_int(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), int)]
    return (sum(values) / len(values)) if values else None


def summarize_harness_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(results),
        "mean_doc_hit": _mean_numeric(results, "doc_hit"),
        "mean_retrieval_keyword_hit": _mean_numeric(results, "retrieval_keyword_hit"),
        "mean_context_precision": _mean_numeric(results, "context_precision"),
        "mean_recall_at_5": _mean_numeric(results, "recall_at_5"),
        "mean_mrr": _mean_numeric(results, "mrr"),
        "mean_f_score": _mean_int(results, "f_score"),
        "mean_r_score": _mean_int(results, "r_score"),
        "mean_s_score": _mean_int(results, "s_score"),
        "mean_correctness_score": _mean_int(results, "correctness_score"),
        "mean_task_success": _mean_bool(results, "task_success"),
        "mean_wrong_refusal": _mean_bool(results, "wrong_refusal"),
        "mean_appropriate_refusal": _mean_bool(results, "appropriate_refusal"),
        "mean_total_latency_ms": _mean_numeric(results, "total_latency_ms"),
    }


def evaluate_row_metrics(
    row: dict[str, Any],
    *,
    answer: str,
    sources: list[Any],
    judge_model: str,
    run_llm_judge: bool = True,
    run_correctness_judge: bool = True,
) -> dict[str, Any]:
    """Run retrieval → (optional) generation judge → answer metrics for one harness row."""
    retrieval = evaluate_retrieval_metrics(row, sources)
    answer_metrics = evaluate_answer_metrics(
        row,
        answer,
        judge_model=judge_model,
        run_correctness_judge=run_correctness_judge,
        doc_hit=retrieval.get("doc_hit"),
        keyword_hit=retrieval.get("retrieval_keyword_hit"),
    )
    merged: dict[str, Any] = {**retrieval, **answer_metrics}
    if run_llm_judge:
        generation = evaluate_generation_metrics(
            str(row.get("question") or ""),
            sources,
            answer,
            judge_model=judge_model,
            run_llm_judge=True,
        )
        merged.update(generation)
    return merged
