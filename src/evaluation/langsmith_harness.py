from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

from src.config import load_config
from src.engine.rag import RagEngine
from src.engine.vector_store import ChromaVectorStore
from src.evaluation.llm_judge import judge_faithfulness_relevance, parse_judge_scores
from src.evaluation.harness_metrics import summarize_harness_results
from src.evaluation.retrieval import evaluate_context_precision, evaluate_retrieval
from src.utils.jsonl import read_jsonl, write_jsonl


def _feedback_scores_safe(client: Client | None, scores: dict[str, float]) -> None:
    if client is None:
        return
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    run_id = run_tree.id
    for key, score in scores.items():
        if score is None:
            continue
        try:
            client.create_feedback(run_id=run_id, key=key, score=float(score))
        except Exception:
            continue


@traceable(name="llm_judge", run_type="llm", tags=["rag-harness", "judge"])
def _traced_judge(
    query: str,
    contexts: list[Any],
    answer: str,
    *,
    judge_model: str,
    client: Client | None,
) -> dict[str, int | str]:
    out = judge_faithfulness_relevance(query, contexts, answer, model=judge_model)
    f_score, r_score, s_score, _err = parse_judge_scores(out)
    _feedback_scores_safe(
        client,
        {
            "faithfulness_0_1": f_score / 5.0,
            "relevance_0_1": r_score / 5.0,
            "synthesis_0_1": s_score / 5.0,
        },
    )
    return out


@traceable(name="rag_eval_row", run_type="chain", tags=["rag-harness"])
def _eval_one_row(
    engine: RagEngine,
    row: dict[str, Any],
    *,
    judge_model: str,
    run_llm_judge: bool,
    langsmith_client: Client | None,
) -> dict[str, Any]:
    question = row["question"]
    keywords = row.get("ground_truth_keywords")
    if keywords is not None and not isinstance(keywords, list):
        keywords = None
    keyword_list = [str(k) for k in keywords] if keywords else []

    start = time.perf_counter()
    result = engine.answer(question, include_source_text=True)
    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
    contexts: list[Any] = list(result.get("sources") or [])

    retrieval_hit: float | None = None
    context_precision: float | None = None
    if keyword_list:
        retrieval_hit = evaluate_retrieval(contexts, keyword_list)
        context_precision = evaluate_context_precision(contexts, keyword_list)

    judge_payload: dict[str, int | str] = {}
    if run_llm_judge:
        judge_payload = _traced_judge(
            question,
            contexts,
            str(result.get("answer", "")),
            judge_model=judge_model,
            client=langsmith_client,
        )

    f_score, r_score, s_score, judge_err = parse_judge_scores(judge_payload)

    merged: dict[str, Any] = {
        **row,
        "answer": result.get("answer"),
        "sources": result.get("sources"),
        "retrieval_keyword_hit": retrieval_hit,
        "context_precision": context_precision,
        "f_score": f_score if run_llm_judge else None,
        "r_score": r_score if run_llm_judge else None,
        "s_score": s_score if run_llm_judge else None,
        "total_latency_ms": total_latency_ms,
    }
    if judge_err:
        merged["judge_error"] = judge_err
    elif judge_payload.get("judge_error"):
        merged["judge_error"] = judge_payload["judge_error"]

    _feedback_scores_safe(
        langsmith_client,
        {
            **({"retrieval_keyword_hit": retrieval_hit} if retrieval_hit is not None else {}),
            **({"context_precision": context_precision} if context_precision is not None else {}),
            "total_latency_ms": total_latency_ms,
            **(
                {
                    "faithfulness_0_1": f_score / 5.0,
                    "relevance_0_1": r_score / 5.0,
                    "synthesis_0_1": s_score / 5.0,
                }
                if run_llm_judge
                else {}
            ),
        },
    )
    return merged


@traceable(name="rag_eval_batch", run_type="chain", tags=["rag-harness", "batch"])
def _eval_batch_traced(
    engine: RagEngine,
    rows: list[dict[str, Any]],
    *,
    judge_model: str,
    run_llm_judge: bool,
    langsmith_client: Client | None,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        outputs.append(
            _eval_one_row(
                engine,
                row,
                judge_model=judge_model,
                run_llm_judge=run_llm_judge,
                langsmith_client=langsmith_client,
                langsmith_extra={"metadata": {"row_index": i}},
            )
        )
    return outputs


def run_eval_harness(
    config_path: str | Path,
    *,
    evaluation_set: Path | None = None,
    output_path: Path | None = None,
    judge_model: str = "gpt-5-mini",
    run_llm_judge: bool = True,
    langsmith_feedback: bool = True,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    eval_path = evaluation_set or config.paths.evaluation_set
    out_path = output_path or Path("outputs/eval_harness_results.jsonl")

    rows = read_jsonl(eval_path)
    if not rows:
        raise RuntimeError(f"평가 질문셋이 없습니다: {eval_path}")

    store = ChromaVectorStore.load(config.paths.index_dir)
    engine = RagEngine(config, store)

    ls_client: Client | None = None
    if langsmith_feedback and (
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    ):
        try:
            ls_client = Client()
        except Exception:
            ls_client = None

    results = _eval_batch_traced(
        engine,
        rows,
        judge_model=judge_model,
        run_llm_judge=run_llm_judge,
        langsmith_client=ls_client,
        langsmith_extra={"metadata": {"evaluation_set": str(eval_path)}},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, results)

    summary = summarize_harness_results(results)
    _feedback_scores_safe(
        ls_client,
        {
            k: float(v)
            for k, v in summary.items()
            if v is not None and k != "n" and isinstance(v, (int, float))
        },
    )
    return out_path, summary
